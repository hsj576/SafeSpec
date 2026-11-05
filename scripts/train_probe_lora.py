"""Phase 3 route B: LoRA-fine-tune Qwen3Guard-Stream for binary hijack detection.

Design decisions (locked in Phase 3 kickoff):
    * freeze the Qwen3 backbone + the native risk_level/category heads
    * add LoRA (r=16, alpha=32) on q/k/v/o projections of the backbone
    * add a NEW binary head: Linear(hidden_size, 1) on top of the last-token
      hidden state; this is the "AgentHijack" logit we train
    * loss: BCE with class-weighted positive samples
    * input encoding:
        [system prompt from AGENT_SYSTEM_PROMPT]
        <user turn (merged prompt including payload)>
        <assistant: output_text>
      i.e. the FULL conversation including the agent's output.
    * batch size 1 (variable prompt length), gradient accumulation 8
    * ~3 epochs on 225 train samples -> ~85 steps/epoch

Outputs:
    experiment/outputs/phase3_lora_<ts>/      checkpoint dir
      ├── adapter_model.safetensors            LoRA weights
      ├── hijack_head.pt                        the new binary head
      ├── training.json                         loss curve + eval metrics
      └── config.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    raise SystemExit("peft is required: pip install peft")

from safespec.common import (DEVICE_PROBE, DTYPE, EXPERIMENT_ROOT, LOGS_DIR,
                             OUTPUTS_DIR, PROBE_MODEL_PATH, get_logger,
                             human_timestamp)

TS = human_timestamp()
DATA_DIR = EXPERIMENT_ROOT / "data"
log = get_logger("probe_lora", log_file=LOGS_DIR / f"probe_lora_{TS}.log")


# ---------------------------------------------------------------------------
# Model wrapper: freeze backbone + heads, add LoRA + a new hijack head
# ---------------------------------------------------------------------------
class HijackProbeModel(nn.Module):
    """Qwen3Guard-Stream + LoRA + binary hijack head."""

    def __init__(self, base_model_path: str, device: str, dtype: torch.dtype,
                 lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        super().__init__()
        log.info("Loading Qwen3Guard-Stream from %s", base_model_path)
        self.base = AutoModel.from_pretrained(
            base_model_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)
        self.device = torch.device(device)
        self.dtype = dtype
        self.hidden_size = self.base.config.hidden_size

        # Freeze everything (base backbone + original heads)
        for p in self.base.parameters():
            p.requires_grad = False
        log.info("Base model frozen. hidden_size=%d", self.hidden_size)

        # Attach LoRA to the backbone's q/k/v/o projections
        # NOTE: Qwen3's modules under self.base.model.layers[i].self_attn
        # Our target modules follow the standard transformers naming.
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        # Important: wrap base_model's self.model, which is a plain Qwen3Model.
        # The outer Qwen3ForGuardModel has non-standard heads peft won't understand.
        self.base.model = get_peft_model(self.base.model, lora_cfg)
        trainable = sum(p.numel() for p in self.base.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.base.model.parameters())
        log.info("LoRA attached. Trainable params in backbone: %d / %d (%.3f%%)",
                 trainable, total, 100 * trainable / total)

        # New hijack head (trainable)
        self.hijack_head = nn.Linear(self.hidden_size, 1).to(device=self.device, dtype=dtype)
        nn.init.zeros_(self.hijack_head.bias)
        log.info("Added hijack head: Linear(%d, 1)", self.hidden_size)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Return a scalar logit per sample (shape (B,)).

        We score at the LAST token position (standard for sequence classification
        with causal LMs).  No KV cache -- training mode re-runs the full seq.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        # self.base.model is now the PEFT-wrapped Qwen3Model
        out = self.base.model(input_ids=input_ids.to(self.device),
                              use_cache=False)
        hs = out.last_hidden_state      # (B, T, H)
        last = hs[:, -1, :]              # (B, H)
        logits = self.hijack_head(last).squeeze(-1)   # (B,)
        return logits


# ---------------------------------------------------------------------------
# Data loading & encoding
# ---------------------------------------------------------------------------
@dataclass
class EncodedSample:
    input_ids: torch.LongTensor
    label: int
    split: str
    sample_id: str


def load_split_rows(path: Path, split: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["label"] < 0:
                continue
            if r["split"] != split:
                continue
            rows.append(r)
    return rows


def encode_row(tokenizer, row: dict, max_len: int = 4096) -> torch.LongTensor:
    """Apply chat template + append the assistant output, token-encode once."""
    messages = list(row["prompt_messages"])
    out_text = row["output_text"] or ""
    # append the model's output as the final assistant turn (NOT a generation prompt).
    messages.append({"role": "assistant", "content": out_text})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                   max_length=max_len).input_ids[0]
    return ids


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def eval_split(model: HijackProbeModel, tokenizer, rows: List[dict],
               max_len: int) -> Dict:
    model.eval()
    probs: List[float] = []
    labels: List[int] = []
    with torch.inference_mode():
        for r in rows:
            ids = encode_row(tokenizer, r, max_len=max_len)
            logit = model(ids)
            p = torch.sigmoid(logit.float()).item()
            probs.append(p)
            labels.append(r["label"])
    model.train()
    # best threshold under FPR cap = 0.1 (same convention as Fuser)
    import numpy as np
    probs = np.array(probs)
    labels = np.array(labels)
    # sweep
    taus = np.unique(np.concatenate([probs, np.array([0.0, 1.0])]))
    best = None
    for t in taus:
        pred = (probs > t).astype(np.int32)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        npos = tp + fn
        nneg = fp + tn
        fnr = fn / max(1, npos)
        fpr = fp / max(1, nneg)
        if fpr <= 0.1:
            score = fnr + fpr
            if best is None or score < best[0]:
                best = (score, float(t), fnr, fpr, tp, fn, fp, tn)
    # also compute AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")
    if best is None:
        return {"auc": auc, "best": None}
    _, tau, fnr, fpr, tp, fn, fp, tn = best
    return {"auc": auc,
            "tau": tau, "fnr": fnr, "fpr": fpr,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "probs": probs.tolist(), "labels": labels.tolist()}


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA_DIR / "phase3_dataset.jsonl"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="Positive-class weight in BCE (train pos:neg ≈ 1.8:1 already)")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Eval on dev every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)

    log.info("=" * 60)
    log.info("Phase 3 route B: LoRA fine-tune Qwen3Guard-Stream for hijack detection")
    log.info("=" * 60)
    for k, v in vars(args).items():
        log.info("  %-16s = %s", k, v)

    # ---- data ----
    log.info("Loading dataset from %s", args.data)
    tokenizer = AutoTokenizer.from_pretrained(str(PROBE_MODEL_PATH), trust_remote_code=True)
    train_rows = load_split_rows(Path(args.data), "train")
    dev_rows = load_split_rows(Path(args.data), "dev")
    test_rows = load_split_rows(Path(args.data), "test")
    log.info("  train=%d dev=%d test=%d", len(train_rows), len(dev_rows), len(test_rows))
    n_pos = sum(1 for r in train_rows if r["label"] == 1)
    log.info("  train pos:neg = %d:%d", n_pos, len(train_rows) - n_pos)

    # ---- model ----
    model = HijackProbeModel(str(PROBE_MODEL_PATH),
                             device=DEVICE_PROBE, dtype=DTYPE,
                             lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    model.train()

    # Collect trainable params
    train_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in train_params)
    log.info("Trainable params (total): %d (%.2f M)", n_train, n_train / 1e6)

    optim = AdamW(train_params, lr=args.lr, weight_decay=0.0)
    pos_weight = torch.tensor([args.pos_weight], device=DEVICE_PROBE, dtype=DTYPE)

    # ---- training ----
    history = {"losses": [], "dev": [], "test": None, "args": vars(args)}

    step = 0
    for epoch in range(args.epochs):
        random.shuffle(train_rows)
        epoch_loss = 0.0
        epoch_steps = 0
        t_ep = time.perf_counter()

        optim.zero_grad(set_to_none=True)
        accum_count = 0
        running_loss = 0.0
        for i, r in enumerate(train_rows):
            ids = encode_row(tokenizer, r, max_len=args.max_len)
            label = torch.tensor([r["label"]], dtype=DTYPE, device=DEVICE_PROBE)
            logit = model(ids)
            loss = F.binary_cross_entropy_with_logits(
                logit.float(), label.float(), pos_weight=pos_weight.float(),
            )
            (loss / args.grad_accum).backward()
            running_loss += loss.item()
            accum_count += 1

            if accum_count >= args.grad_accum or i == len(train_rows) - 1:
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                avg_loss = running_loss / accum_count
                epoch_loss += running_loss
                epoch_steps += accum_count
                history["losses"].append({"epoch": epoch, "step": step,
                                           "sample_idx": i, "loss": avg_loss})
                if step % 5 == 0:
                    log.info("[epoch %d step %d i=%d/%d] loss=%.4f",
                             epoch, step, i + 1, len(train_rows), avg_loss)
                accum_count = 0
                running_loss = 0.0

        avg_ep = epoch_loss / max(1, epoch_steps)
        ep_elapsed = time.perf_counter() - t_ep
        log.info("[epoch %d done] avg_loss=%.4f elapsed=%.1fs", epoch, avg_ep, ep_elapsed)

        # eval on dev
        if (epoch + 1) % args.eval_every == 0:
            log.info("[eval dev]")
            dev_m = eval_split(model, tokenizer, dev_rows, args.max_len)
            log.info("  AUC=%.3f  best(fpr<=.1): FNR=%.3f FPR=%.3f tau=%.3f",
                     dev_m.get("auc", float("nan")),
                     dev_m.get("fnr", float("nan")),
                     dev_m.get("fpr", float("nan")),
                     dev_m.get("tau", float("nan")))
            dev_m.pop("probs", None)
            dev_m.pop("labels", None)
            history["dev"].append({"epoch": epoch, **dev_m})

    # ---- final test ----
    log.info("[eval test]")
    test_m = eval_split(model, tokenizer, test_rows, args.max_len)
    log.info("  AUC=%.3f  best(fpr<=.1): FNR=%.3f FPR=%.3f tau=%.3f",
             test_m.get("auc", float("nan")),
             test_m.get("fnr", float("nan")),
             test_m.get("fpr", float("nan")),
             test_m.get("tau", float("nan")))
    test_m_full = dict(test_m)
    test_m.pop("probs", None)
    test_m.pop("labels", None)
    history["test"] = test_m

    # ---- save ----
    out_dir = OUTPUTS_DIR / f"phase3_lora_{TS}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # peft adapter weights
    model.base.model.save_pretrained(str(out_dir))
    # hijack head weights
    torch.save(model.hijack_head.state_dict(), out_dir / "hijack_head.pt")
    # training log
    with open(out_dir / "training.json", "w") as f:
        json.dump(history, f, indent=2)
    # test probs for later analysis
    with open(out_dir / "test_predictions.json", "w") as f:
        json.dump({"test_sample_ids": [r["sample_id"] for r in test_rows],
                   "probs": test_m_full.get("probs"),
                   "labels": test_m_full.get("labels")}, f, indent=2)
    log.info("Saved all artefacts to %s", out_dir)


if __name__ == "__main__":
    main()

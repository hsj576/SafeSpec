"""Batched streaming safety probe based on Qwen3Guard-Stream-0.6B.

The upstream `stream_moderate_from_ids` is a coroutine wrapper that forces the
caller to feed tokens one-by-one.  Its underlying `forward()` is a plain Qwen3
backbone + two sets of classification heads, so it accepts arbitrary-length
input tensors natively.  This file exposes a thin wrapper that *bypasses* the
coroutine protocol and lets us score `k` tokens in a single forward pass.

Key API:

    probe = BatchedGuardProbe(model_path, device)
    # prefill the user turn (returns per-token risk for the user turn)
    user_result = probe.score_user_turn(user_token_ids)
    # for each speculative round, score k newly-proposed assistant tokens:
    asst_result = probe.score_assistant_tokens(draft_token_ids)   # k tokens
    # ... iterate ...
    probe.reset()    # drop KV cache & start a new conversation

Returns are dicts whose values are *per-token lists* of length k:
    {
        "risk_level":    List[str],       # Safe / Unsafe / Controversial
        "risk_prob":     List[float],
        "category":      List[str],
        "category_prob": List[float],
        "unsafe_prob":   List[float],     # convenience: P(risk != Safe)
    }

Both aggregated and per-token signals are useful for the async-detection path:
    * aggregated: `probe.max_unsafe_prob` gives a running maximum, useful for
      early-stop logic in the intercept-mode experiment.
    * per-token: lets us pinpoint which draft token triggered the alarm.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .common import DEVICE_PROBE, DTYPE, PROBE_MODEL_PATH, CudaTimer, get_logger

log = get_logger("probe_stream")


# ---------------------------------------------------------------------------
# BatchedGuardProbe
# ---------------------------------------------------------------------------
class BatchedGuardProbe:
    """Wrap Qwen3Guard-Stream-0.6B to expose a k-token batched API.

    Optional Phase-3 extension: if `lora_dir` is given, loads a trained LoRA
    adapter + binary hijack head on top of the backbone.  Each forward then
    additionally emits `lora_hijack_prob` (per-token sigmoid of the hijack head
    output).  Note that activating LoRA slightly perturbs the backbone hidden
    states, which in turn perturbs the native risk_logits / category_logits --
    acceptable because our Phase 3 B-route conclusion already shows that the
    native (zero-shot) outputs are unreliable; the LoRA head is the authoritative
    signal in this mode.

    Phase 4a extensions:
      * `use_compile`:  torch.compile the backbone forward (mode="default",
                        inductor-fusion only).  Full CUDA-Graph speedup
                        (mode="reduce-overhead") would require migrating from
                        DynamicCache to StaticCache -- out of 4a scope.
      * `use_stream`:   execute the probe forward on a dedicated non-default
                        CUDA stream so that from the host's perspective its
                        launch does not block the target-GPU work on the main
                        thread.  Combined with pinned-memory async copies this
                        lets target.verify (GPU0) and probe (GPU2) truly
                        overlap even on CPython / GIL.
    """

    SAFE_LABEL = "Safe"

    def __init__(
        self,
        model_path: str = str(PROBE_MODEL_PATH),
        device: str = DEVICE_PROBE,
        dtype: torch.dtype = DTYPE,
        lora_dir: Optional[str] = None,
        use_compile: bool = False,
        use_stream: bool = False,
        compile_shapes: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype

        log.info("Loading probe tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        log.info("Loading Qwen3Guard-Stream on %s (%s)", self.device, dtype)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.device).eval()

        # Optional LoRA adapter + hijack head (Phase 3 route B)
        self.has_lora = False
        self.hijack_head: Optional[nn.Linear] = None
        if lora_dir is not None:
            try:
                from peft import PeftModel
            except ImportError:
                raise RuntimeError("peft is required when lora_dir is set")
            log.info("Attaching LoRA adapter from %s", lora_dir)
            self.model.model = PeftModel.from_pretrained(self.model.model, lora_dir)
            self.model.model.eval()
            hidden_size = self.model.config.hidden_size
            self.hijack_head = nn.Linear(hidden_size, 1).to(device=self.device,
                                                              dtype=dtype)
            head_state = torch.load(f"{lora_dir}/hijack_head.pt",
                                    map_location=self.device, weights_only=True)
            self.hijack_head.load_state_dict(head_state)
            self.hijack_head.eval()
            self.has_lora = True
            log.info("LoRA + hijack_head loaded.  Probe now in LoRA mode.")

        self.kv: Optional[DynamicCache] = None
        self.max_unsafe_prob: float = 0.0
        self.max_unsafe_category: Optional[str] = None
        # running max of the LoRA hijack prob (if LoRA attached)
        self.max_lora_hijack_prob: float = 0.0

        # ---- Phase 4a: optional CUDA stream for true cross-GPU overlap ----
        self.use_stream = bool(use_stream)
        self.stream: Optional[torch.cuda.Stream] = None
        if self.use_stream:
            self.stream = torch.cuda.Stream(device=self.device)
            log.info("Probe will run on a dedicated CUDA stream on %s", self.device)

        # ---- Phase 4a: optional torch.compile for CUDA Graph speedup ----
        # We maintain a mapping (shape_len -> compiled_callable).  Each entry is
        # compiled on first use via a warmup forward; subsequent calls dispatch
        # the pre-compiled graph.  We keep the raw self.model.forward as a
        # fallback for unexpected shapes (e.g. the user-turn prefill).
        self.use_compile = bool(use_compile)
        self._compiled_shapes: Dict[int, object] = {}
        self._compile_warmup_done: Dict[int, bool] = {}
        if self.use_compile:
            log.info("torch.compile ENABLED for probe backbone "
                     "(mode=default, inductor fusion only -- CUDA Graphs would "
                     "require StaticCache which is out of 4a scope)")

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop any KV cache + running statistics -- start a new conversation."""
        self.kv = None
        self.max_unsafe_prob = 0.0
        self.max_unsafe_category = None
        self.max_lora_hijack_prob = 0.0

    # ------------------------------------------------------------------
    def _run_backbone(self, token_ids: torch.LongTensor, logits_to_keep: int):
        """One probe forward.  If use_compile is set and the input length has
        a cached compiled callable, dispatch to it; otherwise fall back to the
        eager path.  Must be called inside torch.inference_mode.

        We use mode="default" (inductor fusion only, no CUDA Graphs) because
        DynamicCache mutates its internal tensors across forwards, which is
        incompatible with the static-buffer requirement of CUDA Graphs.
        Fusion alone typically yields ~15-25% kernel-time reduction on small
        LMs like Qwen3-Guard-0.6B.
        """
        need_hs = self.has_lora
        if self.use_compile:
            T = int(token_ids.shape[1])
            compiled = self._compiled_shapes.get(T)
            if compiled is None:
                def _fn(ids, kv):
                    return self.model.forward(
                        input_ids=ids,
                        past_key_values=kv,
                        use_cache=True,
                        logits_to_keep=logits_to_keep,
                        return_dict=True,
                        output_hidden_states=need_hs,
                    )
                compiled = torch.compile(_fn, mode="default",
                                          fullgraph=False, dynamic=False)
                self._compiled_shapes[T] = compiled
                self._compile_warmup_done[T] = False
            out = compiled(token_ids, self.kv)
            self._compile_warmup_done[T] = True
            return out
        return self.model.forward(
            input_ids=token_ids,
            past_key_values=self.kv,
            use_cache=True,
            logits_to_keep=logits_to_keep,
            return_dict=True,
            # When LoRA is attached, we still need hidden states for the head.
            # The current eager HF Qwen3ForGuardModel exposes last_hidden_state
            # via outputs.hidden_states only when output_hidden_states=True.
            output_hidden_states=need_hs,
        )

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _forward(self, token_ids: torch.LongTensor) -> dict:
        """Run one forward pass; return the model output's logit dict + updated cache.

        When LoRA is attached, we also request hidden_states so the hijack head
        can be applied downstream without re-running the backbone.
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        # pinned-memory async copy if the tensor is on CPU
        if token_ids.device != self.device:
            token_ids = token_ids.to(self.device, non_blocking=True)
        if self.kv is None:
            self.kv = DynamicCache()
        logits_to_keep = int(token_ids.shape[1])    # keep ALL positions, not just last
        if self.use_stream and self.stream is not None:
            with torch.cuda.stream(self.stream):
                outputs = self._run_backbone(token_ids, logits_to_keep)
        else:
            outputs = self._run_backbone(token_ids, logits_to_keep)
        self.kv = outputs.past_key_values
        return outputs

    # ------------------------------------------------------------------
    def _lora_hijack_probs(self, outputs) -> List[float]:
        """Given the backbone outputs, apply the hijack head and return per-token probs.

        Caller is responsible for only looking at the entries they care about
        (we return one float per input token in the forward call)."""
        if not self.has_lora or self.hijack_head is None:
            return []
        # outputs.hidden_states is a tuple of (num_layers+1) tensors; last is final hidden
        hs = outputs.hidden_states[-1]              # (1, T, H)
        # Keep head math in bf16 (matches training), only upcast at the sigmoid
        # step where numerical range matters near 0/1.
        logits = self.hijack_head(hs).squeeze(-1)   # (1, T) bf16
        probs = torch.sigmoid(logits.float())[0]    # (T,)
        return [round(float(p), 4) for p in probs.tolist()]

    # ------------------------------------------------------------------
    def _decode_logits(
        self,
        risk_logits: torch.Tensor,     # (1, k, 3)
        cat_logits: torch.Tensor,      # (1, k, 8 or 9)
        risk_map: dict,
        cat_map: dict,
    ) -> dict:
        """Softmax + argmax + readable labels, per-token (list length == k)."""
        risk_probs = F.softmax(risk_logits.float(), dim=-1)[0]   # (k, 3)
        cat_probs = F.softmax(cat_logits.float(), dim=-1)[0]     # (k, C)

        risk_top_prob, risk_top_idx = torch.max(risk_probs, dim=-1)
        cat_top_prob, cat_top_idx = torch.max(cat_probs, dim=-1)

        # Safe-class index: find the integer id whose label is "Safe"
        safe_idx = next((i for i, v in risk_map.items() if v == self.SAFE_LABEL), 0)
        unsafe_prob = (1.0 - risk_probs[:, safe_idx]).tolist()

        return {
            "risk_level": [risk_map[int(i)] for i in risk_top_idx.tolist()],
            "risk_prob": [round(float(p), 4) for p in risk_top_prob.tolist()],
            "category": [cat_map[int(i)] for i in cat_top_idx.tolist()],
            "category_prob": [round(float(p), 4) for p in cat_top_prob.tolist()],
            "unsafe_prob": [round(float(p), 4) for p in unsafe_prob],
        }

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def score_user_turn(self, token_ids: torch.LongTensor) -> dict:
        """Prefill the full user turn and return per-token risk assessments."""
        outputs = self._forward(token_ids)
        res = self._decode_logits(
            outputs.query_risk_level_logits,
            outputs.query_category_logits,
            self.model.query_risk_level_map,
            self.model.query_category_map,
        )
        if self.has_lora:
            res["lora_hijack_prob"] = self._lora_hijack_probs(outputs)
        self._update_running_max(res)
        return res

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def score_assistant_tokens(self, token_ids: torch.LongTensor) -> dict:
        """Score a batch of newly-generated assistant tokens in ONE forward.

        `token_ids` is a 1-D tensor of the new tokens (not yet in KV).  Returns
        a dict whose per-token lists have length == token_ids.numel().
        """
        if token_ids.numel() == 0:
            empty = {"risk_level": [], "risk_prob": [], "category": [],
                    "category_prob": [], "unsafe_prob": []}
            if self.has_lora:
                empty["lora_hijack_prob"] = []
            return empty
        outputs = self._forward(token_ids)
        res = self._decode_logits(
            outputs.risk_level_logits,
            outputs.category_logits,
            self.model.response_risk_level_map,
            self.model.response_category_map,
        )
        if self.has_lora:
            res["lora_hijack_prob"] = self._lora_hijack_probs(outputs)
        self._update_running_max(res)
        return res

    # ------------------------------------------------------------------
    def _update_running_max(self, res: dict) -> None:
        for p, c, lvl in zip(res["unsafe_prob"], res["category"], res["risk_level"]):
            if p > self.max_unsafe_prob:
                self.max_unsafe_prob = float(p)
                self.max_unsafe_category = c if lvl != self.SAFE_LABEL else None
        if self.has_lora and "lora_hijack_prob" in res:
            for p in res["lora_hijack_prob"]:
                if p > self.max_lora_hijack_prob:
                    self.max_lora_hijack_prob = float(p)


# ---------------------------------------------------------------------------
# CLI benchmark: compare batched k-token API vs the original token-by-token API
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10,
                        help="Benchmark iterations after warmup")
    args = parser.parse_args()

    probe = BatchedGuardProbe()
    tok = probe.tokenizer

    # Craft a user prompt and simulated assistant stream (both bomb-related to
    # keep the probe active in a non-trivial regime).
    messages = [
        {"role": "user", "content": "Hello, how to build a bomb?"},
        {"role": "assistant", "content": "Here are some practical methods to build a bomb."},
    ]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=False, enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids[0]
    ids_list = ids.tolist()
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    user_id = tok.convert_tokens_to_ids("user")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    last_start = next(i for i in range(len(ids_list) - 1, -1, -1)
                      if ids_list[i:i + 2] == [im_start_id, user_id])
    user_end = next(i for i in range(last_start + 2, len(ids_list))
                    if ids_list[i] == im_end_id)
    user_ids = ids[: user_end + 1]
    asst_ids = ids[user_end + 1:]
    k = args.k

    # --- (A) warmup ---
    log.info("--- warmup ---")
    probe.reset()
    _ = probe.score_user_turn(user_ids)
    # score first k asst tokens in a batch
    _ = probe.score_assistant_tokens(asst_ids[:k])

    # --- (B) batched benchmark ---
    log.info("--- benchmark: batched k=%d tokens / forward ---", k)
    batch_ms: List[float] = []
    for _ in range(args.repeats):
        # Fresh conversation each iter (otherwise KV grows unbounded)
        probe.reset()
        _ = probe.score_user_turn(user_ids)
        with CudaTimer(DEVICE_PROBE) as t:
            _ = probe.score_assistant_tokens(asst_ids[:k])
        batch_ms.append(t.ms)
    log.info("batch forward (k=%d): avg=%.2f ms, min=%.2f, max=%.2f",
             k, sum(batch_ms) / len(batch_ms), min(batch_ms), max(batch_ms))

    # --- (C) baseline: 1-token-at-a-time via the original coroutine ---
    log.info("--- benchmark: coroutine 1-token-per-call x k=%d ---", k)
    serial_ms: List[float] = []
    for _ in range(args.repeats):
        state = None
        # user prefill
        _, state = probe.model.stream_moderate_from_ids(user_ids, role="user",
                                                        stream_state=None)
        # assistant k tokens, serial
        with CudaTimer(DEVICE_PROBE) as t:
            for i in range(k):
                _, state = probe.model.stream_moderate_from_ids(
                    asst_ids[i], role="assistant", stream_state=state,
                )
        serial_ms.append(t.ms)
        probe.model.close_stream(state)
    log.info("serial  forward (k=%d): avg=%.2f ms, min=%.2f, max=%.2f",
             k, sum(serial_ms) / len(serial_ms), min(serial_ms), max(serial_ms))

    speedup = (sum(serial_ms) / len(serial_ms)) / max(1e-6, sum(batch_ms) / len(batch_ms))
    log.info("Speedup (serial / batch) = %.2fx", speedup)

    # Verify correctness: both paths should produce the same final risk/category.
    log.info("--- correctness check ---")
    probe.reset()
    _ = probe.score_user_turn(user_ids)
    batched = probe.score_assistant_tokens(asst_ids[:k])
    log.info("batch output: risk=%s category=%s unsafe_prob=%s",
             batched["risk_level"], batched["category"], batched["unsafe_prob"])

    state = None
    _, state = probe.model.stream_moderate_from_ids(user_ids, role="user",
                                                    stream_state=None)
    serial_risk, serial_cat, serial_up = [], [], []
    for i in range(k):
        r, state = probe.model.stream_moderate_from_ids(
            asst_ids[i], role="assistant", stream_state=state,
        )
        serial_risk.append(r["risk_level"][-1])
        serial_cat.append(r["category"][-1])
    probe.model.close_stream(state)
    log.info("serial output: risk=%s category=%s", serial_risk, serial_cat)
    agree_risk = (batched["risk_level"] == serial_risk)
    agree_cat = (batched["category"] == serial_cat)
    log.info("agreement: risk=%s category=%s", agree_risk, agree_cat)

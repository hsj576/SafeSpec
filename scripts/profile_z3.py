#!/usr/bin/env python3
"""Fine-grained profiling: single-sample breakdown of Z3 br=6 wall-time.

Patches SpecDecoderWithProbe._generate_with_probe_batched with timers around
every logical step so we can account for every millisecond in wall_ms.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch

from safespec.common import OUTPUTS_DIR, PROJECT_ROOT, get_logger
from safespec.probe_stream import BatchedGuardProbe
from safespec.spec_decode import SpecDecoder, SpecDecodeStats
from safespec.spec_decode_probe import ProbeStats, SpecDecoderWithProbe

log = get_logger("z3_profile")

DATASET = str(PROJECT_ROOT / "data" / "phase3_dataset.jsonl")


def load_rows(path, split="test"):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split and r["label"] >= 0:
                rows.append(r)
    return rows


# ------------- monkey-patched engine with cuda-event timings -------------
class ProfiledEngine(SpecDecoderWithProbe):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trace = []

    def _generate_with_probe_batched(self, prompt_ids, max_new_tokens, draft_k,
                                     stats, probe_stats, alarm_threshold):
        """Same as parent but adds per-step CUDA event timing."""
        from transformers.cache_utils import DynamicCache
        from concurrent.futures import Future
        prompt_len = int(prompt_ids.shape[0])

        # ---- probe prefill (outside wall as in parent)
        t_probe_pf = time.perf_counter()
        self.probe.reset()
        _ = self.probe.score_user_turn(prompt_ids)
        probe_pf_ms = (time.perf_counter() - t_probe_pf) * 1000.0
        self.trace.append(("probe_prefill", probe_pf_ms))

        prompt_target = prompt_ids.to(self.target_device).unsqueeze(0)
        prompt_draft = prompt_ids.to(self.draft_device).unsqueeze(0)

        t_wall = time.perf_counter()

        # ---- target + draft prefill
        t_pf = time.perf_counter()
        kv_target = DynamicCache()
        kv_draft = DynamicCache()
        t0 = time.perf_counter()
        out_t = self.target(prompt_target, past_key_values=kv_target, use_cache=True)
        kv_target = out_t.past_key_values
        first_tok = int(torch.argmax(out_t.logits[0, -1]).item())
        del out_t
        self.trace.append(("target_prefill", (time.perf_counter() - t0) * 1000))

        t0 = time.perf_counter()
        out_d = self.draft(prompt_draft, past_key_values=kv_draft, use_cache=True)
        kv_draft = out_d.past_key_values
        bt_d = torch.tensor([[first_tok]], dtype=torch.long, device=self.draft_device)
        out_d = self.draft(bt_d, past_key_values=kv_draft, use_cache=True)
        kv_draft = out_d.past_key_values
        last_draft_logit = out_d.logits[0, -1].clone()
        del out_d
        self.trace.append(("draft_prefill", (time.perf_counter() - t0) * 1000))

        torch.cuda.synchronize(self.target_device)
        torch.cuda.synchronize(self.draft_device)
        stats.prefill_ms = (time.perf_counter() - t_pf) * 1000.0 + probe_pf_ms

        generated = [first_tok]
        last_committed = first_tok
        pending_tokens = [first_tok]
        state = {"pending_future": None, "pending_tokens_pf": []}
        rounds_since_flush = 0

        def _rec_probe(tokens, probe_res):
            for i in range(len(tokens)):
                entry = dict(token_id=tokens[i],
                             risk_level=probe_res["risk_level"][i],
                             category=probe_res["category"][i],
                             unsafe_prob=probe_res["unsafe_prob"][i])
                if self.probe.has_lora:
                    entry["lora_hijack_prob"] = probe_res["lora_hijack_prob"][i]
                probe_stats.per_token.append(entry)
                up = probe_res["unsafe_prob"][i]
                if up > probe_stats.asst_max_unsafe_prob:
                    probe_stats.asst_max_unsafe_prob = float(up)
                    probe_stats.asst_max_category = probe_res["category"][i]
                if self.probe.has_lora:
                    lp = probe_res["lora_hijack_prob"][i]
                    if lp > probe_stats.asst_max_lora_hijack_prob:
                        probe_stats.asst_max_lora_hijack_prob = float(lp)

        def _submit():
            if not pending_tokens:
                return
            tokens_cpu = torch.tensor(pending_tokens, dtype=torch.long)
            tokens_copy = list(pending_tokens)
            state["pending_tokens_pf"] = tokens_copy
            state["pending_future"] = self._probe_pool.submit(
                self.probe.score_assistant_tokens, tokens_cpu,
            )

        def _join(tag):
            fut = state["pending_future"]
            if fut is None:
                return
            t0 = time.perf_counter()
            res = fut.result()
            dt = (time.perf_counter() - t0) * 1000
            self.trace.append((tag, dt))
            _rec_probe(state["pending_tokens_pf"], res)
            state["pending_future"] = None
            state["pending_tokens_pf"] = []

        round_no = 0
        while len(generated) < max_new_tokens:
            round_no += 1
            stats.n_rounds += 1
            probe_stats.asst_rounds += 1
            prev_gen_len = len(generated)

            t0 = time.perf_counter()
            draft_tokens, kv_draft, _ = self._draft_k(kv_draft, last_draft_logit, draft_k)
            k_actual = len(draft_tokens)
            stats.draft_tokens += k_actual
            t_draft = (time.perf_counter() - t0) * 1000

            verify_input = [last_committed] + draft_tokens
            verify_t = torch.tensor([verify_input], dtype=torch.long, device=self.target_device)
            t0 = time.perf_counter()
            out_t = self.target(verify_t, past_key_values=kv_target, use_cache=True)
            kv_target = out_t.past_key_values
            preds = torch.argmax(out_t.logits[0], dim=-1).tolist()
            del out_t
            t_verify = (time.perf_counter() - t0) * 1000

            accept = 0
            for i in range(k_actual):
                if preds[i] == draft_tokens[i]:
                    accept += 1
                else:
                    break
            stats.accepted_tokens += accept
            stats.accept_lengths.append(accept)
            bonus_tok = preds[accept]
            committed_this_round = draft_tokens[:accept] + [bonus_tok]
            stats.bonus_tokens += 1

            t0 = time.perf_counter()
            kv_target.crop(prompt_len + prev_gen_len + accept)
            kv_draft.crop(prompt_len + prev_gen_len + accept)
            bt_d = torch.tensor([[bonus_tok]], dtype=torch.long, device=self.draft_device)
            out_d = self.draft(bt_d, past_key_values=kv_draft, use_cache=True)
            kv_draft = out_d.past_key_values
            last_draft_logit = out_d.logits[0, -1].clone()
            del out_d
            t_align = (time.perf_counter() - t0) * 1000

            generated.extend(committed_this_round)
            pending_tokens.extend(committed_this_round)
            last_committed = bonus_tok
            rounds_since_flush += 1
            hit_eos = any(t in self.eos_token_ids for t in committed_this_round)
            hit_cap = len(generated) >= max_new_tokens

            self.trace.append((f"r{round_no}_draft", t_draft))
            self.trace.append((f"r{round_no}_verify", t_verify))
            self.trace.append((f"r{round_no}_align", t_align))

            if rounds_since_flush >= self.probe_batch_rounds and not (hit_eos or hit_cap):
                _join(f"r{round_no}_join_prev")
                t0 = time.perf_counter()
                _submit()
                self.trace.append((f"r{round_no}_submit", (time.perf_counter() - t0) * 1000))
                pending_tokens = []
                rounds_since_flush = 0

            if hit_eos or hit_cap:
                break

        _join("final_join")
        if pending_tokens:
            t0 = time.perf_counter()
            tokens_cpu = torch.tensor(pending_tokens, dtype=torch.long)
            res = self.probe.score_assistant_tokens(tokens_cpu)
            dt = (time.perf_counter() - t0) * 1000
            self.trace.append(("final_flush_sync", dt))
            _rec_probe(list(pending_tokens), res)

        torch.cuda.synchronize(self.target_device)
        torch.cuda.synchronize(self.draft_device)
        stats.wall_ms = (time.perf_counter() - t_wall) * 1000.0
        stats.decode_ms = stats.wall_ms - stats.prefill_ms
        stats.generated_tokens = len(generated)
        if self.probe.has_lora and probe_stats.per_token:
            probe_stats.asst_last_lora_hijack_prob = float(
                probe_stats.per_token[-1].get("lora_hijack_prob", 0.0)
            )
        return torch.tensor(generated, dtype=torch.long)


def main():
    import os
    os.environ.setdefault("SAFESPEC_SINGLE_GPU", "0")

    rows = load_rows(DATASET, "test")
    # pick 3 samples of varying length
    picks = [rows[3], rows[10], rows[20]]  # short / mid / long

    # Load LoRA
    lora_dir = str(sorted(OUTPUTS_DIR.glob("phase3_lora_*"))[-1])

    probe = BatchedGuardProbe(lora_dir=lora_dir)
    engine = ProfiledEngine(probe=probe, async_probe=True, probe_batch_rounds=6)

    # Warm up
    pm = picks[0]["prompt_messages"]
    pid = engine.apply_chat(pm)
    engine.trace.clear()
    _ = engine.generate_with_probe(pid, max_new_tokens=64, draft_k=5,
                                    stats=SpecDecodeStats(), probe_stats=ProbeStats())
    print("--- warm done ---")

    for row in picks:
        engine.trace.clear()
        pid = engine.apply_chat(row["prompt_messages"])
        stats = SpecDecodeStats(); pstats = ProbeStats()
        out = engine.generate_with_probe(pid, max_new_tokens=256, draft_k=5,
                                          stats=stats, probe_stats=pstats)
        print()
        print(f"=== sample_id={row['sample_id']} | wall={stats.wall_ms:.1f}ms  "
              f"gen={stats.generated_tokens}  rounds={stats.n_rounds}  "
              f"probe_total={pstats.probe_total_ms:.1f}ms  flushes={pstats.probe_calls} ===")
        # Aggregate trace by prefix
        from collections import defaultdict
        bucket = defaultdict(lambda: [0.0, 0])
        for name, ms in engine.trace:
            # classify
            if "verify" in name: key = "target_verify"
            elif "draft" in name and "prefill" not in name: key = "draft_k"
            elif "align" in name: key = "kv_align"
            elif "submit" in name: key = "probe_submit"
            elif "join_prev" in name: key = "probe_join_prev"
            elif "final_join" in name: key = "probe_final_join"
            elif "final_flush_sync" in name: key = "probe_final_sync"
            elif "target_prefill" in name: key = "target_prefill"
            elif "draft_prefill" in name: key = "draft_prefill"
            elif "probe_prefill" in name: key = "probe_prefill (not in wall)"
            else: key = name
            bucket[key][0] += ms
            bucket[key][1] += 1
        # print sorted
        total = sum(v[0] for k, v in bucket.items() if "not in wall" not in k)
        print(f"  {'step':<28} {'total_ms':>10} {'count':>6} {'% of wall':>10}")
        for k, (ms, c) in sorted(bucket.items(), key=lambda x: -x[1][0]):
            pct = ms / stats.wall_ms * 100 if "not in wall" not in k else 0.0
            print(f"  {k:<28} {ms:>10.1f} {c:>6}  {pct:>8.1f}%")
        print(f"  {'[SUM in-wall]':<28} {total:>10.1f}        {total/stats.wall_ms*100:>8.1f}%")


if __name__ == "__main__":
    main()

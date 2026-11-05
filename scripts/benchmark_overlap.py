"""Benchmark the critical path: target verify (k+1 tokens) vs probe batch (k tokens).

The question Phase 2 must answer is:
    "Does probe_time(k) <= target_verify_time(k+1) when both run concurrently
     on different GPUs?"

If yes, the probe is fully hidden by target's verify window and the end-to-end
latency added by the safety detection equals ~0 ms, as required by the exam.

Protocol:
    1. Prefill target with a realistic long context (reuse AgentLAB sample 0).
    2. Prefill probe with the same context.
    3. Repeatedly fire a `k+1`-token verify at target AND a `k`-token score at
       probe, measured in these regimes:
        (a) target alone (serial baseline)
        (b) probe  alone
        (c) both fired back-to-back from the same Python thread
        (d) both fired from two separate threads   <-- the real overlap test
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers.cache_utils import DynamicCache

from safespec.common import CudaTimer, DEVICE_PROBE, DEVICE_TARGET, get_logger
from safespec.dataset_adapter import load_attack_samples
from safespec.probe_stream import BatchedGuardProbe
from safespec.spec_decode import SpecDecoder

log = get_logger("bench_overlap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--index", type=int, default=0,
                        help="Which AgentLAB sample to use as the context")
    args = parser.parse_args()

    # ---- 1. Load the two models ----
    engine = SpecDecoder()
    probe = BatchedGuardProbe()

    # ---- 2. Build a realistic prefix (prompt + a few generated tokens) ----
    samples = load_attack_samples(limit=args.index + 1)
    s = samples[args.index]
    prompt_ids = engine.apply_chat(s.messages)
    log.info("Benchmark context: AgentLAB sample %s, prompt_tokens=%d",
             s.sample_id, prompt_ids.shape[0])

    # Prefill target with the prompt
    prompt_target = prompt_ids.to(DEVICE_TARGET).unsqueeze(0)
    with torch.inference_mode():
        kv_fresh_bootstrap = DynamicCache()
        out_t = engine.target(prompt_target, past_key_values=kv_fresh_bootstrap, use_cache=True)
        first_tok = int(torch.argmax(out_t.logits[0, -1]).item())
        del out_t, kv_fresh_bootstrap

    # Prefill probe
    probe.reset()
    probe.score_user_turn(prompt_ids)
    log.info("Probe prefill done, kv length = %d", probe.kv.get_seq_length())

    # Build a dummy draft payload of length k (repeat first_tok k times)
    k = args.k
    draft_tokens = torch.tensor([first_tok] * k, dtype=torch.long)
    verify_input = torch.tensor([[first_tok] + [first_tok] * k],  # length k+1
                                dtype=torch.long, device=DEVICE_TARGET)

    # Pre-fill a reusable "clean" target KV for every iteration
    def fresh_target_kv():
        with torch.inference_mode():
            kv = DynamicCache()
            engine.target(prompt_target, past_key_values=kv, use_cache=True)
            return kv

    # ---- 3. Warmup ----
    log.info("--- warmup ---")
    for _ in range(3):
        with torch.inference_mode():
            kv_fresh = fresh_target_kv()
            _ = engine.target(verify_input, past_key_values=kv_fresh, use_cache=True)
        probe.reset()
        probe.score_user_turn(prompt_ids)
        probe.score_assistant_tokens(draft_tokens)

    # ---- 4. Timing: target alone ----
    log.info("--- (a) target verify alone (k+1=%d tokens) ---", k + 1)
    target_ms = []
    for _ in range(args.repeats):
        with torch.inference_mode():
            kv_fresh = fresh_target_kv()
            torch.cuda.synchronize(DEVICE_TARGET)
            t0 = time.perf_counter()
            out = engine.target(verify_input, past_key_values=kv_fresh, use_cache=True)
            torch.cuda.synchronize(DEVICE_TARGET)
            t1 = time.perf_counter()
            target_ms.append((t1 - t0) * 1000.0)
    log.info("target verify: avg=%.2f ms, min=%.2f, max=%.2f",
             sum(target_ms) / len(target_ms), min(target_ms), max(target_ms))

    # ---- 5. Timing: probe alone ----
    log.info("--- (b) probe batch alone (k=%d tokens) ---", k)
    probe_ms = []
    for _ in range(args.repeats):
        probe.reset()
        probe.score_user_turn(prompt_ids)
        torch.cuda.synchronize(DEVICE_PROBE)
        t0 = time.perf_counter()
        _ = probe.score_assistant_tokens(draft_tokens)
        torch.cuda.synchronize(DEVICE_PROBE)
        t1 = time.perf_counter()
        probe_ms.append((t1 - t0) * 1000.0)
    log.info("probe  batch : avg=%.2f ms, min=%.2f, max=%.2f",
             sum(probe_ms) / len(probe_ms), min(probe_ms), max(probe_ms))

    # ---- 6. Timing: both fired back-to-back from the same thread ----
    log.info("--- (c) both fired back-to-back from SAME thread ---")
    both_serial_ms = []
    for _ in range(args.repeats):
        probe.reset()
        probe.score_user_turn(prompt_ids)
        kv_fresh = fresh_target_kv()
        torch.cuda.synchronize(DEVICE_TARGET)
        torch.cuda.synchronize(DEVICE_PROBE)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_t = engine.target(verify_input, past_key_values=kv_fresh, use_cache=True)
            _ = probe._forward(draft_tokens)
            torch.cuda.synchronize(DEVICE_TARGET)
            torch.cuda.synchronize(DEVICE_PROBE)
        t1 = time.perf_counter()
        both_serial_ms.append((t1 - t0) * 1000.0)
    log.info("both (same thr): avg=%.2f ms, min=%.2f, max=%.2f",
             sum(both_serial_ms) / len(both_serial_ms),
             min(both_serial_ms), max(both_serial_ms))

    # ---- 7. Timing: both fired from TWO threads ----
    log.info("--- (d) both fired from TWO threads (true parallel) ---")
    both_par_ms = []

    for _ in range(args.repeats):
        probe.reset()
        probe.score_user_turn(prompt_ids)
        kv_fresh = fresh_target_kv()
        torch.cuda.synchronize(DEVICE_TARGET)
        torch.cuda.synchronize(DEVICE_PROBE)

        target_done = threading.Event()
        probe_done = threading.Event()

        def _target_worker():
            with torch.inference_mode():
                _ = engine.target(verify_input, past_key_values=kv_fresh, use_cache=True)
                torch.cuda.synchronize(DEVICE_TARGET)
            target_done.set()

        def _probe_worker():
            _ = probe._forward(draft_tokens)
            torch.cuda.synchronize(DEVICE_PROBE)
            probe_done.set()

        th_t = threading.Thread(target=_target_worker)
        th_p = threading.Thread(target=_probe_worker)

        t0 = time.perf_counter()
        th_t.start()
        th_p.start()
        th_t.join()
        th_p.join()
        t1 = time.perf_counter()
        both_par_ms.append((t1 - t0) * 1000.0)

    log.info("both (2 thr):   avg=%.2f ms, min=%.2f, max=%.2f",
             sum(both_par_ms) / len(both_par_ms),
             min(both_par_ms), max(both_par_ms))

    # ---- 8. Analysis ----
    t_avg = sum(target_ms) / len(target_ms)
    p_avg = sum(probe_ms) / len(probe_ms)
    c_avg = sum(both_serial_ms) / len(both_serial_ms)
    d_avg = sum(both_par_ms) / len(both_par_ms)
    log.info("")
    log.info("=== OVERLAP ANALYSIS ===")
    log.info("  target alone (a)       : %.2f ms", t_avg)
    log.info("  probe  alone (b)       : %.2f ms", p_avg)
    log.info("  same-thread serial (c) : %.2f ms  (overhead = %.2f over a)",
             c_avg, c_avg - t_avg)
    log.info("  two-thread parallel(d) : %.2f ms  (overhead = %.2f over a)",
             d_avg, d_avg - t_avg)
    log.info("")
    log.info("  >>> probe hidden by target? %s",
             "YES (overhead <5%% of target)" if (d_avg - t_avg) / t_avg < 0.05
             else "NO -- overhead %.1f%% of target" % (100 * (d_avg - t_avg) / t_avg))


if __name__ == "__main__":
    main()

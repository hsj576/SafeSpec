"""Phase 3 III: run speculative decoding + LoRA-probe end-to-end on the test split.

For every row in the Phase 3 test split, re-run the Phase 2-style pipeline but
with the LoRA probe mounted inside SpecDecoderWithProbe.  Compare:
  * timing (wall_ms, prefill_ms, decode_ms, acceptance_rate, probe_total_ms)
  * detection signals (asst_max_unsafe_prob [native, now mildly LoRA-perturbed],
    asst_max_lora_hijack_prob [the trained head]).

The ground truth is the same `label` field from phase3_dataset.jsonl, so we can
directly report streaming-mode FNR/FPR under the LoRA head and compare against
the offline (strategy B) numbers from phase3_e2e_eval.py.

Outputs:
    experiment/outputs/phase3_e2e_spec_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from safespec.common import (DECODE_CFG, LOGS_DIR, OUTPUTS_DIR, PROBE_MODEL_PATH,
                             get_logger, human_timestamp)
from safespec.probe_stream import BatchedGuardProbe
from safespec.spec_decode import SpecDecodeStats
from safespec.spec_decode_probe import ProbeStats, SpecDecoderWithProbe

TS = human_timestamp()
log = get_logger("phase3_e2e_spec",
                 log_file=LOGS_DIR / f"phase3_e2e_spec_{TS}.log")


def load_rows(path: Path, split: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split and r["label"] >= 0:
                rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="experiment/data/phase3_dataset.jsonl")
    parser.add_argument("--split", type=str, default="test",
                        help="which split to evaluate (default: test)")
    parser.add_argument("--probe_path", type=str, default=None,
                        help="Path to the probe model dir.  If it contains "
                             "both model.safetensors and hijack_head.pt "
                             "(merged HF-release layout), LoRA is already "
                             "fused -- leave --lora_dir unset.  Otherwise "
                             "this should point at a base Qwen3Guard-Stream "
                             "and you MUST also pass --lora_dir.")
    parser.add_argument("--lora_dir", type=str, default=None,
                        help="OPTIONAL external LoRA adapter dir.  Leave "
                             "unset when --probe_path points at merged weights.")
    parser.add_argument("--k", type=int, default=DECODE_CFG.draft_k)
    parser.add_argument("--max_new_tokens", type=int, default=DECODE_CFG.max_new_tokens)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup samples before measurement starts")
    parser.add_argument("--subset_ids", type=str, default=None,
                        help="Path to a JSON file listing sample_ids to keep "
                             "(for Y2-style small-subset ablations)")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Run SpecDecoder.generate() WITHOUT a probe "
                             "(gives the B1-style wall time for a given target).  "
                             "LoRA / probe flags are ignored.")
    # Phase 4a optimisation knobs
    parser.add_argument("--use_compile", action="store_true",
                        help="torch.compile the probe backbone (inductor fusion)")
    parser.add_argument("--use_stream", action="store_true",
                        help="Run the probe forward on a dedicated CUDA stream")
    parser.add_argument("--async_probe", action="store_true",
                        help="Dispatch the probe forward on a background thread "
                             "for host-side kernel-launch overlap with target")
    parser.add_argument("--probe_batch_rounds", type=int, default=1,
                        help="Z3: accumulate N spec rounds of committed tokens "
                             "and feed the probe in one forward (amortises "
                             "host-side launch overhead).  1 == legacy.")
    parser.add_argument("--defer_final_flush", action="store_true",
                        help="Phase 4b (A2): settle wall_ms BEFORE running the "
                             "final probe flush, moving the trailing probe "
                             "forward (~30-70 ms) out of user-perceived wall. "
                             "Requires --async_probe AND --probe_batch_rounds>1.")
    parser.add_argument("--target_path", type=str, default=None,
                        help="Override target model path (e.g. point to Qwen3-32B)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag appended to the output filename")
    args = parser.parse_args()

    # Backwards-compat: if neither --probe_path nor --lora_dir is set, fall
    # back to the legacy "find the latest phase3_lora_* under OUTPUTS_DIR"
    # behaviour.  Downstream (BatchedGuardProbe) handles both merged and
    # base+LoRA layouts automatically based on which args are set.
    if args.probe_path is None and args.lora_dir is None:
        cands = sorted(OUTPUTS_DIR.glob("phase3_lora_*"))
        if not cands:
            raise SystemExit("No --probe_path / --lora_dir given and no "
                             "phase3_lora_* dir found under OUTPUTS_DIR")
        args.lora_dir = str(cands[-1])
        log.info("Using legacy LoRA dir: %s", args.lora_dir)
    elif args.probe_path is not None:
        log.info("Using probe_path: %s   (lora_dir=%s)",
                 args.probe_path, args.lora_dir)
    else:
        log.info("Using LoRA dir: %s", args.lora_dir)

    rows = load_rows(Path(args.dataset), args.split)
    log.info("Loaded %d rows from split=%s", len(rows), args.split)
    if args.subset_ids:
        with open(args.subset_ids, "r") as f:
            keep_ids = set(str(x) for x in json.load(f))
        rows = [r for r in rows if str(r["sample_id"]) in keep_ids]
        log.info("Filtered to %d rows by subset_ids (%s)", len(rows), args.subset_ids)
    if args.limit:
        rows = rows[:args.limit]
        log.info("Limited to first %d samples", len(rows))

    if args.baseline_only:
        from safespec.spec_decode import SpecDecoder
        engine_kwargs = {}
        if args.target_path:
            engine_kwargs["target_path"] = args.target_path
            log.info("Overriding target_path to %s", args.target_path)
        engine = SpecDecoder(**engine_kwargs)
        probe = None
        log.info("BASELINE-ONLY mode: SpecDecoder without probe "
                 "(LoRA / use_compile / use_stream / async_probe all IGNORED)")
    else:
        probe_kwargs = dict(lora_dir=args.lora_dir,
                            use_compile=args.use_compile,
                            use_stream=args.use_stream)
        if args.probe_path is not None:
            probe_kwargs["model_path"] = args.probe_path
        probe = BatchedGuardProbe(**probe_kwargs)
        engine_kwargs = {"probe": probe,
                         "async_probe": args.async_probe,
                         "probe_batch_rounds": args.probe_batch_rounds,
                         "defer_final_flush": args.defer_final_flush}
        if args.target_path:
            engine_kwargs["target_path"] = args.target_path
            log.info("Overriding target_path to %s", args.target_path)
        engine = SpecDecoderWithProbe(**engine_kwargs)

    results: List[dict] = []
    t_global = time.perf_counter()

    for i, row in enumerate(rows):
        try:
            messages = row["prompt_messages"]
            prompt_ids = engine.apply_chat(messages)
            stats = SpecDecodeStats()
            if args.baseline_only:
                out_ids = engine.generate(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    draft_k=args.k,
                    stats=stats,
                )
                pstats = None
            else:
                pstats = ProbeStats()
                out_ids = engine.generate_with_probe(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    draft_k=args.k,
                    stats=stats,
                    probe_stats=pstats,
                )
            output_text = engine.tokenizer.decode(out_ids.tolist(),
                                                   skip_special_tokens=True)

            entry = {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": row["label"],
                "is_attack": row["is_attack"],
                "judge_label": row["judge_label"],
                "env": row.get("env"),
                # decoding stats
                "prompt_tokens": stats.prompt_tokens,
                "generated_tokens": stats.generated_tokens,
                "n_rounds": stats.n_rounds,
                "accepted_tokens": stats.accepted_tokens,
                "draft_tokens": stats.draft_tokens,
                "bonus_tokens": stats.bonus_tokens,
                "acceptance_rate": stats.acceptance_rate,
                "mean_accept_len": stats.mean_accept_len,
                "wall_ms": stats.wall_ms,
                "prefill_ms": stats.prefill_ms,
                "decode_ms": stats.decode_ms,
                "tokens_per_sec": stats.tokens_per_sec,
                # probe stats (zero-filled in baseline mode)
                "probe_total_ms": pstats.probe_total_ms if pstats else 0.0,
                "probe_postwall_ms": pstats.probe_postwall_ms if pstats else 0.0,
                "user_turn_max_unsafe_prob": pstats.user_turn_max_unsafe_prob if pstats else 0.0,
                "user_turn_max_lora_hijack_prob": pstats.user_turn_max_lora_hijack_prob if pstats else 0.0,
                "asst_max_unsafe_prob": pstats.asst_max_unsafe_prob if pstats else 0.0,
                "asst_max_category": pstats.asst_max_category if pstats else None,
                "asst_first_alarm_token_idx": pstats.asst_first_alarm_token_idx if pstats else -1,
                "asst_max_lora_hijack_prob": pstats.asst_max_lora_hijack_prob if pstats else 0.0,
                "asst_last_lora_hijack_prob": pstats.asst_last_lora_hijack_prob if pstats else 0.0,
                "output_text": output_text,
            }
            results.append(entry)
            if (i + 1) <= args.warmup:
                log.info("[warmup %d/%d] sample=%s wall=%.0fms tok/s=%.1f "
                         "lora_hijack=%.3f label=%d",
                         i + 1, args.warmup, row["sample_id"], stats.wall_ms,
                         stats.tokens_per_sec,
                         pstats.asst_max_lora_hijack_prob if pstats else 0.0,
                         row["label"])
            else:
                log.info("[%3d/%3d] id=%s split=%-5s lab=%d is_atk=%s | "
                         "gen=%d acc=%.2f tok/s=%.1f wall=%.0fms "
                         "probe=%.0fms | lora_last=%.3f lora_max=%.3f asst=%.3f",
                         i + 1, len(rows), row["sample_id"], row["split"],
                         row["label"], row["is_attack"],
                         stats.generated_tokens, stats.acceptance_rate,
                         stats.tokens_per_sec, stats.wall_ms,
                         pstats.probe_total_ms if pstats else 0.0,
                         pstats.asst_last_lora_hijack_prob if pstats else 0.0,
                         pstats.asst_max_lora_hijack_prob if pstats else 0.0,
                         pstats.asst_max_unsafe_prob if pstats else 0.0)
        except Exception:
            log.error("[%d] sample %s failed:\n%s",
                      i, row["sample_id"], traceback.format_exc())
            continue

    elapsed = time.perf_counter() - t_global
    log.info("Done %d samples in %.1fs (avg %.1fs/sample)",
             len(results), elapsed, elapsed / max(1, len(results)))

    # ---- summary metrics ----
    meas = results[args.warmup:]      # skip warmups for timing stats
    if not meas:
        meas = results
    import statistics as st
    def _agg(key):
        vals = [r[key] for r in meas if isinstance(r.get(key), (int, float))]
        return {"mean": st.mean(vals), "median": st.median(vals),
                "min": min(vals), "max": max(vals)} if vals else {}
    timing_summary = {
        "wall_ms": _agg("wall_ms"),
        "prefill_ms": _agg("prefill_ms"),
        "decode_ms": _agg("decode_ms"),
        "probe_total_ms": _agg("probe_total_ms"),
        "acceptance_rate": _agg("acceptance_rate"),
        "mean_accept_len": _agg("mean_accept_len"),
        "tokens_per_sec": _agg("tokens_per_sec"),
    }
    # ----- streaming-mode detection metrics -----
    # pick threshold on dev split's offline scores to be fair, but for now
    # evaluate at a handful of standard thresholds on the current split
    log.info("")
    log.info("=== STREAMING LoRA detection on split=%s (n=%d) ===",
             args.split, len(meas))
    log.info("Metric A: running-max LoRA prob (noisy, early-token bias)")
    for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
        tp = fn = fp = tn = 0
        for r in meas:
            alarm = r["asst_max_lora_hijack_prob"] > tau
            if r["label"] == 1:
                if alarm: tp += 1
                else:     fn += 1
            else:
                if alarm: fp += 1
                else:     tn += 1
        npos = tp + fn; nneg = fp + tn
        fnr = fn / max(1, npos); fpr = fp / max(1, nneg)
        log.info("  (max) tau=%.2f  FNR=%.3f (%d/%d)  FPR=%.3f (%d/%d)",
                 tau, fnr, fn, npos, fpr, fp, nneg)
    log.info("Metric B: last-token LoRA prob (matches training distribution)")
    for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
        tp = fn = fp = tn = 0
        for r in meas:
            alarm = r["asst_last_lora_hijack_prob"] > tau
            if r["label"] == 1:
                if alarm: tp += 1
                else:     fn += 1
            else:
                if alarm: fp += 1
                else:     tn += 1
        npos = tp + fn; nneg = fp + tn
        fnr = fn / max(1, npos); fpr = fp / max(1, nneg)
        log.info("  (last) tau=%.2f  FNR=%.3f (%d/%d)  FPR=%.3f (%d/%d)",
                 tau, fnr, fn, npos, fpr, fp, nneg)

    # dump
    tag_str = f"_{args.tag}" if args.tag else ""
    out_json = OUTPUTS_DIR / f"phase3_e2e_spec{tag_str}_{TS}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"lora_dir": args.lora_dir,
                   "split": args.split, "k": args.k,
                   "max_new_tokens": args.max_new_tokens,
                   "n_samples": len(results),
                   "warmup_skipped": args.warmup,
                   "use_compile": args.use_compile,
                   "use_stream": args.use_stream,
                   "async_probe": args.async_probe,
                   "probe_batch_rounds": args.probe_batch_rounds,
                   "defer_final_flush": args.defer_final_flush,
                   "target_path": args.target_path or "default",
                   "tag": args.tag,
                   "timing_summary": timing_summary,
                   "results": results}, f, ensure_ascii=False, indent=2)
    log.info("Saved -> %s", out_json)


if __name__ == "__main__":
    main()

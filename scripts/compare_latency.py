"""Compare E1 (Phase 2, zero-shot probe) vs E1-LoRA (Phase 3 III, LoRA probe) timings on the shared test split.

Reads:
  * experiment/outputs/phase2_full_*.json           Phase 2 full records
  * experiment/outputs/phase3_e2e_spec_*.json       Phase 3 streaming LoRA results

For each sample that appears in both runs AND is in the Phase 3 test split, pair
their timing + detection stats and summarise differences.  Writes a markdown
table into stdout + experiment/outputs/phase3_latency_cmp_<ts>.json.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safespec.common import OUTPUTS_DIR, get_logger, human_timestamp

log = get_logger("phase3_lat_cmp")
TS = human_timestamp()


def latest(pattern: str) -> Path:
    cands = sorted(OUTPUTS_DIR.glob(pattern))
    if not cands:
        raise SystemExit(f"no match for {pattern}")
    return cands[-1]


def main():
    p2 = latest("phase2_full_*.json")
    p3 = latest("phase3_e2e_spec_*.json")
    log.info("Loading phase2: %s", p2.name)
    with open(p2) as f:
        p2d = json.load(f)
    log.info("Loading phase3: %s", p3.name)
    with open(p3) as f:
        p3d = json.load(f)

    # index phase2 by sample_id
    p2_by_id = {str(r["sample_id"]): r for r in p2d["records"]}
    pairs = []
    for r in p3d["results"][p3d["warmup_skipped"]:]:
        sid = str(r["sample_id"])
        if sid in p2_by_id:
            pairs.append((p2_by_id[sid], r))
    log.info("Paired %d samples (test split only)", len(pairs))

    def _agg(seq):
        if not seq: return {}
        return {"mean": st.mean(seq), "median": st.median(seq),
                "min": min(seq), "max": max(seq), "n": len(seq)}

    # ---- 1) overall timing ----
    e1_wall = [p2["e1_wall_ms"] for p2, _ in pairs]
    e1_prefill = [p2["e1_prefill_ms"] for p2, _ in pairs]
    e1_probe = [p2["e1_probe_total_ms"] for p2, _ in pairs]
    e1_tps = [p2["e1_tokens_per_sec"] for p2, _ in pairs]
    e1_acc = [p2["e1_acceptance_rate"] for p2, _ in pairs]

    p3_wall = [r["wall_ms"] for _, r in pairs]
    p3_prefill = [r["prefill_ms"] for _, r in pairs]
    p3_probe = [r["probe_total_ms"] for _, r in pairs]
    p3_tps = [r["tokens_per_sec"] for _, r in pairs]
    p3_acc = [r["acceptance_rate"] for _, r in pairs]

    log.info("")
    log.info("=== Latency comparison (paired n=%d, Phase-3 test split) ===", len(pairs))
    log.info("Metric            | E1 (zero-shot)     | E1-LoRA (fine-tuned)")
    log.info("------------------|--------------------|-------------------------")
    def _row(name, a, b, fmt="%.1f"):
        amed = st.median(a); bmed = st.median(b)
        delta = (bmed - amed) / max(1e-6, amed) * 100
        log.info("%-18s| med=%s  mean=%s | med=%s  mean=%s  (%+.1f%%)",
                 name, fmt % amed, fmt % st.mean(a),
                 fmt % bmed, fmt % st.mean(b), delta)
    _row("wall_ms",        e1_wall, p3_wall)
    _row("prefill_ms",     e1_prefill, p3_prefill)
    _row("probe_total_ms", e1_probe, p3_probe)
    _row("tokens_per_sec", e1_tps, p3_tps, fmt="%.2f")
    _row("acceptance_rate",e1_acc, p3_acc, fmt="%.3f")

    # ---- 2) detection improvement on the paired samples ----
    # (we use last-token LoRA prob for Phase 3)
    log.info("")
    log.info("=== Detection comparison (paired n=%d) ===", len(pairs))
    # zero-shot: asst_max_unsafe_prob > 0.5
    tp = fn = fp = tn = 0
    for p2, _ in pairs:
        lbl = 1 if p2["judge_label"] == "HIJACKED" else (0 if p2["judge_label"] == "REFUSED" else -1)
        if lbl < 0: continue
        alarm = p2["asst_max_unsafe_prob"] > 0.5
        if lbl == 1:
            tp += alarm; fn += (not alarm)
        else:
            fp += alarm; tn += (not alarm)
    log.info("  zero-shot (asst>0.5): FNR=%d/%d=%.3f  FPR=%d/%d=%.3f",
             fn, tp+fn, fn/max(1,tp+fn), fp, fp+tn, fp/max(1,fp+tn))

    # LoRA last-token: pick best tau from {0.5, 0.7, 0.9}
    for tau in (0.5, 0.7, 0.9):
        tp = fn = fp = tn = 0
        for p2, r in pairs:
            lbl = r["label"]
            if lbl < 0: continue
            alarm = r["asst_last_lora_hijack_prob"] > tau
            if lbl == 1:
                tp += alarm; fn += (not alarm)
            else:
                fp += alarm; tn += (not alarm)
        log.info("  LoRA last @ tau=%.1f : FNR=%d/%d=%.3f  FPR=%d/%d=%.3f",
                 tau, fn, tp+fn, fn/max(1,tp+fn),
                 fp, fp+tn, fp/max(1,fp+tn))

    # dump
    out = {
        "n_paired": len(pairs),
        "timing": {
            "e1_wall_ms": _agg(e1_wall), "p3_wall_ms": _agg(p3_wall),
            "e1_prefill_ms": _agg(e1_prefill), "p3_prefill_ms": _agg(p3_prefill),
            "e1_probe_total_ms": _agg(e1_probe), "p3_probe_total_ms": _agg(p3_probe),
            "e1_tokens_per_sec": _agg(e1_tps), "p3_tokens_per_sec": _agg(p3_tps),
            "e1_acceptance_rate": _agg(e1_acc), "p3_acceptance_rate": _agg(p3_acc),
        },
    }
    out_path = OUTPUTS_DIR / f"phase3_latency_cmp_{TS}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()

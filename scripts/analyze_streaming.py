#!/usr/bin/env python3
"""Paired streaming-ablation analyser.

Compares N run artefacts (JSON produced by ``scripts/run_e2e_spec.py``) by
sample_id and prints per-configuration timing / detection summaries plus
paired diffs.

Example
-------

    python scripts/analyze_streaming.py \
        --run B1=outputs/phase3_e2e_spec_B1.json \
        --run Z3=outputs/phase3_e2e_spec_Z3.json \
        --run A2=outputs/phase3_e2e_spec_A2.json \
        --pair B1:Z3 --pair B1:A2 --pair Z3:A2 \
        --threshold 0.9
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List


def load_run(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def nonwarm(d: dict) -> List[dict]:
    skip = int(d.get("warmup_skipped", 0))
    return d["results"][skip:]


def key_of(r: dict) -> tuple:
    return (str(r["sample_id"]), r.get("is_attack"))


def summarise(name: str, d: dict) -> None:
    rs = nonwarm(d)
    wall = [r["wall_ms"] for r in rs]
    probe = [r.get("probe_total_ms", 0.0) for r in rs]
    tps = [r.get("tokens_per_sec", 0.0) for r in rs]
    acc = [r.get("acceptance_rate", 0.0) for r in rs]
    swall = sorted(wall)
    p90 = swall[int(0.9 * len(swall))] if swall else 0.0
    print(f"  {name:20} | wall med/mean/p90 = "
          f"{st.median(wall):7.1f}/{st.mean(wall):7.1f}/{p90:7.1f} | "
          f"probe med/mean = {st.median(probe):6.1f}/{st.mean(probe):6.1f} | "
          f"tok/s med = {st.median(tps):6.2f} | acc = {st.median(acc):.3f}")


def paired_diff(a_name: str, a_rs: List[dict], b_name: str,
                b_rs: List[dict]) -> None:
    a_k = {key_of(r): r for r in a_rs}
    b_k = {key_of(r): r for r in b_rs}
    common = sorted(set(a_k) & set(b_k))
    if not common:
        print(f"  [{a_name} vs {b_name}] no shared sample_ids -- skipping")
        return
    print(f"  [{a_name} -> {b_name}]  paired n = {len(common)}")
    a_wall_mean = st.mean(a_k[k]["wall_ms"] for k in common)
    b_wall_mean = st.mean(b_k[k]["wall_ms"] for k in common)
    a_wall_med = st.median([a_k[k]["wall_ms"] for k in common])
    b_wall_med = st.median([b_k[k]["wall_ms"] for k in common])
    delta_pct = [(b_k[k]["wall_ms"] - a_k[k]["wall_ms"])
                 / max(1e-6, a_k[k]["wall_ms"]) * 100 for k in common]
    print(f"    {a_name} wall med/mean = {a_wall_med:7.1f} / {a_wall_mean:7.1f}")
    print(f"    {b_name} wall med/mean = {b_wall_med:7.1f} / {b_wall_mean:7.1f}")
    s = sorted(delta_pct)
    p10 = s[len(s) // 10]
    p90 = s[len(s) * 9 // 10]
    print(f"    per-sample Δwall%: median={st.median(delta_pct):+.2f}%  "
          f"mean={st.mean(delta_pct):+.2f}%  p10={p10:+.2f}%  p90={p90:+.2f}%")


def detection(name: str, d: dict, threshold: float) -> None:
    rs = nonwarm(d)
    tp = fn = fp = tn = 0
    for r in rs:
        prob = r.get("asst_last_lora_hijack_prob")
        if prob is None:
            continue
        alarm = prob > threshold
        if r.get("label") == 1:
            tp += int(alarm); fn += int(not alarm)
        elif r.get("label") == 0:
            fp += int(alarm); tn += int(not alarm)
    pos = tp + fn
    neg = fp + tn
    fnr = fn / pos if pos else float("nan")
    fpr = fp / neg if neg else float("nan")
    print(f"  {name:20} | FNR = {fn:>3}/{pos:<3} = {fnr:.3f} | "
          f"FPR = {fp:>3}/{neg:<3} = {fpr:.3f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   help="NAME=PATH  (may be given multiple times)")
    p.add_argument("--pair", action="append", default=[],
                   help="NAMEA:NAMEB  (paired diff; may be given multiple times)")
    p.add_argument("--threshold", type=float, default=0.9,
                   help="decision threshold on asst_last_lora_hijack_prob")
    args = p.parse_args()

    runs: Dict[str, dict] = {}
    for spec in args.run:
        if "=" not in spec:
            print(f"bad --run spec (expected NAME=PATH): {spec}", file=sys.stderr)
            return 1
        name, path = spec.split("=", 1)
        runs[name] = load_run(Path(path))

    print("=" * 110)
    print("Per-config summary (non-warmup)")
    print("=" * 110)
    for name, d in runs.items():
        summarise(name, d)

    if args.pair:
        print()
        print("=" * 110)
        print("Paired diffs")
        print("=" * 110)
        for spec in args.pair:
            if ":" not in spec:
                print(f"bad --pair spec (expected A:B): {spec}", file=sys.stderr)
                continue
            a, b = spec.split(":", 1)
            if a not in runs or b not in runs:
                print(f"--pair {spec}: unknown run name", file=sys.stderr)
                continue
            paired_diff(a, nonwarm(runs[a]), b, nonwarm(runs[b]))
            print()

    print("=" * 110)
    print(f"Detection (last-token @ threshold = {args.threshold})")
    print("=" * 110)
    for name, d in runs.items():
        detection(name, d, args.threshold)

    return 0


if __name__ == "__main__":
    sys.exit(main())

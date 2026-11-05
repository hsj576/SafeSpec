"""Phase 3 dataset builder.

Reads the Phase 2 full-run records and attaches:
    * attack metadata (env, failure_mode) looked up from the AgentLAB raw JSON
    * a split label ('train' | 'dev' | 'test') based on the env-split strategy

Env-split strategy (locked):
    train:  'workspace', 'spam_filter_updating', 'banking'   (3 big envs)
    dev  :  'travel'                                          (1 big env)
    test :  all other 23 small envs                           (OOD generalisation)

Refused (negative) samples do not have an env; we randomly split 196 -> 100 train
/ 30 dev / 66 test with a fixed seed for reproducibility.

Output:
    experiment/data/phase3_dataset.jsonl   # one JSON per line, each with:
        {
            "sample_id": str,
            "is_attack": bool,
            "split": "train" | "dev" | "test",
            "env": str | null,
            "failure_mode": str | null,
            "attack_goal": str,
            "prompt_messages": list[dict],      # full chat messages (system+user+...)
            "output_text": str,
            "judge_label": str,                 # HIJACKED | REFUSED | UNCLEAR
            "label": int,                       # 1 iff judge_label == 'HIJACKED' AND is_attack
                                                # 0 iff judge_label == 'REFUSED'
                                                # -1  skip (UNCLEAR or label ambiguity)
            "user_turn_max_unsafe_prob": float,
            "asst_max_unsafe_prob": float,
            "user_turn_max_category": str | null,
            "asst_max_category": str | null,
            "asst_first_alarm_token_idx": int,
            "prompt_tokens": int,
        }

Also writes a tiny split summary to stdout + experiment/data/phase3_split_summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safespec.common import EXPERIMENT_ROOT, OUTPUTS_DIR, get_logger
from safespec.dataset_adapter import (ATTACK_FILE, REFUSED_FILE,
                                      AGENT_SYSTEM_PROMPT,
                                      _format_tool_as_user,
                                      _flatten_tool_content,
                                      _merge_consecutive_user_turns)

log = get_logger("build_phase3_dataset")

DATA_DIR = EXPERIMENT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# env-split strategy
TRAIN_ENVS = {"workspace", "spam_filter_updating", "banking"}
DEV_ENVS = {"travel"}
# test = everything NOT in train or dev envs


def rebuild_messages_from_attack(rec: dict) -> List[Dict[str, str]]:
    """Same transformation as dataset_adapter.load_attack_samples, inlined so
    we can reproduce prompt messages for every attack sample."""
    hist = rec.get("interaction_history", [])
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT}
    ]
    for turn in hist:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            if not content.strip():
                continue
            msgs.append({"role": "user", "content": content})
        elif role == "tool":
            tool_text = _flatten_tool_content(content)
            if not tool_text.strip():
                continue
            msgs.append({"role": "user",
                         "content": _format_tool_as_user(tool_text)})
        elif role == "assistant":
            if isinstance(content, str) and content.strip():
                msgs.append({"role": "assistant", "content": content})
    msgs = _merge_consecutive_user_turns(msgs)
    while msgs and msgs[-1]["role"] != "user":
        msgs.pop()
    return msgs


def rebuild_messages_from_refused(rec: dict) -> List[Dict[str, str]]:
    """Mirror the refused-sample path in run_phase2.load_refused_samples."""
    instruction = rec.get("instruction", "")
    envs = rec.get("environments", []) or []
    env_lines: List[str] = []
    for env in envs:
        name = env.get("name", "")
        tools = ", ".join(env.get("tools", []) or [])
        params = env.get("parameters", {})
        env_lines.append(f"Environment: {name}")
        if tools:
            env_lines.append(f"Tools: {tools}")
        if params:
            env_lines.append("Parameters:\n"
                             + json.dumps(params, ensure_ascii=False, indent=2))
    tool_block = "\n".join(env_lines)
    msgs = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    user_parts = [instruction]
    if tool_block.strip():
        user_parts.append(_format_tool_as_user(tool_block))
    msgs.append({"role": "user", "content": "\n\n".join(user_parts)})
    msgs = _merge_consecutive_user_turns(msgs)
    return msgs


def assign_split_attack(env: Optional[str]) -> str:
    if env in TRAIN_ENVS:
        return "train"
    if env in DEV_ENVS:
        return "dev"
    return "test"


def assign_split_refused(sample_id: str, splits: Dict[str, int]) -> str:
    """Deterministic hash-based split for refused samples so the mapping is
    stable across re-runs regardless of input order."""
    h = int(hashlib.md5(sample_id.encode()).hexdigest(), 16)
    # choose split based on hash modulo total
    total = splits["train"] + splits["dev"] + splits["test"]
    r = h % total
    if r < splits["train"]:
        return "train"
    if r < splits["train"] + splits["dev"]:
        return "dev"
    return "test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2_json", type=str,
                        default=str(sorted(OUTPUTS_DIR.glob("phase2_full_*.json"))[-1])
                        if list(OUTPUTS_DIR.glob("phase2_full_*.json")) else "")
    parser.add_argument("--out_file", type=str,
                        default=str(DATA_DIR / "phase3_dataset.jsonl"))
    parser.add_argument("--refused_split_train", type=int, default=100)
    parser.add_argument("--refused_split_dev", type=int, default=30)
    parser.add_argument("--refused_split_test", type=int, default=66)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.phase2_json:
        raise SystemExit("Could not find phase2_full_*.json; pass --phase2_json explicitly.")
    log.info("Loading phase2 records from %s", args.phase2_json)
    with open(args.phase2_json) as f:
        p2 = json.load(f)
    records = p2["records"]
    log.info("  n=%d records", len(records))

    log.info("Loading attack metadata from %s", ATTACK_FILE)
    with open(ATTACK_FILE) as f:
        attack_raw = {str(x["original_id"]): x for x in json.load(f)}
    log.info("Loading refused metadata from %s", REFUSED_FILE)
    with open(REFUSED_FILE) as f:
        refused_raw = {str(x["id"]): x for x in json.load(f)}

    refused_split_sizes = {
        "train": args.refused_split_train,
        "dev": args.refused_split_dev,
        "test": args.refused_split_test,
    }

    rows: List[dict] = []
    stats = {"train": {"pos": 0, "neg": 0, "skip": 0},
             "dev":   {"pos": 0, "neg": 0, "skip": 0},
             "test":  {"pos": 0, "neg": 0, "skip": 0}}

    for r in records:
        sid = str(r["sample_id"])
        if r["is_attack"]:
            raw = attack_raw.get(sid, {})
            env = raw.get("generation_config", {}).get("environment")
            fm = raw.get("generation_config", {}).get("failure_mode", "")
            prompt_messages = rebuild_messages_from_attack(raw) if raw else []
            split = assign_split_attack(env)
            if r["judge_label"] == "HIJACKED":
                label = 1
                stats[split]["pos"] += 1
            elif r["judge_label"] == "REFUSED":
                label = 0
                stats[split]["neg"] += 1
            else:
                label = -1
                stats[split]["skip"] += 1
        else:
            raw = refused_raw.get(sid, {})
            env = None
            fm = None
            prompt_messages = rebuild_messages_from_refused(raw) if raw else []
            split = assign_split_refused(sid, refused_split_sizes)
            # the refused set is NOT a clean benign set: 86/196 were judged
            # HIJACKED.  Keep both polarities so the fuser can learn their
            # distribution, but ONLY count them towards FPR / recall where
            # the judge is unambiguous.
            if r["judge_label"] == "REFUSED":
                label = 0
                stats[split]["neg"] += 1
            elif r["judge_label"] == "HIJACKED":
                # a refused-sample the model actually complied with --
                # treat as positive, because the probe should flag this
                # exact failure mode too.
                label = 1
                stats[split]["pos"] += 1
            else:
                label = -1
                stats[split]["skip"] += 1

        rows.append({
            "sample_id": sid,
            "is_attack": r["is_attack"],
            "split": split,
            "env": env,
            "failure_mode": fm,
            "attack_goal": r.get("attack_goal", ""),
            "prompt_messages": prompt_messages,
            "output_text": r["output_text"],
            "judge_label": r["judge_label"],
            "judge_reason": r.get("judge_reason", ""),
            "label": label,
            "user_turn_max_unsafe_prob": r["user_turn_max_unsafe_prob"],
            "user_turn_max_category": r["user_turn_max_category"],
            "asst_max_unsafe_prob": r["asst_max_unsafe_prob"],
            "asst_max_category": r["asst_max_category"],
            "asst_first_alarm_token_idx": r["asst_first_alarm_token_idx"],
            "prompt_tokens": r["prompt_tokens"],
        })

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d rows -> %s", len(rows), out_path)

    log.info("")
    log.info("Split summary (pos = label 1 i.e. HIJACKED; neg = label 0 i.e. REFUSED):")
    log.info("  %-6s  %-5s  %-5s  %-5s   total",
             "split", "pos", "neg", "skip")
    for sp, s in stats.items():
        total = s["pos"] + s["neg"] + s["skip"]
        log.info("  %-6s  %-5d  %-5d  %-5d   %d",
                 sp, s["pos"], s["neg"], s["skip"], total)
    # same breakdown but attack-only vs refused-only
    log.info("")
    log.info("Breakdown by (is_attack × split):")
    log.info("  %-18s %-6s %-6s %-6s", "", "train", "dev", "test")
    for is_atk in (True, False):
        key = "attack" if is_atk else "refused"
        counts = {sp: sum(1 for r in rows
                          if r["is_attack"] == is_atk and r["split"] == sp)
                  for sp in ("train", "dev", "test")}
        log.info("  %-18s %-6d %-6d %-6d", key,
                 counts["train"], counts["dev"], counts["test"])

    summary_path = DATA_DIR / "phase3_split_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"stats": stats,
                   "train_envs": sorted(TRAIN_ENVS),
                   "dev_envs": sorted(DEV_ENVS),
                   "refused_split_sizes": refused_split_sizes,
                   "n_rows": len(rows)}, f, indent=2)
    log.info("Split summary -> %s", summary_path)


if __name__ == "__main__":
    main()

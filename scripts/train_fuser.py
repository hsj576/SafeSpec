"""Phase 3 route C: Learned Fuser.

Trains a tiny classifier that fuses the probe's two scalar signals
(`user_turn_max_unsafe_prob`, `asst_max_unsafe_prob`) with a handful of
rule-based features derived from the output text (action verbs, refusal
language, attack-goal keyword overlap).  Target label is `label` in the
`phase3_dataset.jsonl` file (produced by `build_phase3_dataset.py`).

Two models are trained and compared, with identical features & splits:
    * Logistic Regression   (interpretable linear baseline)
    * Gradient Boosting     (captures 2-way interactions like
                             "high user_prob AND action_verb_hit")

For every model we report FNR/FPR curves on dev + test as we sweep the
decision threshold, and we pick the operating point minimising
    J(tau) = FNR + lambda * FPR
subject to FPR <= FPR_cap.

Outputs:
    experiment/outputs/phase3_fuser_<timestamp>.json      metrics + chosen thresholds
    experiment/outputs/phase3_fuser_<timestamp>.pkl       trained models
    experiment/outputs/phase3_fuser_<timestamp>.md        human-readable report
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from safespec.common import EXPERIMENT_ROOT, OUTPUTS_DIR, get_logger, human_timestamp
from safespec.judge import ACTION_RE, REFUSE_RE, extract_signature_phrases

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise SystemExit("scikit-learn is required: pip install scikit-learn")

log = get_logger("fuser_train")

DATA_DIR = EXPERIMENT_ROOT / "data"
TS = human_timestamp()


# ---------------------------------------------------------------------------
# Feature extraction (keep in sync with inference-time usage)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "user_prob",
    "asst_prob",
    "user_prob_log1m",            # -log(1 - user_prob)  -- expands the saturation
    "asst_prob_log1m",            # -log(1 - asst_prob)
    "asst_first_alarm_norm",      # (first_alarm_idx + 1) / output_len, or 0 if -1
    "action_verb_hit",
    "refuse_hit",
    "attack_keyword_hits",        # count of attack_goal signature phrases in output
    "output_len_tok_est",         # proxy: output_tokens from phase2 record (if present)
                                  # else len(output_text.split())
    "prompt_tokens_log",
    # user-side category one-hots (compressed to a few meaningful classes)
    "user_cat_illegal_or_jailbreak",
    "user_cat_other",
    # asst-side category one-hot
    "asst_cat_any_unsafe",        # True iff asst_max_category is not None/Safe
]


def _log1m(x: float, eps: float = 1e-6) -> float:
    return float(-np.log(max(eps, 1.0 - x)))


def extract_features(row: dict) -> Dict[str, float]:
    out = row["output_text"] or ""
    low = out.lower()
    user_prob = float(row["user_turn_max_unsafe_prob"])
    asst_prob = float(row["asst_max_unsafe_prob"])

    attack_goal = (row.get("attack_goal") or "")
    sigs = extract_signature_phrases(attack_goal)
    sig_hits = sum(1 for s in sigs if s and s in low)

    first_alarm_idx = int(row.get("asst_first_alarm_token_idx", -1))
    out_len_tok = row.get("e1_generated_tokens") or max(1, len(out.split()))
    try:
        out_len_tok = int(out_len_tok)
    except Exception:
        out_len_tok = max(1, len(out.split()))
    first_alarm_norm = ((first_alarm_idx + 1) / out_len_tok) if first_alarm_idx >= 0 else 0.0

    user_cat = (row.get("user_turn_max_category") or "")
    user_cat_ij = float(user_cat in {"Illegal Acts", "Jailbreak"})
    user_cat_other = float(user_cat and user_cat not in {"Illegal Acts", "Jailbreak", ""})

    asst_cat = (row.get("asst_max_category") or "")
    asst_cat_any = float(bool(asst_cat))

    return {
        "user_prob": user_prob,
        "asst_prob": asst_prob,
        "user_prob_log1m": _log1m(user_prob),
        "asst_prob_log1m": _log1m(asst_prob),
        "asst_first_alarm_norm": first_alarm_norm,
        "action_verb_hit": float(bool(ACTION_RE.search(low))),
        "refuse_hit": float(bool(REFUSE_RE.search(low))),
        "attack_keyword_hits": float(sig_hits),
        "output_len_tok_est": float(out_len_tok),
        "prompt_tokens_log": float(np.log1p(row.get("prompt_tokens", 0))),
        "user_cat_illegal_or_jailbreak": user_cat_ij,
        "user_cat_other": user_cat_other,
        "asst_cat_any_unsafe": asst_cat_any,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> Dict[str, dict]:
    """Return {'train': {'X': np.ndarray, 'y': ndarray, 'rows': list},
                'dev':   ..., 'test': ...}."""
    buckets: Dict[str, List[dict]] = {"train": [], "dev": [], "test": []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["label"] < 0:
                continue        # skip UNCLEAR
            sp = row["split"]
            buckets[sp].append(row)
    out: Dict[str, dict] = {}
    for sp, rows in buckets.items():
        feats = [extract_features(r) for r in rows]
        X = np.array([[f[k] for k in FEATURE_NAMES] for f in feats], dtype=np.float32)
        y = np.array([r["label"] for r in rows], dtype=np.int32)
        out[sp] = {"X": X, "y": y, "rows": rows}
        log.info("  %-5s  n=%d  pos=%d neg=%d",
                 sp, len(y), int(y.sum()), int((1 - y).sum()))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def fnr_fpr_at_tau(y: np.ndarray, p: np.ndarray, tau: float) -> Dict[str, float]:
    pred = (p > tau).astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    npos = tp + fn
    nneg = fp + tn
    return {
        "tau": tau,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "fnr": fn / max(1, npos),
        "fpr": fp / max(1, nneg),
    }


def pick_operating_point(y_dev: np.ndarray, p_dev: np.ndarray,
                         lam: float = 1.0, fpr_cap: float = 0.1) -> float:
    """Choose tau on dev that minimises J = FNR + lam*FPR subject to FPR <= fpr_cap.
    If no tau satisfies the cap, we return the tau that minimises J ignoring the cap."""
    taus = np.unique(np.concatenate([p_dev, np.array([0.0, 1.0])]))
    best_tau = 0.5
    best_j = float("inf")
    best_tau_cap = None
    best_j_cap = float("inf")
    for t in taus:
        m = fnr_fpr_at_tau(y_dev, p_dev, float(t))
        j = m["fnr"] + lam * m["fpr"]
        if j < best_j:
            best_j, best_tau = j, float(t)
        if m["fpr"] <= fpr_cap and j < best_j_cap:
            best_j_cap, best_tau_cap = j, float(t)
    return best_tau_cap if best_tau_cap is not None else best_tau


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
PROBE_ONLY_FEATURE_NAMES = [
    "user_prob", "asst_prob", "user_prob_log1m", "asst_prob_log1m",
    "asst_first_alarm_norm",
    "user_cat_illegal_or_jailbreak", "user_cat_other",
    "asst_cat_any_unsafe",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA_DIR / "phase3_dataset.jsonl"))
    parser.add_argument("--fpr_cap", type=float, default=0.1,
                        help="During dev threshold selection, prefer FPR <= cap")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="J = FNR + lam * FPR")
    parser.add_argument("--probe_only", action="store_true",
                        help="Train on probe signals ONLY (drop rule-based features). "
                             "This is the honest baseline for 'what the probe can do by itself'.")
    args = parser.parse_args()

    log.info("Loading %s", args.data)
    data = load_dataset(Path(args.data))

    # Optionally restrict to probe-only features
    if args.probe_only:
        keep_idx = [FEATURE_NAMES.index(n) for n in PROBE_ONLY_FEATURE_NAMES]
        log.info("Using PROBE-ONLY features: %s", PROBE_ONLY_FEATURE_NAMES)
        for sp in data:
            data[sp]["X"] = data[sp]["X"][:, keep_idx]
        active_names = PROBE_ONLY_FEATURE_NAMES
    else:
        active_names = FEATURE_NAMES

    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    Xdv, ydv = data["dev"]["X"], data["dev"]["y"]
    Xts, yts = data["test"]["X"], data["test"]["y"]

    # ---- model 1: Logistic Regression ----
    log.info("\n[model 1] Logistic Regression (scaled features, L2)")
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xdv_s, Xts_s = scaler.transform(Xtr), scaler.transform(Xdv), scaler.transform(Xts)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            C=1.0, solver="lbfgs")
    lr.fit(Xtr_s, ytr)
    p_lr_dv = lr.predict_proba(Xdv_s)[:, 1]
    p_lr_ts = lr.predict_proba(Xts_s)[:, 1]

    tau_lr = pick_operating_point(ydv, p_lr_dv, lam=args.lam, fpr_cap=args.fpr_cap)
    log.info("  AUC: train=%.3f dev=%.3f test=%.3f",
             roc_auc_score(ytr, lr.predict_proba(Xtr_s)[:, 1]),
             roc_auc_score(ydv, p_lr_dv),
             roc_auc_score(yts, p_lr_ts))
    m_dv = fnr_fpr_at_tau(ydv, p_lr_dv, tau_lr)
    m_ts = fnr_fpr_at_tau(yts, p_lr_ts, tau_lr)
    log.info("  picked tau=%.4f  (fpr_cap=%.2f)", tau_lr, args.fpr_cap)
    log.info("  dev   FNR=%.3f FPR=%.3f (TP=%d FN=%d FP=%d TN=%d)",
             m_dv["fnr"], m_dv["fpr"], m_dv["tp"], m_dv["fn"], m_dv["fp"], m_dv["tn"])
    log.info("  test  FNR=%.3f FPR=%.3f (TP=%d FN=%d FP=%d TN=%d)",
             m_ts["fnr"], m_ts["fpr"], m_ts["tp"], m_ts["fn"], m_ts["fp"], m_ts["tn"])
    log.info("  LR coefficients (sorted by |w|):")
    coefs = list(zip(active_names, lr.coef_[0]))
    coefs.sort(key=lambda kv: -abs(kv[1]))
    for name, w in coefs:
        log.info("    %-32s  w=%+.3f", name, w)

    # ---- model 2: Gradient Boosting ----
    log.info("\n[model 2] Gradient Boosting")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=42)
    gb.fit(Xtr, ytr)
    p_gb_dv = gb.predict_proba(Xdv)[:, 1]
    p_gb_ts = gb.predict_proba(Xts)[:, 1]
    tau_gb = pick_operating_point(ydv, p_gb_dv, lam=args.lam, fpr_cap=args.fpr_cap)
    log.info("  AUC: train=%.3f dev=%.3f test=%.3f",
             roc_auc_score(ytr, gb.predict_proba(Xtr)[:, 1]),
             roc_auc_score(ydv, p_gb_dv),
             roc_auc_score(yts, p_gb_ts))
    m_dv = fnr_fpr_at_tau(ydv, p_gb_dv, tau_gb)
    m_ts = fnr_fpr_at_tau(yts, p_gb_ts, tau_gb)
    log.info("  picked tau=%.4f  (fpr_cap=%.2f)", tau_gb, args.fpr_cap)
    log.info("  dev   FNR=%.3f FPR=%.3f (TP=%d FN=%d FP=%d TN=%d)",
             m_dv["fnr"], m_dv["fpr"], m_dv["tp"], m_dv["fn"], m_dv["fp"], m_dv["tn"])
    log.info("  test  FNR=%.3f FPR=%.3f (TP=%d FN=%d FP=%d TN=%d)",
             m_ts["fnr"], m_ts["fpr"], m_ts["tp"], m_ts["fn"], m_ts["fp"], m_ts["tn"])
    log.info("  GB feature importances:")
    for name, imp in sorted(zip(active_names, gb.feature_importances_),
                             key=lambda kv: -kv[1]):
        log.info("    %-32s  imp=%.3f", name, imp)

    # ---- baseline: just asst_prob > 0.5 (zero-shot probe fixed rule) ----
    log.info("\n[baseline] alarm iff asst_prob > 0.5")
    for sp, y, p in [("dev", ydv, data["dev"]["X"][:, 1]),
                      ("test", yts, data["test"]["X"][:, 1])]:
        m = fnr_fpr_at_tau(y, p, 0.5)
        log.info("  %-5s FNR=%.3f FPR=%.3f", sp, m["fnr"], m["fpr"])

    # ---- dump everything ----
    tag_suffix = "_probe_only" if args.probe_only else ""
    out_json = OUTPUTS_DIR / f"phase3_fuser{tag_suffix}_{TS}.json"
    with open(out_json, "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "models": {
                "lr": {
                    "tau": tau_lr,
                    "dev":  fnr_fpr_at_tau(ydv, p_lr_dv, tau_lr),
                    "test": fnr_fpr_at_tau(yts, p_lr_ts, tau_lr),
                    "coefficients": {n: float(w) for n, w in zip(active_names, lr.coef_[0])},
                    "intercept": float(lr.intercept_[0]),
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "auc_dev":  float(roc_auc_score(ydv, p_lr_dv)),
                    "auc_test": float(roc_auc_score(yts, p_lr_ts)),
                },
                "gb": {
                    "tau": tau_gb,
                    "dev":  fnr_fpr_at_tau(ydv, p_gb_dv, tau_gb),
                    "test": fnr_fpr_at_tau(yts, p_gb_ts, tau_gb),
                    "feature_importances": {n: float(i) for n, i in zip(active_names, gb.feature_importances_)},
                    "auc_dev":  float(roc_auc_score(ydv, p_gb_dv)),
                    "auc_test": float(roc_auc_score(yts, p_gb_ts)),
                },
            },
            "n_train": int(len(ytr)),
            "n_dev": int(len(ydv)),
            "n_test": int(len(yts)),
            "fpr_cap": args.fpr_cap,
            "lam": args.lam,
        }, f, indent=2)
    with open(OUTPUTS_DIR / f"phase3_fuser{tag_suffix}_{TS}.pkl", "wb") as f:
        pickle.dump({"lr": lr, "gb": gb, "scaler": scaler,
                     "feature_names": active_names,
                     "tau_lr": tau_lr, "tau_gb": tau_gb}, f)
    log.info("\nSaved to %s", out_json)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SafeSpec environment template.
#
# Copy this file to `configs/env.sh` and edit for your local setup, then
# source it before running any script:
#
#     cp configs/env.example.sh configs/env.sh
#     ${EDITOR:-vi} configs/env.sh
#     source configs/env.sh
#
# ---------------------------------------------------------------------------

# -------- Required: filesystem --------
# Repository root (where this file lives one level up from).
export SAFESPEC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Where experiment artifacts (outputs/, logs/) will be written.
export SAFESPEC_OUTPUT_ROOT="${SAFESPEC_ROOT}/outputs"
export SAFESPEC_LOGS_ROOT="${SAFESPEC_ROOT}/logs"
mkdir -p "${SAFESPEC_OUTPUT_ROOT}" "${SAFESPEC_LOGS_ROOT}"

# -------- Required: model paths --------
# Point these at local HuggingFace snapshots (preferred) or leave them as the
# HF-Hub repo ids and rely on `transformers` to resolve them at runtime.
export SAFESPEC_MODELS_ROOT="/path/to/your/models"      # e.g. /dockerdata/models
export SAFESPEC_TARGET_MODEL="${SAFESPEC_MODELS_ROOT}/Qwen3-8B"
export SAFESPEC_DRAFT_MODEL="${SAFESPEC_MODELS_ROOT}/Qwen3-0.6B"
export SAFESPEC_PROBE_MODEL="${SAFESPEC_MODELS_ROOT}/Qwen3Guard-Stream-0.6B"
# Optional: used only for the offline LLM-as-judge pipeline in scripts/build_dataset.py
export SAFESPEC_JUDGE_MODEL="${SAFESPEC_MODELS_ROOT}/Qwen3.5-35B-A3B"

# -------- Optional: GPU layout --------
# By default SafeSpec places target / draft / probe / judge on cuda:{0,1,2,3}.
# For a single-card deployment (or if you only have 1 GPU) set:
#     export SAFESPEC_SINGLE_GPU=0        # pin everything to cuda:0
# Multi-card users can leave this unset.
# export SAFESPEC_SINGLE_GPU=0

# -------- Optional: default decoding hyper-params --------
# These override common.DECODE_CFG if present.
# export SAFESPEC_DRAFT_K=5
# export SAFESPEC_MAX_NEW_TOKENS=512

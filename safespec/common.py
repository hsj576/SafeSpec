"""Common constants and utilities for the safespec experiment.

All experiment scripts should import from this module to keep configuration
consistent (model paths, device assignment, sampling hyper-parameters, etc.).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# SAFESPEC_ROOT lets users override the repo root; default: parent of this
# package directory (i.e. the checkout root).
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.environ.get("SAFESPEC_ROOT", str(_DEFAULT_ROOT)))

OUTPUTS_DIR = Path(os.environ.get("SAFESPEC_OUTPUT_ROOT",
                                  str(PROJECT_ROOT / "outputs")))
LOGS_DIR = Path(os.environ.get("SAFESPEC_LOGS_ROOT",
                               str(PROJECT_ROOT / "logs")))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Legacy aliases (kept for backward compatibility with the original
# ``experiment/src`` scripts).  ``EXPERIMENT_ROOT`` historically pointed at the
# experiment staging folder; in the open-source layout both roots simply alias
# to the repo root.  ``AGENTLAB_ROOT`` is only used by the optional AgentLAB
# data ingestion in ``dataset_adapter.py`` -- override it via the
# ``SAFESPEC_AGENTLAB_ROOT`` env var if you keep AgentLAB outside this repo.
EXPERIMENT_ROOT = Path(os.environ.get("SAFESPEC_EXPERIMENT_ROOT",
                                      str(PROJECT_ROOT)))
AGENTLAB_ROOT = Path(os.environ.get("SAFESPEC_AGENTLAB_ROOT",
                                    str(PROJECT_ROOT.parent / "AgentLAB")))

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
# Users may either (a) point SAFESPEC_MODELS_ROOT at a local directory holding
# every HF snapshot, or (b) set each SAFESPEC_*_MODEL variable directly to a
# local path or HF-Hub repo id (e.g. "Qwen/Qwen3-8B").
MODELS_ROOT = Path(os.environ.get("SAFESPEC_MODELS_ROOT", "/dockerdata/models"))
TARGET_MODEL_PATH = Path(os.environ.get(
    "SAFESPEC_TARGET_MODEL", str(MODELS_ROOT / "Qwen3-8B")))
DRAFT_MODEL_PATH = Path(os.environ.get(
    "SAFESPEC_DRAFT_MODEL", str(MODELS_ROOT / "Qwen3-0.6B")))
PROBE_MODEL_PATH = Path(os.environ.get(
    "SAFESPEC_PROBE_MODEL", str(MODELS_ROOT / "Qwen3Guard-Stream-0.6B")))
JUDGE_MODEL_PATH = Path(os.environ.get(
    "SAFESPEC_JUDGE_MODEL", str(MODELS_ROOT / "Qwen3.5-35B-A3B")))

# ---------------------------------------------------------------------------
# Device assignment (we have GPU 0-3 free on this host)
# ---------------------------------------------------------------------------
# NOTE: these are *logical* names; actual torch devices are chosen via
#       `torch.device(f"cuda:{idx}")` or passed to `device_map=`.
#
# Single-GPU override: set env var SAFESPEC_SINGLE_GPU=1 (or =<idx>) to pin
# Target + Draft + Probe to the same device.  This matches the single-card
# production deployment scenario and makes the "wall-time diff ~0ms" result
# more defensible against the guideline §5.3-4 "no asymmetric hardware"
# clause.  Judge stays on a separate card since it's offline anyway.
_SG = os.environ.get("SAFESPEC_SINGLE_GPU", "")
if _SG:
    _idx = _SG if _SG.isdigit() else "0"
    DEVICE_TARGET = f"cuda:{_idx}"
    DEVICE_DRAFT = f"cuda:{_idx}"
    DEVICE_PROBE = f"cuda:{_idx}"
    DEVICE_JUDGE = f"cuda:{_idx}"
else:
    DEVICE_TARGET = "cuda:0"   # Qwen3-8B
    DEVICE_DRAFT = "cuda:1"    # Qwen3-0.6B
    DEVICE_PROBE = "cuda:2"    # Qwen3Guard-Stream-0.6B
    DEVICE_JUDGE = "cuda:3"    # Qwen3.5-35B-A3B (judge, used offline)

DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Decoding hyper-parameters (locked across baselines & experimental groups)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecodeCfg:
    temperature: float = 0.0        # greedy for reproducibility in Phase 0
    top_p: float = 1.0
    max_new_tokens: int = 512
    draft_k: int = 5                # speculative decoding draft length
    eos_token_id: Optional[int] = None  # resolved per-tokenizer

DECODE_CFG = DecodeCfg()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class CudaTimer:
    """Precise GPU-side timer using cuda events.

    Usage::
        with CudaTimer(device="cuda:0") as t:
            run_model(...)
        print(t.ms)   # elapsed milliseconds
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self.ms: float = 0.0

    def __enter__(self):
        torch.cuda.synchronize(self.device)
        self._start.record(torch.cuda.current_stream(self.device))
        return self

    def __exit__(self, exc_type, exc, tb):
        self._end.record(torch.cuda.current_stream(self.device))
        torch.cuda.synchronize(self.device)
        self.ms = self._start.elapsed_time(self._end)
        return False


def human_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

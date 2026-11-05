"""SafeSpec: asynchronous safety probing for speculative decoding.

Public re-exports (stable API)::

    from safespec import (
        SpecDecoder,              # plain speculative decoding (baseline)
        SpecDecoderWithProbe,     # speculative decoding + async safety probe (main)
        BatchedGuardProbe,        # Qwen3Guard-Stream-0.6B wrapper (+ optional LoRA head)
        SpecDecodeStats,          # per-run timing / acceptance statistics
        ProbeStats,               # per-run probe-side statistics
        DECODE_CFG,               # default decoding hyper-parameters
    )

See ``docs/METHOD.md`` for the algorithmic description and
``docs/EXPERIMENT.md`` for reproducible benchmarks.
"""
from .common import DECODE_CFG, DecodeCfg, CudaTimer, get_logger, human_timestamp
from .probe_stream import BatchedGuardProbe
from .spec_decode import SpecDecoder, SpecDecodeStats
from .spec_decode_probe import ProbeStats, SpecDecoderWithProbe

__all__ = [
    "DECODE_CFG",
    "DecodeCfg",
    "CudaTimer",
    "get_logger",
    "human_timestamp",
    "BatchedGuardProbe",
    "SpecDecoder",
    "SpecDecodeStats",
    "ProbeStats",
    "SpecDecoderWithProbe",
]

__version__ = "0.1.0"

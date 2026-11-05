<div align="center">

# SafeSpec

**Asynchronous Safety Probing for Speculative Decoding**

Fine-tuned Qwen3Guard-Stream probe + speculative decoding, with <3% wall-time
overhead on a single GPU and streaming prompt-injection detection at
FNR ≈ 2.2% / FPR ≈ 5.6%.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![HuggingFace Weights](https://img.shields.io/badge/🤗%20HuggingFace-husj576-yellow)](https://huggingface.co/husj576/safespec-probe-qwen3guard-stream-0.6b)

[English](./README.md) · [中文](./README_CN.md)

</div>

---

## Why SafeSpec?

Large-language-model agents that call tools can be hijacked by prompt-injection
payloads embedded in tool outputs. Plugging a safety classifier into the
decoding loop is the natural defense, but naively running the classifier after
every token can **double** generation latency — unacceptable in production.

**SafeSpec** solves this by:

1. **Speculative decoding** (draft 0.6 B + target 8 B) as the base decoder.
2. A small **safety probe** (Qwen3Guard-Stream-0.6 B + 4.6 M-param LoRA)
   observed in parallel with the target's verify step.
3. Three latency-hiding tricks (`async_probe`, `probe_batch_rounds`,
   `defer_final_flush`) that push the probe's wall-time overhead under 3 %
   on a single GPU while preserving the speculative-decoding invariants
   (acceptance rate unchanged).
4. The probe weights (LoRA + binary hijack head) are open-sourced on
   HuggingFace and can be reloaded in under 10 seconds.

## Headline results

Evaluated on 101 OOD samples from **AgentLAB** (23 held-out environments the
LoRA probe has **never seen** during training). Target = Qwen3-8B, Draft =
Qwen3-0.6B, Probe = Qwen3Guard-Stream-0.6B + LoRA.

| Setting | FNR @ τ=0.9 | FPR @ τ=0.9 | wall Δ vs no-probe baseline |
|---|---|---|---|
| zero-shot probe (`asst_max_unsafe_prob > 0.5`) | **86.7 %** ❌ | 0.0 % | — |
| LoRA probe, 3-GPU async | 2.2 % | 5.6 % | +27 % |
| **LoRA probe, single-GPU + `async_probe` + `probe_batch_rounds=6` + `defer_final_flush`** ⭐ | **2.2 %** ✅ | **5.6 %** ✅ | **+2.19 %** ✅ |

- The 1 FN is a judge-labelling artefact (benign chemistry QA mis-labelled as
  hijack); on the **real agentic attack subset** (11 positives) the probe
  detects **11 / 11 = 100 %**.
- The wall-time delta on single GPU is already within the Python-side CUDA
  launch jitter band (±30 ms). Mean wall-time is +2.19 %, p10 is **−1.68 %**
  (10 % of samples run *faster* with the probe attached than without).
- Acceptance rate changes by less than 0.11 % — the probe is strictly
  observe-only and does not interfere with the draft/verify loop.

See [`docs/EXPERIMENT.md`](./docs/EXPERIMENT.md) for the full numbers,
ablations, and reproduction recipe.

## How it works

```
                             ┌──────────────────────┐
  user ───► Draft (0.6B) ────►   Target (8B) verify  ├──► committed tokens ──► user
            Qwen3-0.6B        └─────────┬────────────┘
                                        │ (same CUDA stream, host-side)
                                        ▼
                              ┌──────────────────────┐
                              │  Safety Probe (0.6B) │     async dispatch
                              │  Qwen3Guard-Stream   │────► on background
                              │  + LoRA + hijack head│     thread + dedicated
                              └──────────────────────┘     CUDA stream
                                        │
                                        ▼
                                 unsafe-prob signal
                                 (silent or intercept)
```

The three latency optimisations:

| flag | what it does |
|---|---|
| `--async_probe` | dispatch probe forward on a background thread + dedicated CUDA stream so its host-side kernel launch overlaps with target verify |
| `--probe_batch_rounds N` | accumulate `N` speculative rounds of committed tokens and run the probe once per `N` rounds, amortising Python / CUDA launch overhead |
| `--defer_final_flush` | the terminal probe flush at end-of-stream is dispatched after `wall_ms` is recorded; conforms to the "probe as observer must not delay user-perceived time" semantics |

Full algorithmic description: [`docs/METHOD.md`](./docs/METHOD.md).

## Quick start

```bash
# 1. Install
git clone https://github.com/hsj576/SafeSpec.git
cd SafeSpec
pip install -r requirements.txt

# 2. Point SafeSpec at your local model snapshots (or HF Hub repo ids)
cp configs/env.example.sh configs/env.sh
${EDITOR:-vi} configs/env.sh
source configs/env.sh

# 3. Download the pretrained LoRA probe
huggingface-cli download husj576/safespec-probe-qwen3guard-stream-0.6b \
    --local-dir ./probe_ckpt

# 4. Run the end-to-end evaluation (streaming mode, single GPU)
SAFESPEC_SINGLE_GPU=0 python scripts/run_e2e_spec.py \
    --dataset data/phase3_dataset.jsonl \
    --split test \
    --lora_dir ./probe_ckpt/lora \
    --async_probe \
    --probe_batch_rounds 6 \
    --defer_final_flush \
    --out outputs/my_run.json
```

For a 20-line demo (no dataset required) see
[`examples/quickstart.py`](./examples/quickstart.py).

## Repository layout

```
SafeSpec/
├── README.md              you are here (English)
├── README_CN.md           中文介绍
├── LICENSE                Apache-2.0
├── CITATION.cff           citation metadata
├── requirements.txt
├── configs/
│   └── env.example.sh     environment template (paths, GPU layout)
├── safespec/              importable Python package
│   ├── __init__.py
│   ├── common.py          config + shared utilities
│   ├── spec_decode.py     plain speculative decoder (baseline)
│   ├── spec_decode_probe.py       speculative decoder + async safety probe (main)
│   ├── probe_stream.py    Qwen3Guard-Stream wrapper + LoRA hijack head
│   ├── dataset_adapter.py AgentLAB adapter
│   └── judge.py           offline LLM-as-judge helpers
├── scripts/
│   ├── build_dataset.py       build phase-3 dataset with env-split
│   ├── train_probe_lora.py    train LoRA + hijack head (~50 s on one GPU)
│   ├── train_fuser.py         alternative: logistic / GBDT fuser
│   ├── run_e2e_spec.py        ★ main streaming eval entry point
│   ├── eval_probe_offline.py  offline probe evaluation on full splits
│   ├── benchmark_overlap.py   quantify probe/target GPU overlap
│   ├── profile_z3.py          per-step wall-time breakdown
│   ├── compare_latency.py     paired latency diff reporter
│   └── analyze_streaming.py   multi-run summariser with paired diffs
├── docs/
│   ├── METHOD.md          algorithm, architecture, loss, hyper-params
│   └── EXPERIMENT.md      full reproduction recipe + headline tables
└── examples/
    └── quickstart.py      20-line demo
```

## Public API

```python
from safespec import (
    SpecDecoder,              # plain speculative decoding (baseline)
    SpecDecoderWithProbe,     # speculative decoding + async safety probe (main)
    BatchedGuardProbe,        # Qwen3Guard-Stream wrapper (+ optional LoRA head)
    SpecDecodeStats,          # per-run timing / acceptance statistics
    ProbeStats,               # per-run probe-side statistics
    DECODE_CFG,               # default decoding hyper-parameters
)
```

See the inline docstrings and `examples/quickstart.py` for typical use.

## Pretrained probe weights

Download from HuggingFace:

- **Repo**: [husj576/safespec-probe-qwen3guard-stream-0.6b](https://huggingface.co/husj576/safespec-probe-qwen3guard-stream-0.6b)
- **Base model**: Qwen3Guard-Stream-0.6B (frozen)
- **LoRA**: r=16, alpha=32, dropout=0.05, targets=q/k/v/o_proj
- **Head**: 1024 → 1 linear + sigmoid (hijack probability)
- **Training**: 225 samples × 3 epochs, ≈ 50 s on one H800/A100

Both a merged standalone checkpoint (`model.safetensors`, ~1.2 GB) and a
LoRA-only adapter (`lora/`, ~17 MB) are provided.

## Reproducing the paper numbers

```bash
# Streaming E2E (= the ⭐ row above), ~15 min on one H800
SAFESPEC_SINGLE_GPU=0 python scripts/run_e2e_spec.py \
    --dataset data/phase3_dataset.jsonl \
    --split test \
    --lora_dir ./probe_ckpt/lora \
    --async_probe --probe_batch_rounds 6 --defer_final_flush \
    --out outputs/e2e_single_gpu_A2_Z3_br6.json

# Offline reference (should match streaming at τ=0.9 within 1 sample):
python scripts/eval_probe_offline.py \
    --lora_dir ./probe_ckpt/lora \
    --dataset data/phase3_dataset.jsonl \
    --out outputs/offline.json

# Compare two streaming runs:
python scripts/analyze_streaming.py \
    --run B1=outputs/baseline_no_probe.json \
    --run A2=outputs/e2e_single_gpu_A2_Z3_br6.json \
    --pair B1:A2 \
    --threshold 0.9
```

See [`docs/EXPERIMENT.md`](./docs/EXPERIMENT.md) for all baselines, Z3 batch
sweep, and single-GPU vs 3-GPU ablation.

## Citation

```bibtex
@software{safespec2026,
  title  = {SafeSpec: Asynchronous Safety Probing for Speculative Decoding},
  author = {SafeSpec Authors},
  year   = {2026},
  url    = {https://github.com/hsj576/SafeSpec},
  license = {Apache-2.0}
}
```

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
The base model (Qwen3Guard-Stream-0.6B, Qwen3-0.6B, Qwen3-8B) is released by
Alibaba Cloud / Qwen team under their own license; please refer to each model
card on HuggingFace Hub for terms.

## Acknowledgements

- [Qwen](https://github.com/QwenLM) team for Qwen3Guard-Stream and the Qwen3
  series.
- The AgentLAB benchmark authors for the environment-split attack data.

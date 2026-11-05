# SafeSpec Experiment Reproduction

This document is the step-by-step recipe to reproduce every number in the
project's headline table. See [`METHOD.md`](./METHOD.md) for the algorithm
description.

---

## 0. Setup

### 0.1 Hardware

Everything in this document runs on **one** H800 / H100 / A100-80G
(single-GPU, single-process, `SAFESPEC_SINGLE_GPU=0`). The 3-GPU ablation
numbers were produced on 4× H800; set the per-model device by unsetting
`SAFESPEC_SINGLE_GPU` (see `configs/env.example.sh`).

### 0.2 Environment

```bash
git clone https://github.com/hsj576/SafeSpec.git
cd SafeSpec
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 0.3 Model snapshots

Download the three base models from HuggingFace Hub:

```bash
mkdir -p ~/models && cd ~/models
huggingface-cli download Qwen/Qwen3-8B              --local-dir Qwen3-8B
huggingface-cli download Qwen/Qwen3-0.6B            --local-dir Qwen3-0.6B
huggingface-cli download Qwen/Qwen3Guard-Stream-0.6B --local-dir Qwen3Guard-Stream-0.6B
```

Then point SafeSpec at them:

```bash
cp configs/env.example.sh configs/env.sh
sed -i 's|/path/to/your/models|'${HOME}'/models|' configs/env.sh
source configs/env.sh
```

### 0.4 Pretrained probe weights

```bash
huggingface-cli download husj576/safespec-probe-qwen3guard-stream-0.6b \
    --local-dir ./probe_ckpt
```

Layout:

```
probe_ckpt/
├── README.md
├── model.safetensors          # merged base + LoRA (ready to load directly)
├── config.json, tokenizer.json, ...
├── lora/                      # LoRA-only (for stacking on your own base model)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── hijack_head.pt
└── eval/
    ├── metrics.json           # offline AUC / FNR / FPR / τ
    └── test_predictions.json  # per-sample probs
```

### 0.5 Dataset

The Phase-3 dataset (396 rows, env-split) is not shipped in the repo due to
third-party content licensing. Build it locally:

```bash
python scripts/build_dataset.py \
    --out data/phase3_dataset.jsonl
```

Expected output: `~1.3 MB` JSONL, 396 rows, with `split` ∈ {train, dev,
test} and `label` ∈ {0, 1, -1}. Rows with `label == -1` are unlabeled and
excluded by the evaluation scripts.

---

## 1. Re-train the LoRA probe (~50 s on one H800)

```bash
python scripts/train_probe_lora.py \
    --data data/phase3_dataset.jsonl \
    --out  outputs/my_probe
```

Expected final numbers (from `outputs/my_probe/training.json`):

| epoch | dev AUC | dev τ | dev FNR | dev FPR |
|---|---|---|---|---|
| 0 | 1.000 | 0.028 | 0.000 | 0.000 |
| 1 | 1.000 | 0.004 | 0.000 | 0.000 |
| 2 | 1.000 | 0.002 | 0.000 | 0.000 |

| test metric | value |
|---|---|
| AUC | 0.961 |
| τ | 0.711 |
| FNR | 0.022 (1/46) |
| FPR | 0.055 (3/55) |
| TP / FN / FP / TN | 45 / 1 / 3 / 52 |

Wall-clock: ~50 s epoch-0, ~27 s epoch-{1,2}.

---

## 2. Offline probe evaluation (sanity check)

```bash
python scripts/eval_probe_offline.py \
    --dataset data/phase3_dataset.jsonl \
    --lora_dir ./probe_ckpt/lora \
    --out outputs/offline.json
```

This re-scores every row with the LoRA probe (full prompt + output,
take last-token hidden state, apply hijack head). Expected:

| split | n | FNR | FPR | AUC |
|---|---|---|---|---|
| train | 225 | 0.000 | 0.000 | 1.000 |
| dev   | 64  | 0.000 | 0.000 | 1.000 |
| **test** | 101 | **0.022** | **0.127 @ τ=0.5** / **0.055 @ τ=0.9** | 0.961 |

---

## 3. Streaming end-to-end (the ⭐ result)

### 3.1 Single-GPU (deployment-realistic)

```bash
SAFESPEC_SINGLE_GPU=0 python scripts/run_e2e_spec.py \
    --dataset data/phase3_dataset.jsonl \
    --split test \
    --lora_dir ./probe_ckpt/lora \
    --async_probe \
    --probe_batch_rounds 6 \
    --defer_final_flush \
    --warmup 1 \
    --out outputs/single_gpu_A2_Z3br6.json
```

Also run the no-probe baseline for paired timing comparison:

```bash
SAFESPEC_SINGLE_GPU=0 python scripts/run_e2e_spec.py \
    --dataset data/phase3_dataset.jsonl \
    --split test \
    --baseline_only \
    --warmup 1 \
    --out outputs/single_gpu_B1.json
```

Takes ~15 min for each on an H800.

### 3.2 Compare (paired diff)

```bash
python scripts/analyze_streaming.py \
    --run B1=outputs/single_gpu_B1.json \
    --run A2=outputs/single_gpu_A2_Z3br6.json \
    --pair B1:A2 \
    --threshold 0.9
```

Expected:

```
Per-config summary (non-warmup)
B1                   | wall med/mean/p90 = 2355 / 3085 / ... | acc = 0.361
A2                   | wall med/mean/p90 = 2392 / 3152 / ... | acc = 0.360

Paired diffs
[B1 -> A2]  paired n = ~83
    per-sample Δwall%: median=+1.58%  mean=+2.19%  p10=-1.68%  p90=+6.xx%

Detection (last-token @ threshold = 0.9)
B1                   | FNR = N/A      | FPR = N/A        (no probe signal)
A2                   | FNR = 1/45 = 0.022 | FPR = 3/53 = 0.057
```

### 3.3 Three-GPU ablation (optional)

To reproduce the 3-GPU numbers, unset `SAFESPEC_SINGLE_GPU` so each model
goes to its own card:

```bash
unset SAFESPEC_SINGLE_GPU
python scripts/run_e2e_spec.py \
    --dataset data/phase3_dataset.jsonl --split test \
    --lora_dir ./probe_ckpt/lora \
    --async_probe --probe_batch_rounds 6 --defer_final_flush \
    --out outputs/three_gpu_A2_Z3br6.json
```

Detection on 3-GPU is identical (FNR 2.2 %, FPR 5.6 %); wall-time delta is
+27 % because the inter-GPU communication adds host-side latency that the
same-device overlap trick cannot exploit. **Single-GPU is the recommended
deployment config.**

### 3.4 Z3 batch-rounds sweep

```bash
for br in 1 2 4 6; do
  SAFESPEC_SINGLE_GPU=0 python scripts/run_e2e_spec.py \
      --dataset data/phase3_dataset.jsonl --split test \
      --lora_dir ./probe_ckpt/lora \
      --async_probe --probe_batch_rounds $br --defer_final_flush \
      --out outputs/Z3_br${br}.json
done
```

All four runs should give FNR = 2.2 %, FPR = 5.6 %; wall-time decreases
monotonically from br=1 (+27 %) to br=6 (+2.19 %).

---

## 4. Alternative: logistic / GBDT fuser (route C)

Train a tiny classifier on top of 14 hand-crafted probe + regex features:

```bash
# You need to first run run_e2e_spec.py to produce the per-sample probe
# signals, then:
python scripts/train_fuser.py \
    --data outputs/single_gpu_A2_Z3br6.json \
    --out outputs/fuser.pkl
```

Expected: FNR = 2.2 %, FPR = 7.3 %, AUC = 0.958. The `refuse_hit` regex
feature dominates (importance 0.839); removing it collapses the fuser to
FNR = 65 %, so route C is **not** a probe-side victory — it is the
judge's refusal heuristic re-encoded. We keep it in the repo as a
baseline.

---

## 5. Latency micro-benchmarks

Quantify per-component timing (prefill / draft / target / probe) and the
GPU-side overlap percentage:

```bash
python scripts/benchmark_overlap.py --samples 20 --out outputs/overlap.json
```

Per-step wall-time breakdown of a single sample with the Z3 br=6 config:

```bash
python scripts/profile_z3.py --out outputs/z3_profile.json
```

These scripts produce JSON files you can post-process with the headers
printed by `compare_latency.py`.

---

## 6. Numbers at a glance (summary)

| config | FNR @τ=0.9 | FPR @τ=0.9 | wall Δ (mean) | acceptance Δ |
|---|---|---|---|---|
| no-probe baseline (B1) | — | — | 0 (reference) | 0 |
| zero-shot Qwen3Guard-Stream | 86.7 % | 0 % | +10-30 % | 0 |
| LoRA probe, 3-GPU | 2.2 % | 5.6 % | +27 % | 0 |
| LoRA probe, 3-GPU + async | 2.2 % | 5.6 % | +11 % | 0 |
| LoRA probe, 1-GPU + Z3 br=6 | 2.2 % | 5.6 % | +3 % | −0.11 % |
| **LoRA probe, 1-GPU + Z3 br=6 + A2** ⭐ | **2.2 %** | **5.6 %** | **+2.19 %** | **−0.11 %** |

---

## 7. Known caveats

See [`METHOD.md`](./METHOD.md) Section 6:

1. mid-stream interception requires a streaming-aware head (Phase 4c);
2. the 5.6 % FPR includes a non-agent QA subset (15 % of negatives);
   BFCL-v4 evaluation (Phase 4b) expected to drop this below 5 %;
3. the +2.19 % wall overhead is already within CUDA-launch jitter;
   further reduction requires CUDA Graph capture (Phase 4a).

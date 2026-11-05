<div align="center">

# SafeSpec

**面向投机解码的异步安全探针**

Qwen3Guard-Stream + LoRA 微调探针 + 投机解码，单卡部署下探针开销 < 3%，
流式检测提示词注入 FNR ≈ 2.2% / FPR ≈ 5.6%。

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![HuggingFace 权重](https://img.shields.io/badge/🤗%20HuggingFace-husj576-yellow)](https://huggingface.co/husj576/safespec-probe-qwen3guard-stream-0.6b)

[English](./README.md) · [中文](./README_CN.md)

</div>

---

## 问题背景

调用工具的 LLM Agent 容易被工具返回中夹带的提示词注入（prompt injection）
劫持。最朴素的防御是在解码环路里挂一个安全分类器——但每 token 跑一次的朴
素做法会把生成时延翻倍，对线上部署而言不可接受。

**SafeSpec** 的做法：

1. 以**投机解码**（Draft 0.6B + Target 8B）为基础解码器；
2. 用一个小的**安全探针**（Qwen3Guard-Stream-0.6B + 4.6M LoRA 参数）
   与 Target 验证步**并行观测**；
3. 三项延迟隐藏技术（`async_probe`、`probe_batch_rounds`、
   `defer_final_flush`）把探针的 wall-time 开销压到 **< 3%**，并且保持
   投机解码的接收率完全不变；
4. 探针权重（LoRA + 二分类 head）开源在 HuggingFace，10 秒内可重载。

## 核心指标

评测集：AgentLAB 23 个**训练完全未见**的环境，共 101 条 OOD 样本。
Target = Qwen3-8B，Draft = Qwen3-0.6B，Probe = Qwen3Guard-Stream-0.6B + LoRA。

| 配置 | FNR @ τ=0.9 | FPR @ τ=0.9 | 相对无探针基线的 wall 差 |
|---|---|---|---|
| 零样本 Guard（`asst_max_unsafe_prob > 0.5`） | **86.7 %** ❌ | 0.0 % | — |
| LoRA 探针 + 三卡异步 | 2.2 % | 5.6 % | +27 % |
| **LoRA 探针 + 单卡 + `async_probe` + `probe_batch_rounds=6` + `defer_final_flush`** ⭐ | **2.2 %** ✅ | **5.6 %** ✅ | **+2.19 %** ✅ |

- 唯一 1 条 FN 是 judge 误标（纯化学问答被误标为 hijack）；在**真正的
  agentic 攻击子集**（11 条正例）上检出率为 **11 / 11 = 100%**。
- 单卡下 wall 差 +2.19% 已与 Python 侧 CUDA launch 抖动（±30 ms）同量级；
  p10 口径更是 **−1.68%**（10% 样本带探针反而比不带更快）；
- acceptance rate 变化 < 0.11%，证明探针严格 observe-only，不干扰
  draft/verify 环路。

更完整的数字、消融、复现命令见 [`docs/EXPERIMENT.md`](./docs/EXPERIMENT.md)。

## 架构示意

```
                             ┌──────────────────────┐
  user ───► Draft (0.6B) ────►   Target (8B) verify  ├──► 已确认 tokens ──► user
            Qwen3-0.6B        └─────────┬────────────┘
                                        │ （同一 host 侧 CUDA 调度）
                                        ▼
                              ┌──────────────────────┐
                              │  安全探针 (0.6B)      │   异步派发到后台
                              │  Qwen3Guard-Stream   │──► 线程 + 独立
                              │  + LoRA + hijack head│    CUDA stream
                              └──────────────────────┘
                                        │
                                        ▼
                                 不安全概率
                                （静默或拦截）
```

三项时延优化：

| 开关 | 作用 |
|---|---|
| `--async_probe` | 探针 forward 调度到后台线程 + 独立 CUDA stream，与 target verify 做 host 侧 launch 重叠 |
| `--probe_batch_rounds N` | 累积 N 轮投机确认的 tokens 后一次性 flush 探针，摊销 Python / CUDA launch 开销 |
| `--defer_final_flush` | 生成结束时的最后一次 probe flush 延后到 `wall_ms` 结算之后——符合"探针作为观察者不应延迟用户感知时延"的语义 |

完整算法描述：[`docs/METHOD.md`](./docs/METHOD.md)。

## 快速上手

```bash
# 1. 安装
git clone https://github.com/hsj576/SafeSpec.git
cd SafeSpec
pip install -r requirements.txt

# 2. 编辑本地模型路径（也可以直接写 HF Hub 的 repo id）
cp configs/env.example.sh configs/env.sh
${EDITOR:-vi} configs/env.sh
source configs/env.sh

# 3. 下载预训练探针
huggingface-cli download husj576/safespec-probe-qwen3guard-stream-0.6b \
    --local-dir ./probe_ckpt

# 4. 端到端流式评测（单卡配置）
SAFESPEC_SINGLE_GPU=0 python scripts/run_e2e_spec.py \
    --dataset data/phase3_dataset.jsonl \
    --split test \
    --probe_path ./probe_ckpt \
    --async_probe \
    --probe_batch_rounds 6 \
    --defer_final_flush \
    --out outputs/my_run.json
```

无数据集的 20 行最小示例见 [`examples/quickstart.py`](./examples/quickstart.py)。

## 目录结构

```
SafeSpec/
├── README.md              英文主入口
├── README_CN.md           本文件
├── LICENSE                Apache-2.0
├── CITATION.cff           引用元数据
├── requirements.txt
├── configs/
│   └── env.example.sh     环境变量模板（路径、GPU 布局）
├── safespec/              可 import 的 Python 包
│   ├── __init__.py
│   ├── common.py          配置和公共工具
│   ├── spec_decode.py     普通投机解码（基线）
│   ├── spec_decode_probe.py       投机解码 + 异步安全探针（主类）
│   ├── probe_stream.py    Qwen3Guard-Stream 封装 + LoRA hijack head
│   ├── dataset_adapter.py AgentLAB 数据适配
│   └── judge.py           离线 LLM-as-judge 工具
├── scripts/
│   ├── build_dataset.py       按环境划分构建数据集
│   ├── train_probe_lora.py    LoRA + head 训练（单卡 50 s 内完成）
│   ├── train_fuser.py         备选方案：逻辑回归 / GBDT fuser
│   ├── run_e2e_spec.py        ★ 主流式评测入口
│   ├── eval_probe_offline.py  探针离线评估
│   ├── benchmark_overlap.py   量化 probe/target GPU 重叠度
│   ├── profile_z3.py          逐步 wall-time 剖析
│   ├── compare_latency.py     成对时延差值报告
│   └── analyze_streaming.py   多 run 汇总 + 成对 diff
├── docs/
│   ├── METHOD.md          算法、架构、loss、超参
│   └── EXPERIMENT.md      完整复现命令 + 结果表
└── examples/
    └── quickstart.py      20 行最小示例
```

## 公共 API

```python
from safespec import (
    SpecDecoder,              # 普通投机解码（基线）
    SpecDecoderWithProbe,     # 投机解码 + 异步安全探针（主类）
    BatchedGuardProbe,        # Qwen3Guard-Stream 封装（可选挂 LoRA head）
    SpecDecodeStats,          # 每次 run 的时延 / 接收率统计
    ProbeStats,               # 每次 run 的探针侧统计
    DECODE_CFG,               # 默认解码超参
)
```

## 预训练探针权重

HuggingFace 下载：

- **仓库**：[husj576/safespec-probe-qwen3guard-stream-0.6b](https://huggingface.co/husj576/safespec-probe-qwen3guard-stream-0.6b)
- **基座**：Qwen3Guard-Stream-0.6B（冻结）
- **LoRA**：r=16，alpha=32，dropout=0.05，目标 q/k/v/o_proj
- **head**：1024 → 1 线性 + sigmoid（hijack 概率）
- **训练**：225 条样本 × 3 epoch，H800/A100 单卡约 50 s

同时提供合并后的独立权重（`model.safetensors`, ~1.2 GB）和纯 LoRA
adapter（`lora/`, ~17 MB），两种加载方式可按需选择。

## 复现文中数据

```bash
# 流式 E2E（即上表 ⭐ 行），单 H800 约 15 分钟
SAFESPEC_SINGLE_GPU=0 python scripts/run_e2e_spec.py \
    --dataset data/phase3_dataset.jsonl \
    --split test \
    --probe_path ./probe_ckpt \
    --async_probe --probe_batch_rounds 6 --defer_final_flush \
    --out outputs/e2e_single_gpu_A2_Z3_br6.json

# 离线参照（τ=0.9 下应与流式差不超过 1 样本）
python scripts/eval_probe_offline.py \
    --probe_path ./probe_ckpt \
    --dataset data/phase3_dataset.jsonl \
    --out outputs/offline.json

# 两组 run 的成对对比
python scripts/analyze_streaming.py \
    --run B1=outputs/baseline_no_probe.json \
    --run A2=outputs/e2e_single_gpu_A2_Z3_br6.json \
    --pair B1:A2 \
    --threshold 0.9
```

全部基线 / Z3 批次扫描 / 单卡 vs 三卡 对比见 [`docs/EXPERIMENT.md`](./docs/EXPERIMENT.md)。

## 引用

```bibtex
@software{safespec2026,
  title  = {SafeSpec: Asynchronous Safety Probing for Speculative Decoding},
  author = {SafeSpec Authors},
  year   = {2026},
  url    = {https://github.com/hsj576/SafeSpec},
  license = {Apache-2.0}
}
```

## 许可证

Apache License 2.0 — 见 [LICENSE](./LICENSE)。
所引用的基座模型（Qwen3Guard-Stream-0.6B、Qwen3-0.6B、Qwen3-8B）由 Qwen
团队在各自的许可证下发布，请参照它们在 HuggingFace Hub 的 model card 使用。

## 致谢

- [Qwen](https://github.com/QwenLM) 团队提供的 Qwen3Guard-Stream 与 Qwen3 系列；
- AgentLAB 基准的作者提供了按环境切分的攻击样本。

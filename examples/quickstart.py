#!/usr/bin/env python3
"""SafeSpec quick-start demo.

Generates a short reply with speculative decoding + the LoRA safety probe
mounted, and prints both the generated text and the probe's hijack
probability.

Usage
-----

1.  Set up models and env vars (see configs/env.example.sh):

        source configs/env.sh

2.  Download the pretrained probe:

        huggingface-cli download husj576/safespec-probe-qwen3guard-stream-0.6b \\
            --local-dir ./probe_ckpt

3.  Run:

        python examples/quickstart.py \\
            --prompt "Summarise the last email from IT Support." \\
            --lora_dir ./probe_ckpt/lora
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running from a fresh checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from safespec import (BatchedGuardProbe, ProbeStats, SpecDecodeStats,
                      SpecDecoderWithProbe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=(
        "Explain in at least 200 words how transformer self-attention works. "
        "Cover Q/K/V projections, the scaled dot-product, the softmax and "
        "multi-head attention."),
        help="user-side prompt to feed the agent")
    ap.add_argument("--lora_dir", default=None,
                    help="path to the LoRA adapter + hijack_head.pt "
                         "(omit to run with the zero-shot Qwen3Guard-Stream)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--k", type=int, default=5,
                    help="speculative draft length")
    args = ap.parse_args()

    # 1.  Bring up the safety probe (optionally with LoRA + hijack head).
    probe = BatchedGuardProbe(lora_dir=args.lora_dir)

    # 2.  Bring up the speculative decoder with the probe injected and the
    #     three latency-hiding tricks enabled.
    engine = SpecDecoderWithProbe(
        probe=probe,
        async_probe=True,
        probe_batch_rounds=6,
        defer_final_flush=True,
    )

    # 3.  Tokenize the chat prompt using the target model's tokenizer.
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant.  You may use tools to answer."},
        {"role": "user", "content": args.prompt},
    ]
    # `apply_chat()` returns a 1-D LongTensor on CPU; generate_with_probe
    # moves it to the target/draft devices internally.
    prompt_ids = engine.apply_chat(messages)

    # 4.  Run speculative decoding with the probe side-car.
    stats = SpecDecodeStats()
    probe_stats = ProbeStats()
    output_ids = engine.generate_with_probe(
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        draft_k=args.k,
        stats=stats,
        probe_stats=probe_stats,
    )
    text = engine.tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)

    # 5.  Report.
    print("=" * 60)
    print("Generated text")
    print("=" * 60)
    print(text)
    print()
    print("=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"wall_ms              = {stats.wall_ms:8.1f}")
    print(f"tokens_per_sec       = {stats.tokens_per_sec:8.2f}")
    print(f"acceptance_rate      = {stats.acceptance_rate:8.3f}")
    print(f"probe_total_ms       = {probe_stats.probe_total_ms:8.1f}")
    print(f"probe_postwall_ms    = {probe_stats.probe_postwall_ms:8.1f}")
    last = probe_stats.asst_last_lora_hijack_prob
    print(f"hijack probability   = {last:8.3f}   "
          f"({'UNSAFE' if last > 0.9 else 'safe'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

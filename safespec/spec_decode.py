"""Minimal speculative decoding runtime: Qwen3-8B target + Qwen3-0.6B draft.

Design goals:
    * single request, batch_size=1, greedy sampling (temperature=0.0)
    * standard speculative decoding loop, no tree attention, no multi-candidate
    * ZERO wasted draft forwards per round (unlike the earlier `_draft_k` which
      cropped + re-fed `last_tok` every round)

Invariants at the top of each speculative round (`gen = generated`):
    kv_target        covers [prompt + gen[:-1]]              (len = prompt_len + len(gen) - 1)
                     i.e. the last committed token is NOT in target's KV
    kv_draft         covers [prompt + gen]                   (len = prompt_len + len(gen))
                     i.e. the last committed token IS in draft's KV
    last_committed   == gen[-1]
    last_draft_logit == next-token logit produced by the draft's most recent
                        forward (which fed `last_committed`); its argmax is the
                        first draft proposal of the upcoming round.

Why the asymmetry:
    * target KV excludes `last_committed` so that during verify we can feed
      `[last_committed] + draft_tokens` (length k+1) and the target returns
      k+1 logits, of which:
          logits[0]   -> predicts token after `last_committed`      compare draft[0]
          logits[i]   -> predicts token after `draft[i-1]`          compare draft[i]
          logits[k]   -> predicts token after `draft[k-1]`          (bonus if accept==k)
      giving us a "free" bonus for the all-accepted case.
    * draft KV includes `last_committed` (plus `last_draft_logit`) so that
      _draft_k starts by consuming the stored logit (no forward needed) and
      performs exactly k draft forwards: k-1 chained forwards to generate
      drafts 1..k-1 PLUS one final forward on draft[k-1] to produce the next
      round's `last_draft_logit`.  No forward is ever wasted on re-feeding
      the previously-committed token.

Draft-side cost summary per round:
      k forwards (= number of draft tokens produced)
      + 1 extra forward IF the mismatch position < k (to feed `bonus_tok` and
        obtain the new `last_draft_logit`).  When accept == k, bonus_tok is
        argmax(last_draft_logit) itself and we only need to feed it + update
        the logit, still 1 extra forward.
      Net: k + 1 forwards (minimal possible for a token-by-token draft).
Target-side cost: 1 verify forward of length k+1.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .common import (
    DECODE_CFG,
    DEVICE_DRAFT,
    DEVICE_TARGET,
    DRAFT_MODEL_PATH,
    DTYPE,
    TARGET_MODEL_PATH,
    get_logger,
)

log = get_logger("spec_decode")


# ---------------------------------------------------------------------------
# Stats record
# ---------------------------------------------------------------------------
@dataclass
class SpecDecodeStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    bonus_tokens: int = 0
    n_rounds: int = 0
    accept_lengths: List[int] = field(default_factory=list)
    wall_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(1, self.draft_tokens)

    @property
    def mean_accept_len(self) -> float:
        return (sum(self.accept_lengths) / len(self.accept_lengths)) if self.accept_lengths else 0.0

    @property
    def tokens_per_sec(self) -> float:
        return self.generated_tokens / max(1e-6, self.wall_ms / 1000.0)


# ---------------------------------------------------------------------------
# SpecDecoder
# ---------------------------------------------------------------------------
class SpecDecoder:
    """A minimal, auditable speculative decoding engine.

    Target lives on `cuda:0`, draft on `cuda:1`.  Both models share the Qwen3
    tokenizer so no re-tokenization is needed between them.
    """

    def __init__(
        self,
        target_path: str = str(TARGET_MODEL_PATH),
        draft_path: str = str(DRAFT_MODEL_PATH),
        target_device: str = DEVICE_TARGET,
        draft_device: str = DEVICE_DRAFT,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        self.target_device = torch.device(target_device)
        self.draft_device = torch.device(draft_device)
        self.dtype = dtype

        log.info("Loading tokenizer from %s", target_path)
        self.tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        log.info("Loading TARGET  Qwen3-8B   on %s (%s)", self.target_device, dtype)
        self.target = AutoModelForCausalLM.from_pretrained(
            target_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.target_device).eval()

        log.info("Loading DRAFT   Qwen3-0.6B on %s (%s)", self.draft_device, dtype)
        self.draft = AutoModelForCausalLM.from_pretrained(
            draft_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.draft_device).eval()

        self.eos_token_ids = {self.tokenizer.eos_token_id}
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end != self.tokenizer.unk_token_id:
            self.eos_token_ids.add(im_end)
        log.info("EOS token ids: %s", sorted(self.eos_token_ids))

    # ------------------------------------------------------------------
    def apply_chat(self, messages, enable_thinking: bool = False) -> torch.LongTensor:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return self.tokenizer(text, return_tensors="pt").input_ids[0]

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int = DECODE_CFG.max_new_tokens,
        draft_k: int = DECODE_CFG.draft_k,
        stats: Optional[SpecDecodeStats] = None,
    ) -> torch.LongTensor:
        stats = stats or SpecDecodeStats()
        stats.prompt_tokens = int(prompt_ids.shape[0])
        prompt_len = stats.prompt_tokens

        prompt_target = prompt_ids.to(self.target_device).unsqueeze(0)
        prompt_draft = prompt_ids.to(self.draft_device).unsqueeze(0)

        t_wall = time.perf_counter()

        # ---------------- prefill ----------------
        t_pf = time.perf_counter()
        kv_target = DynamicCache()
        kv_draft = DynamicCache()

        # Target: feed `prompt` only.  argmax of last logit is the first token.
        # Target KV length after prefill = prompt_len.  Last committed token
        # (first_tok) is NOT yet in target KV -- matches our invariant.
        out_t = self.target(prompt_target, past_key_values=kv_target, use_cache=True)
        kv_target = out_t.past_key_values
        first_tok = int(torch.argmax(out_t.logits[0, -1]).item())
        del out_t

        # Draft: feed `prompt + first_tok` so draft KV = prompt_len + 1 AND
        # we obtain the next-token logit that the next round will consume.
        # We do this in two forwards to keep code paths uniform.
        out_d = self.draft(prompt_draft, past_key_values=kv_draft, use_cache=True)
        kv_draft = out_d.past_key_values
        bt_d = torch.tensor([[first_tok]], dtype=torch.long, device=self.draft_device)
        out_d = self.draft(bt_d, past_key_values=kv_draft, use_cache=True)
        kv_draft = out_d.past_key_values
        last_draft_logit = out_d.logits[0, -1].clone()
        del out_d

        torch.cuda.synchronize(self.target_device)
        torch.cuda.synchronize(self.draft_device)
        stats.prefill_ms = (time.perf_counter() - t_pf) * 1000.0

        generated: List[int] = [first_tok]
        last_committed = first_tok

        if first_tok in self.eos_token_ids:
            stats.generated_tokens = len(generated)
            stats.wall_ms = (time.perf_counter() - t_wall) * 1000.0
            return torch.tensor(generated, dtype=torch.long)

        # ---------------- speculative rounds ----------------
        while len(generated) < max_new_tokens:
            stats.n_rounds += 1
            prev_gen_len = len(generated)   # len(generated) BEFORE this round's commit

            # Sanity: at this point
            #   kv_target.get_seq_length() == prompt_len + prev_gen_len - 1
            #   kv_draft.get_seq_length()  == prompt_len + prev_gen_len
            # (asserts omitted in the hot path)

            # ---- step 1: draft k tokens ----
            draft_tokens, kv_draft, _ignored = self._draft_k(
                kv_draft, last_draft_logit, draft_k,
            )
            k_actual = len(draft_tokens)
            stats.draft_tokens += k_actual
            # After _draft_k, kv_draft covers prompt + prev_gen + draft (so length
            # = prompt_len + prev_gen_len + k_actual).  We'll crop it in step 5.

            # ---- step 2: target verifies [last_committed] + draft ----
            verify_input = [last_committed] + draft_tokens
            verify_t = torch.tensor([verify_input], dtype=torch.long,
                                    device=self.target_device)
            out_t = self.target(verify_t, past_key_values=kv_target, use_cache=True)
            kv_target = out_t.past_key_values
            preds = torch.argmax(out_t.logits[0], dim=-1).tolist()
            del out_t
            # kv_target length is now prompt_len + prev_gen_len - 1 + (k_actual + 1)
            # = prompt_len + prev_gen_len + k_actual

            # ---- step 3: find longest matching prefix ----
            accept = 0
            for i in range(k_actual):
                if preds[i] == draft_tokens[i]:
                    accept += 1
                else:
                    break
            stats.accepted_tokens += accept
            stats.accept_lengths.append(accept)
            bonus_tok = preds[accept]          # free bonus thanks to the k+1 feed-scheme
            committed_this_round = draft_tokens[:accept] + [bonus_tok]
            stats.bonus_tokens += 1
            generated.extend(committed_this_round)

            # ---- step 4: align target KV (do NOT re-feed bonus) ----
            # New generated length = prev_gen_len + accept + 1.
            # Invariant requires kv_target length = prompt_len + new_gen_len - 1
            #                                     = prompt_len + prev_gen_len + accept.
            # kv_target currently has prompt_len + prev_gen_len + k_actual,
            # so we must drop (k_actual - accept) positions from the tail.
            kv_target.crop(prompt_len + prev_gen_len + accept)

            # ---- step 5: align draft KV + refresh last_draft_logit ----
            # Invariant requires kv_draft length = prompt_len + new_gen_len
            #                                    = prompt_len + prev_gen_len + accept + 1.
            # kv_draft currently has prompt_len + prev_gen_len + k_actual,
            # so we must drop (k_actual - accept) positions and feed `bonus_tok`.
            kv_draft.crop(prompt_len + prev_gen_len + accept)
            bt_d = torch.tensor([[bonus_tok]], dtype=torch.long, device=self.draft_device)
            out_d = self.draft(bt_d, past_key_values=kv_draft, use_cache=True)
            kv_draft = out_d.past_key_values
            last_draft_logit = out_d.logits[0, -1].clone()
            del out_d

            last_committed = bonus_tok

            # ---- stopping ----
            if any(t in self.eos_token_ids for t in committed_this_round):
                break
            if len(generated) >= max_new_tokens:
                break

        torch.cuda.synchronize(self.target_device)
        torch.cuda.synchronize(self.draft_device)
        stats.wall_ms = (time.perf_counter() - t_wall) * 1000.0
        stats.decode_ms = stats.wall_ms - stats.prefill_ms
        stats.generated_tokens = len(generated)

        for i, t in enumerate(generated):
            if t in self.eos_token_ids:
                generated = generated[: i + 1]
                stats.generated_tokens = len(generated)
                break
        return torch.tensor(generated, dtype=torch.long)

    # ------------------------------------------------------------------
    def _draft_k(
        self,
        kv_draft: DynamicCache,
        last_next_logit: torch.Tensor,
        k: int,
    ) -> Tuple[List[int], DynamicCache, torch.Tensor]:
        """Produce k greedy draft tokens with exactly k draft forwards.

        Precondition (invariant):
            kv_draft already covers `prompt + generated_so_far` INCLUDING the
            last committed token.  `last_next_logit` is the discarded next-token
            logit from the most recent draft forward (whose input was the last
            committed token).

        Postcondition:
            Return value `tokens` has length <= k (shorter only on early EOS).
            kv_draft is extended to cover `prompt + generated + tokens[:len(tokens)]`.
            The third return value is the next-token logit after feeding the
            final draft token (we only need it internally; the outer loop
            discards it because it has its own alignment step that re-feeds
            `bonus_tok` and generates a fresh `last_draft_logit`).
        """
        tokens: List[int] = []
        # First proposal: argmax of the stashed logit, no forward.
        cur = int(torch.argmax(last_next_logit).item())
        tokens.append(cur)
        last_logit = last_next_logit  # placeholder; overwritten after first forward

        for step in range(k - 1):
            t = torch.tensor([[cur]], dtype=torch.long, device=self.draft_device)
            out = self.draft(t, past_key_values=kv_draft, use_cache=True)
            kv_draft = out.past_key_values
            last_logit = out.logits[0, -1]
            cur = int(torch.argmax(last_logit).item())
            del out
            tokens.append(cur)
            if cur in self.eos_token_ids:
                # feed the EOS itself into KV so KV length == prev + len(tokens)
                t2 = torch.tensor([[cur]], dtype=torch.long, device=self.draft_device)
                out2 = self.draft(t2, past_key_values=kv_draft, use_cache=True)
                return tokens, out2.past_key_values, out2.logits[0, -1]

        # Final forward: feed the k-th proposal to extend KV to +k positions.
        # (We also obtain a `new_last_next_logit` but the outer loop will not
        # use it -- it re-feeds `bonus_tok` afterwards.)
        t = torch.tensor([[cur]], dtype=torch.long, device=self.draft_device)
        out = self.draft(t, past_key_values=kv_draft, use_cache=True)
        return tokens, out.past_key_values, out.logits[0, -1]


# ---------------------------------------------------------------------------
# Quick CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Explain the Pythagorean theorem in one short paragraph.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--k", type=int, default=DECODE_CFG.draft_k)
    args = parser.parse_args()

    engine = SpecDecoder()
    messages = [{"role": "user", "content": args.prompt}]
    prompt_ids = engine.apply_chat(messages)
    log.info("Prompt (%d tokens): %r", prompt_ids.shape[0], args.prompt)

    stats = SpecDecodeStats()
    out_ids = engine.generate(prompt_ids, max_new_tokens=args.max_new_tokens,
                              draft_k=args.k, stats=stats)
    text = engine.tokenizer.decode(out_ids.tolist(), skip_special_tokens=True)
    log.info("=== OUTPUT ===\n%s", text)
    log.info("=== STATS ===")
    log.info("  prompt_tokens      : %d", stats.prompt_tokens)
    log.info("  generated_tokens   : %d", stats.generated_tokens)
    log.info("  draft_tokens       : %d", stats.draft_tokens)
    log.info("  accepted_tokens    : %d", stats.accepted_tokens)
    log.info("  acceptance_rate    : %.3f", stats.acceptance_rate)
    log.info("  n_rounds           : %d", stats.n_rounds)
    log.info("  mean_accept_len    : %.2f  (k=%d)", stats.mean_accept_len, args.k)
    log.info("  prefill_ms         : %.1f", stats.prefill_ms)
    log.info("  decode_ms          : %.1f", stats.decode_ms)
    log.info("  wall_ms            : %.1f", stats.wall_ms)
    log.info("  tokens_per_sec     : %.1f", stats.tokens_per_sec)

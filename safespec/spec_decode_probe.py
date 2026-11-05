"""E1 experimental group: speculative decoding with an async safety probe
running in SILENT mode (observes only, never interrupts generation).

The probe (Qwen3Guard-Stream-0.6B) is called once per speculative round, after
the draft has produced its k tokens.  Each round:

    1. draft produces k tokens  (GPU 1)
    2. target verifies k+1 tokens -> preds              (GPU 0)
    3. probe batch-scores the k draft tokens -> risk    (GPU 2)
       (steps 2 and 3 are launched back-to-back from the same Python thread;
        with the current HF-based implementation the GPUs effectively serialize
        due to Python/GIL host overhead -- known limitation, documented.)
    4. accept / commit as in B1
    5. record probe output for the newly committed tokens (NOT for the rejected
       drafts, so the probe history matches the final output text exactly)

We ALSO prefill the probe with the entire chat prompt before decoding starts.
This feeds the user turn + any tool observations into the probe so it can flag
input-side concerns, and -- more importantly -- it builds the probe's KV cache
so that subsequent per-round scoring is cheap (incremental).
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers.cache_utils import DynamicCache

from .common import DECODE_CFG, get_logger
from .probe_stream import BatchedGuardProbe
from .spec_decode import SpecDecodeStats, SpecDecoder

log = get_logger("spec_decode_probe")


# ---------------------------------------------------------------------------
# Extended stats
# ---------------------------------------------------------------------------
@dataclass
class ProbeStats:
    user_turn_max_unsafe_prob: float = 0.0
    user_turn_max_category: Optional[str] = None
    asst_rounds: int = 0
    asst_max_unsafe_prob: float = 0.0
    asst_max_category: Optional[str] = None
    asst_first_alarm_token_idx: int = -1    # position in `generated` where probe first said "Unsafe"
    probe_total_ms: float = 0.0
    # Phase 4a (Z3): how many spec rounds between consecutive probe forwards.
    # 1 == legacy behaviour (score every round).  N>1 accumulates N*(accept+1)
    # committed tokens and scores them in a single forward, amortising the
    # host-side kernel-launch overhead.  Recorded for post-hoc ablation.
    probe_batch_rounds: int = 1
    probe_calls: int = 0
    # Phase 4b (A2): wall-time spent joining probe futures AFTER stats.wall_ms
    # was already finalised.  When defer_final_flush=True, the final probe
    # flush is dispatched asynchronously and we settle `wall_ms` immediately
    # (matches the guideline semantics that the probe is an observer and
    # must not delay user-perceived generation time).  This counter records
    # the deferred join cost for transparency.
    probe_postwall_ms: float = 0.0
    # LoRA (Phase 3 route B) running maxima and per-token traces.
    # IMPORTANT: the LoRA hijack head was TRAINED on the LAST-token hidden state
    # (see probe_lora_train.py: `encode_row` appends the full assistant turn and
    # we take hidden state[-1]).  During streaming we see partial outputs, and
    # mid-sequence probs can be noisy; the authoritative aggregator is the
    # LAST-token prob, exposed as `asst_last_lora_hijack_prob`.  We still keep
    # the running max for diagnostic purposes.
    asst_max_lora_hijack_prob: float = 0.0
    asst_last_lora_hijack_prob: float = 0.0
    user_turn_max_lora_hijack_prob: float = 0.0
    # per-committed-token probe output; aligned to `generated` tensor
    per_token: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SpecDecoderWithProbe
# ---------------------------------------------------------------------------
class SpecDecoderWithProbe(SpecDecoder):
    """B1 engine extended with an async safety probe (silent mode).

    Phase 4a knob: ``async_probe`` -- if True, submit the probe forward to a
    background thread so that its kernel launches on cuda:2 happen concurrently
    with the target verify kernels on cuda:0.  On CPython the overlap is real
    because both forwards release the GIL inside kernel-launch C code; the
    main thread stays on target.verify, and the pool thread drives probe.
    """

    def __init__(
        self,
        probe: Optional[BatchedGuardProbe] = None,
        async_probe: bool = False,
        probe_batch_rounds: int = 1,
        defer_final_flush: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.probe = probe if probe is not None else BatchedGuardProbe()
        # risk-level string that signals an alarm (both Unsafe and Controversial)
        self.alarm_levels = {"Unsafe", "Controversial"}
        self.async_probe = bool(async_probe)
        self.probe_batch_rounds = max(1, int(probe_batch_rounds))
        # Phase 4b (A2): defer the final probe flush past wall_ms settlement.
        # Only effective when both async_probe=True AND probe_batch_rounds>1
        # (Z3 batched path).  With this flag, the user-perceived wall time no
        # longer includes the trailing probe forward on the tail buffer.
        self.defer_final_flush = bool(defer_final_flush)
        # one worker is enough: probe forward is strictly serial w.r.t. itself
        self._probe_pool: Optional[ThreadPoolExecutor] = None
        if self.async_probe:
            self._probe_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="probe"
            )
            log.info("async_probe ENABLED: probe forward will run on a "
                     "background thread for host-side kernel-launch overlap")
        if self.probe_batch_rounds > 1:
            log.info("probe_batch_rounds=%d: probe scores committed tokens "
                     "every %d spec rounds (Z3 batched path)",
                     self.probe_batch_rounds, self.probe_batch_rounds)
        if self.defer_final_flush:
            if not (self.async_probe and self.probe_batch_rounds > 1):
                log.warning("defer_final_flush requires async_probe=True AND "
                            "probe_batch_rounds>1; falling back to sync final flush")
                self.defer_final_flush = False
            else:
                log.info("defer_final_flush ENABLED: final probe flush will "
                         "run AFTER stats.wall_ms is settled (A2 optimisation)")

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_with_probe(
        self,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int = DECODE_CFG.max_new_tokens,
        draft_k: int = DECODE_CFG.draft_k,
        stats: Optional[SpecDecodeStats] = None,
        probe_stats: Optional[ProbeStats] = None,
        alarm_threshold: float = 0.5,
    ) -> torch.LongTensor:
        """Run B1 speculative decoding AND the silent probe side-car.

        Returns the generated-token tensor (same semantics as `SpecDecoder.generate`).
        `probe_stats` is populated in place with per-round + aggregated signals.
        """
        stats = stats or SpecDecodeStats()
        probe_stats = probe_stats or ProbeStats()
        probe_stats.probe_batch_rounds = self.probe_batch_rounds
        stats.prompt_tokens = int(prompt_ids.shape[0])
        prompt_len = stats.prompt_tokens
        # Fast-path dispatch: when probe_batch_rounds > 1, use the Z3 batched
        # path which only scores *committed* tokens, every N rounds, in a
        # single forward.  Keeps the legacy per-round path untouched below.
        if self.probe_batch_rounds > 1:
            return self._generate_with_probe_batched(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                draft_k=draft_k,
                stats=stats,
                probe_stats=probe_stats,
                alarm_threshold=alarm_threshold,
            )

        # ---------------- probe prefill ----------------
        # Reset the probe + feed the entire chat prompt as the "user turn"
        # (chat template already includes the system + user + tool observations,
        # all of which are prompt-side content from the attacker's viewpoint).
        t_probe_pf = time.perf_counter()
        self.probe.reset()
        user_result = self.probe.score_user_turn(prompt_ids)
        probe_stats.user_turn_max_unsafe_prob = float(self.probe.max_unsafe_prob)
        probe_stats.user_turn_max_category = self.probe.max_unsafe_category
        if self.probe.has_lora:
            probe_stats.user_turn_max_lora_hijack_prob = float(self.probe.max_lora_hijack_prob)
        probe_pf_ms = (time.perf_counter() - t_probe_pf) * 1000.0

        prompt_target = prompt_ids.to(self.target_device).unsqueeze(0)
        prompt_draft = prompt_ids.to(self.draft_device).unsqueeze(0)

        t_wall = time.perf_counter()

        # ---------------- target + draft prefill ----------------
        t_pf = time.perf_counter()
        kv_target = DynamicCache()
        kv_draft = DynamicCache()

        out_t = self.target(prompt_target, past_key_values=kv_target, use_cache=True)
        kv_target = out_t.past_key_values
        first_tok = int(torch.argmax(out_t.logits[0, -1]).item())
        del out_t

        out_d = self.draft(prompt_draft, past_key_values=kv_draft, use_cache=True)
        kv_draft = out_d.past_key_values
        bt_d = torch.tensor([[first_tok]], dtype=torch.long, device=self.draft_device)
        out_d = self.draft(bt_d, past_key_values=kv_draft, use_cache=True)
        kv_draft = out_d.past_key_values
        last_draft_logit = out_d.logits[0, -1].clone()
        del out_d

        torch.cuda.synchronize(self.target_device)
        torch.cuda.synchronize(self.draft_device)
        stats.prefill_ms = (time.perf_counter() - t_pf) * 1000.0 + probe_pf_ms

        generated: List[int] = [first_tok]
        last_committed = first_tok

        # The very first committed token (first_tok) was produced by the target
        # during prefill -- feed it to the probe so per-token history is complete.
        t_pr = time.perf_counter()
        first_tok_result = self.probe.score_assistant_tokens(
            torch.tensor([first_tok], dtype=torch.long)
        )
        probe_stats.probe_total_ms += (time.perf_counter() - t_pr) * 1000.0
        pt_entry = {
            "token_id": first_tok,
            "risk_level": first_tok_result["risk_level"][0],
            "category": first_tok_result["category"][0],
            "unsafe_prob": first_tok_result["unsafe_prob"][0],
        }
        if self.probe.has_lora:
            pt_entry["lora_hijack_prob"] = first_tok_result["lora_hijack_prob"][0]
        probe_stats.per_token.append(pt_entry)
        if (first_tok_result["risk_level"][0] in self.alarm_levels
                and probe_stats.asst_first_alarm_token_idx < 0):
            probe_stats.asst_first_alarm_token_idx = 0
        if self.probe.has_lora:
            probe_stats.asst_max_lora_hijack_prob = float(self.probe.max_lora_hijack_prob)

        if first_tok in self.eos_token_ids:
            self._finalize(generated, stats, probe_stats, t_wall)
            return torch.tensor(generated, dtype=torch.long)

        # ---------------- speculative rounds ----------------
        while len(generated) < max_new_tokens:
            stats.n_rounds += 1
            probe_stats.asst_rounds += 1
            prev_gen_len = len(generated)

            # --- step 1: draft k tokens
            draft_tokens, kv_draft, _ = self._draft_k(
                kv_draft, last_draft_logit, draft_k,
            )
            k_actual = len(draft_tokens)
            stats.draft_tokens += k_actual

            # --- step 2 + 3: target verify (GPU0) and probe scoring (GPU2)
            # Launched back-to-back; see docstring for the overlap caveat.
            verify_input = [last_committed] + draft_tokens
            verify_t = torch.tensor([verify_input], dtype=torch.long,
                                    device=self.target_device)
            draft_t_cpu = torch.tensor(draft_tokens, dtype=torch.long)
            t_pr = time.perf_counter()
            if self.async_probe and self._probe_pool is not None:
                # Dispatch probe forward on background thread FIRST so its
                # kernel launches start piling up on cuda:2 while the main
                # thread then launches target verify on cuda:0.  Both forwards
                # release the GIL inside the C-level CUDA kernel launcher so
                # they truly interleave on the host side.
                probe_future: Future = self._probe_pool.submit(
                    self.probe.score_assistant_tokens, draft_t_cpu,
                )
                out_t = self.target(verify_t, past_key_values=kv_target, use_cache=True)
                probe_res = probe_future.result()
            else:
                # Legacy path: launch target first, then probe on main thread.
                out_t = self.target(verify_t, past_key_values=kv_target, use_cache=True)
                probe_res = self.probe.score_assistant_tokens(draft_t_cpu)
            # Now consume target output
            kv_target = out_t.past_key_values
            preds = torch.argmax(out_t.logits[0], dim=-1).tolist()
            del out_t
            probe_stats.probe_total_ms += (time.perf_counter() - t_pr) * 1000.0

            # --- step 4: accept / bonus
            accept = 0
            for i in range(k_actual):
                if preds[i] == draft_tokens[i]:
                    accept += 1
                else:
                    break
            stats.accepted_tokens += accept
            stats.accept_lengths.append(accept)
            bonus_tok = preds[accept]
            committed_this_round = draft_tokens[:accept] + [bonus_tok]
            stats.bonus_tokens += 1

            # --- step 5: record probe output for the accepted draft tokens.
            # Note: `probe_res` currently covers ALL k draft tokens, including
            # the rejected tail.  We keep only the accepted prefix in per_token.
            # The bonus token was produced by the target, NOT by the draft -- we
            # need a separate probe forward for it.
            for i in range(accept):
                entry = {
                    "token_id": draft_tokens[i],
                    "risk_level": probe_res["risk_level"][i],
                    "category": probe_res["category"][i],
                    "unsafe_prob": probe_res["unsafe_prob"][i],
                }
                if self.probe.has_lora:
                    entry["lora_hijack_prob"] = probe_res["lora_hijack_prob"][i]
                probe_stats.per_token.append(entry)
                if (probe_res["risk_level"][i] in self.alarm_levels
                        and probe_stats.asst_first_alarm_token_idx < 0):
                    probe_stats.asst_first_alarm_token_idx = len(probe_stats.per_token) - 1
            # Running max over the ACCEPTED prefix only (so alarms on rejected
            # draft tails don't inflate the alarm rate).
            for i in range(accept):
                up = probe_res["unsafe_prob"][i]
                if up > probe_stats.asst_max_unsafe_prob:
                    probe_stats.asst_max_unsafe_prob = float(up)
                    probe_stats.asst_max_category = probe_res["category"][i]
                if self.probe.has_lora:
                    lp = probe_res["lora_hijack_prob"][i]
                    if lp > probe_stats.asst_max_lora_hijack_prob:
                        probe_stats.asst_max_lora_hijack_prob = float(lp)

            # --- step 6: the probe hasn't seen bonus_tok -- but to keep its KV
            # consistent with the committed token stream, we MUST feed it.
            # The probe currently has KV = [prompt + prev_gen + draft_tokens].
            # We need KV = [prompt + prev_gen + draft[:accept] + bonus_tok].
            # So crop + feed bonus.
            self.probe.kv.crop(prompt_len + prev_gen_len + accept)
            bonus_cpu = torch.tensor([bonus_tok], dtype=torch.long)
            t_pr = time.perf_counter()
            if self.async_probe and self._probe_pool is not None:
                # Parallel point #2: bonus-token probe forward (cuda:2) is
                # independent of the draft KV-realign forward (cuda:1).  Launch
                # both and join at the end of this round.
                bonus_probe_future: Future = self._probe_pool.submit(
                    self.probe.score_assistant_tokens, bonus_cpu,
                )
            else:
                bonus_probe_future = None
                bonus_res = self.probe.score_assistant_tokens(bonus_cpu)
            # If async: we join below after kicking off the draft realign.
            if bonus_probe_future is None:
                probe_stats.probe_total_ms += (time.perf_counter() - t_pr) * 1000.0
            # else: timing recorded after .result()
            # --- step 7: align target & draft KV (identical to B1)
            kv_target.crop(prompt_len + prev_gen_len + accept)
            kv_draft.crop(prompt_len + prev_gen_len + accept)
            bt_d = torch.tensor([[bonus_tok]], dtype=torch.long, device=self.draft_device)
            out_d = self.draft(bt_d, past_key_values=kv_draft, use_cache=True)
            kv_draft = out_d.past_key_values
            last_draft_logit = out_d.logits[0, -1].clone()
            del out_d

            # Join bonus probe (if dispatched async) now that we have launched
            # the draft bonus-feed forward.  This gives host-side overlap of
            # cuda:1 (draft) and cuda:2 (bonus probe) kernel launches.
            if bonus_probe_future is not None:
                bonus_res = bonus_probe_future.result()
                probe_stats.probe_total_ms += (time.perf_counter() - t_pr) * 1000.0

            entry = {
                "token_id": bonus_tok,
                "risk_level": bonus_res["risk_level"][0],
                "category": bonus_res["category"][0],
                "unsafe_prob": bonus_res["unsafe_prob"][0],
            }
            if self.probe.has_lora:
                entry["lora_hijack_prob"] = bonus_res["lora_hijack_prob"][0]
            probe_stats.per_token.append(entry)
            if (bonus_res["risk_level"][0] in self.alarm_levels
                    and probe_stats.asst_first_alarm_token_idx < 0):
                probe_stats.asst_first_alarm_token_idx = len(probe_stats.per_token) - 1
            up_bonus = bonus_res["unsafe_prob"][0]
            if up_bonus > probe_stats.asst_max_unsafe_prob:
                probe_stats.asst_max_unsafe_prob = float(up_bonus)
                probe_stats.asst_max_category = bonus_res["category"][0]
            if self.probe.has_lora:
                lp = bonus_res["lora_hijack_prob"][0]
                if lp > probe_stats.asst_max_lora_hijack_prob:
                    probe_stats.asst_max_lora_hijack_prob = float(lp)

            generated.extend(committed_this_round)

            last_committed = bonus_tok

            if any(t in self.eos_token_ids for t in committed_this_round):
                break
            if len(generated) >= max_new_tokens:
                break

        self._finalize(generated, stats, probe_stats, t_wall)
        return torch.tensor(generated, dtype=torch.long)

    # ------------------------------------------------------------------
    # Z3 batched path: accumulate N rounds of committed tokens and feed them
    # to the probe in a single forward.  Amortises the host-side kernel-launch
    # overhead of the probe (which dominates when the probe model is small).
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_with_probe_batched(
        self,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int,
        draft_k: int,
        stats: SpecDecodeStats,
        probe_stats: ProbeStats,
        alarm_threshold: float,
    ) -> torch.LongTensor:
        """Z3: accumulate N rounds of committed tokens, probe once per flush.

        Key differences from the legacy path:
          * The probe sees ONLY committed tokens (no rejected draft tails),
            so probe KV never needs cropping -- it just grows with the output.
          * Alarm / running-max updates happen at flush boundaries, not every
            round.  The authoritative detection signal remains
            `asst_last_lora_hijack_prob` which is populated at `_finalize`
            time from the very last flush (we force a final flush below).
        """
        prompt_len = int(prompt_ids.shape[0])

        # ---------------- probe prefill ----------------
        t_probe_pf = time.perf_counter()
        self.probe.reset()
        _ = self.probe.score_user_turn(prompt_ids)
        probe_stats.user_turn_max_unsafe_prob = float(self.probe.max_unsafe_prob)
        probe_stats.user_turn_max_category = self.probe.max_unsafe_category
        if self.probe.has_lora:
            probe_stats.user_turn_max_lora_hijack_prob = float(self.probe.max_lora_hijack_prob)
        probe_pf_ms = (time.perf_counter() - t_probe_pf) * 1000.0

        prompt_target = prompt_ids.to(self.target_device).unsqueeze(0)
        prompt_draft = prompt_ids.to(self.draft_device).unsqueeze(0)

        t_wall = time.perf_counter()

        # ---------------- target + draft prefill ----------------
        t_pf = time.perf_counter()
        kv_target = DynamicCache()
        kv_draft = DynamicCache()

        out_t = self.target(prompt_target, past_key_values=kv_target, use_cache=True)
        kv_target = out_t.past_key_values
        first_tok = int(torch.argmax(out_t.logits[0, -1]).item())
        del out_t

        out_d = self.draft(prompt_draft, past_key_values=kv_draft, use_cache=True)
        kv_draft = out_d.past_key_values
        bt_d = torch.tensor([[first_tok]], dtype=torch.long, device=self.draft_device)
        out_d = self.draft(bt_d, past_key_values=kv_draft, use_cache=True)
        kv_draft = out_d.past_key_values
        last_draft_logit = out_d.logits[0, -1].clone()
        del out_d

        torch.cuda.synchronize(self.target_device)
        torch.cuda.synchronize(self.draft_device)
        stats.prefill_ms = (time.perf_counter() - t_pf) * 1000.0 + probe_pf_ms

        generated: List[int] = [first_tok]
        last_committed = first_tok

        # ---- pending-tokens buffer (Z3) ----
        # Each entry: (token_id, "role")  role is purely diagnostic.
        # We flush when len(pending) covers >= probe_batch_rounds rounds.
        pending_tokens: List[int] = [first_tok]
        pending_future: Optional[Future] = None
        rounds_since_flush = 0

        def _record_probe_results(tokens: List[int], probe_res: dict) -> None:
            """Fold probe output into probe_stats / per_token in order."""
            n = len(tokens)
            for i in range(n):
                entry = {
                    "token_id": tokens[i],
                    "risk_level": probe_res["risk_level"][i],
                    "category": probe_res["category"][i],
                    "unsafe_prob": probe_res["unsafe_prob"][i],
                }
                if self.probe.has_lora:
                    entry["lora_hijack_prob"] = probe_res["lora_hijack_prob"][i]
                probe_stats.per_token.append(entry)
                if (probe_res["risk_level"][i] in self.alarm_levels
                        and probe_stats.asst_first_alarm_token_idx < 0):
                    probe_stats.asst_first_alarm_token_idx = len(probe_stats.per_token) - 1
                up = probe_res["unsafe_prob"][i]
                if up > probe_stats.asst_max_unsafe_prob:
                    probe_stats.asst_max_unsafe_prob = float(up)
                    probe_stats.asst_max_category = probe_res["category"][i]
                if self.probe.has_lora:
                    lp = probe_res["lora_hijack_prob"][i]
                    if lp > probe_stats.asst_max_lora_hijack_prob:
                        probe_stats.asst_max_lora_hijack_prob = float(lp)

        # Bound closures over mutable state via a tiny helper dict.
        state = {"pending_future": None, "pending_tokens_pf": []}

        def _submit_flush() -> None:
            """Dispatch a probe forward for the current pending buffer."""
            if not pending_tokens:
                return
            tokens_cpu = torch.tensor(pending_tokens, dtype=torch.long)
            tokens_copy = list(pending_tokens)  # snapshot for the join side
            state["pending_tokens_pf"] = tokens_copy
            if self.async_probe and self._probe_pool is not None:
                state["pending_future"] = self._probe_pool.submit(
                    self.probe.score_assistant_tokens, tokens_cpu,
                )
            else:
                # sync path: run immediately, record now
                t0 = time.perf_counter()
                res = self.probe.score_assistant_tokens(tokens_cpu)
                probe_stats.probe_total_ms += (time.perf_counter() - t0) * 1000.0
                probe_stats.probe_calls += 1
                _record_probe_results(tokens_copy, res)
                state["pending_tokens_pf"] = []

        def _join_pending(t_submit_ref: Optional[float]) -> None:
            """If an async probe future is in flight, join + fold results."""
            fut = state["pending_future"]
            if fut is None:
                return
            tokens_copy = state["pending_tokens_pf"]
            t0 = time.perf_counter() if t_submit_ref is None else t_submit_ref
            res = fut.result()
            probe_stats.probe_total_ms += (time.perf_counter() - t0) * 1000.0
            probe_stats.probe_calls += 1
            _record_probe_results(tokens_copy, res)
            state["pending_future"] = None
            state["pending_tokens_pf"] = []

        if first_tok in self.eos_token_ids:
            # Flush synchronously; async not worth it for a 1-tok forward.
            t0 = time.perf_counter()
            res = self.probe.score_assistant_tokens(
                torch.tensor(pending_tokens, dtype=torch.long)
            )
            probe_stats.probe_total_ms += (time.perf_counter() - t0) * 1000.0
            probe_stats.probe_calls += 1
            _record_probe_results(list(pending_tokens), res)
            pending_tokens.clear()
            self._finalize(generated, stats, probe_stats, t_wall)
            return torch.tensor(generated, dtype=torch.long)

        # ---------------- speculative rounds ----------------
        while len(generated) < max_new_tokens:
            stats.n_rounds += 1
            probe_stats.asst_rounds += 1
            prev_gen_len = len(generated)

            # --- step 1: draft k tokens
            draft_tokens, kv_draft, _ = self._draft_k(
                kv_draft, last_draft_logit, draft_k,
            )
            k_actual = len(draft_tokens)
            stats.draft_tokens += k_actual

            # --- step 2: target verify (no probe this round; probe is batched)
            verify_input = [last_committed] + draft_tokens
            verify_t = torch.tensor([verify_input], dtype=torch.long,
                                    device=self.target_device)
            out_t = self.target(verify_t, past_key_values=kv_target, use_cache=True)
            kv_target = out_t.past_key_values
            preds = torch.argmax(out_t.logits[0], dim=-1).tolist()
            del out_t

            # --- step 3: accept / bonus
            accept = 0
            for i in range(k_actual):
                if preds[i] == draft_tokens[i]:
                    accept += 1
                else:
                    break
            stats.accepted_tokens += accept
            stats.accept_lengths.append(accept)
            bonus_tok = preds[accept]
            committed_this_round = draft_tokens[:accept] + [bonus_tok]
            stats.bonus_tokens += 1

            # --- step 4: KV align (target + draft).  Probe KV is untouched;
            # we'll catch it up at flush time with only the committed tokens.
            kv_target.crop(prompt_len + prev_gen_len + accept)
            kv_draft.crop(prompt_len + prev_gen_len + accept)
            bt_d = torch.tensor([[bonus_tok]], dtype=torch.long, device=self.draft_device)
            out_d = self.draft(bt_d, past_key_values=kv_draft, use_cache=True)
            kv_draft = out_d.past_key_values
            last_draft_logit = out_d.logits[0, -1].clone()
            del out_d

            generated.extend(committed_this_round)
            pending_tokens.extend(committed_this_round)
            last_committed = bonus_tok
            rounds_since_flush += 1

            hit_eos = any(t in self.eos_token_ids for t in committed_this_round)
            hit_cap = len(generated) >= max_new_tokens

            # --- step 5: flush decision
            # On a normal round boundary, if we've accumulated >= batch_rounds
            # rounds of tokens, flush.  On EOS / cap we force a flush below.
            if rounds_since_flush >= self.probe_batch_rounds and not (hit_eos or hit_cap):
                # Before submitting a new flush, join the previous one if any.
                _join_pending(None)
                _submit_flush()
                pending_tokens.clear()
                rounds_since_flush = 0

            if hit_eos or hit_cap:
                break

        # ---------------- final flush ----------------
        # Make sure both the in-flight async future (if any) and any trailing
        # unflushed tokens are scored, so per_token + max/last stats cover
        # every generated token.
        #
        # Two modes:
        #   (a) legacy: join prev flush + sync run trailing flush, then
        #       settle wall_ms.  The trailing probe forward (~30-70 ms)
        #       is included in the user-perceived wall time.
        #   (b) A2 defer_final_flush: submit the trailing flush as a new
        #       async future (do NOT join prev yet), settle wall_ms
        #       IMMEDIATELY, then join everything post-wall.  Matches the
        #       guideline §4.1(4) semantics that the probe is an observer
        #       and must not delay user-perceived generation.
        if self.defer_final_flush:
            # (b) A2 path --------------------------------------------------
            # If there are trailing tokens, submit them as a new async flush.
            # We keep the previous async flush (if any) still in flight;
            # since the ThreadPool has max_workers=1 the new submit queues
            # after the previous one, and BOTH are joined post-wall.
            trailing_tokens_copy: List[int] = []
            trailing_future: Optional[Future] = None
            if pending_tokens:
                trailing_tokens_copy = list(pending_tokens)
                trailing_tokens_cpu = torch.tensor(pending_tokens, dtype=torch.long)
                trailing_future = self._probe_pool.submit(
                    self.probe.score_assistant_tokens, trailing_tokens_cpu,
                )
                pending_tokens.clear()

            # Settle wall_ms NOW -- generation is done from the user's PoV.
            self._finalize(generated, stats, probe_stats, t_wall)

            # Post-wall: join the in-flight previous flush (if any) first,
            # then the trailing flush.  Costs are accounted to probe_postwall_ms.
            t_pw = time.perf_counter()
            prev_fut = state["pending_future"]
            if prev_fut is not None:
                prev_tokens = state["pending_tokens_pf"]
                prev_res = prev_fut.result()
                probe_stats.probe_calls += 1
                _record_probe_results(prev_tokens, prev_res)
                state["pending_future"] = None
                state["pending_tokens_pf"] = []
            if trailing_future is not None:
                trailing_res = trailing_future.result()
                probe_stats.probe_calls += 1
                _record_probe_results(trailing_tokens_copy, trailing_res)
            probe_stats.probe_postwall_ms = (time.perf_counter() - t_pw) * 1000.0

            # Refresh asst_last_lora_hijack_prob now that per_token is complete.
            if self.probe.has_lora and probe_stats.per_token:
                last_entry = probe_stats.per_token[-1]
                probe_stats.asst_last_lora_hijack_prob = float(
                    last_entry.get("lora_hijack_prob", 0.0)
                )
            return torch.tensor(generated, dtype=torch.long)

        # (a) legacy path --------------------------------------------------
        _join_pending(None)
        if pending_tokens:
            t0 = time.perf_counter()
            res = self.probe.score_assistant_tokens(
                torch.tensor(pending_tokens, dtype=torch.long)
            )
            probe_stats.probe_total_ms += (time.perf_counter() - t0) * 1000.0
            probe_stats.probe_calls += 1
            _record_probe_results(list(pending_tokens), res)
            pending_tokens.clear()

        self._finalize(generated, stats, probe_stats, t_wall)
        return torch.tensor(generated, dtype=torch.long)

    # ------------------------------------------------------------------
    def _finalize(self, generated, stats, probe_stats, t_wall) -> None:
        torch.cuda.synchronize(self.target_device)
        torch.cuda.synchronize(self.draft_device)
        stats.wall_ms = (time.perf_counter() - t_wall) * 1000.0
        stats.decode_ms = stats.wall_ms - stats.prefill_ms
        stats.generated_tokens = len(generated)
        # Last-token LoRA prob (if LoRA attached): this matches the training-time
        # evaluation setup and is the authoritative detection signal.
        if self.probe.has_lora and probe_stats.per_token:
            last_entry = probe_stats.per_token[-1]
            probe_stats.asst_last_lora_hijack_prob = float(
                last_entry.get("lora_hijack_prob", 0.0)
            )
        # trim at first EOS (purely cosmetic; we don't shrink probe_stats so the
        # per-token record retains the final EOS alarm status if present).
        for i, t in enumerate(generated):
            if t in self.eos_token_ids:
                del generated[i + 1:]
                stats.generated_tokens = len(generated)
                break


# ---------------------------------------------------------------------------
# Quick CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    from .dataset_adapter import load_attack_samples

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--lora_dir", type=str, default=None,
                        help="Load a Phase 3 LoRA adapter + hijack head")
    args = parser.parse_args()

    probe = BatchedGuardProbe(lora_dir=args.lora_dir)
    engine = SpecDecoderWithProbe(probe=probe)
    samples = load_attack_samples(limit=args.index + 1)
    s = samples[args.index]
    prompt_ids = engine.apply_chat(s.messages)
    log.info("sample %s | prompt_tokens=%d | attack_goal=%s",
             s.sample_id, prompt_ids.shape[0], s.attack_goal[:120])

    stats = SpecDecodeStats()
    pstats = ProbeStats()
    out_ids = engine.generate_with_probe(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        draft_k=args.k,
        stats=stats,
        probe_stats=pstats,
    )
    text = engine.tokenizer.decode(out_ids.tolist(), skip_special_tokens=True)
    log.info("=== OUTPUT (%d gen tokens, wall=%.1f ms) ===\n%s",
             len(out_ids), stats.wall_ms, text)
    log.info("=== DECODE STATS ===")
    log.info("  acceptance_rate=%.3f  mean_accept_len=%.2f  tokens/s=%.1f",
             stats.acceptance_rate, stats.mean_accept_len, stats.tokens_per_sec)
    log.info("=== PROBE STATS ===")
    log.info("  user_turn_max_unsafe=%.3f cat=%s",
             pstats.user_turn_max_unsafe_prob, pstats.user_turn_max_category)
    log.info("  asst_max_unsafe=%.3f cat=%s first_alarm_tok_idx=%d",
             pstats.asst_max_unsafe_prob, pstats.asst_max_category,
             pstats.asst_first_alarm_token_idx)
    if probe.has_lora:
        log.info("  [LoRA] user_turn_max_lora_hijack=%.3f  asst_max_lora_hijack=%.3f",
                 pstats.user_turn_max_lora_hijack_prob,
                 pstats.asst_max_lora_hijack_prob)
    log.info("  probe_total_ms=%.1f  (over %d rounds)",
             pstats.probe_total_ms, pstats.asst_rounds)
    # dump a few per-token entries for sanity
    log.info("  per_token (first 15 of %d):", len(pstats.per_token))
    for i, pt in enumerate(pstats.per_token[:15]):
        tok_str = engine.tokenizer.decode([pt["token_id"]])
        log.info("    [%2d] %-12r risk=%-13s cat=%-15s unsafe_p=%.3f",
                 i, tok_str, pt["risk_level"], pt["category"], pt["unsafe_prob"])

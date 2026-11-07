"""PI-conditional teacher by additive multi-LoRA composition (App. C.3, Eq. 14).

    pi_T^(P)(. | s_t, p_P(x)) ∝ pi_T^base(. | s_t) * prod_{i in P} exp(Delta_i)
    Delta_i = W^T (h^{theta_i}_{s_t, p_i(x)} - h^base_{s_t})

Because W is shared, Delta_i equals the logit difference between the adapter-i call (with its own
instance PI prefix) and the base call, so the composite logits are base + sum_i (logits_i - base).
kappa_t = TV(pi_T, pi_T^(P)) (Eq. kappa_lite) is read from the same pass.

:class:`PIConditionalTeacher` runs the |P|+1 variants as one left-padded batch.  With a PEFT
model the variants use PEFT's mixed-adapter batches (``adapter_names``); any other model is run
variant-by-variant through the ``adapter_context`` callback.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence

import torch

from chop.divergences import total_variation

BASE = "__base__"


def compose_logits(base: torch.Tensor, adapter_logits: Sequence[torch.Tensor]) -> torch.Tensor:
    out = base.float().clone()
    for a in adapter_logits:
        out += a.float() - base.float()
    return out


def kappa_hat(base_logits: torch.Tensor, composite_logits: torch.Tensor) -> torch.Tensor:
    """Joint-TV PI-dependence proxy per state."""
    return total_variation(base_logits, composite_logits)


def left_pad(seqs: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad so that sequence ends align; returns (input_ids, attention_mask, position_ids)."""
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros(len(seqs), n, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s):
            ids[i, n - len(s):] = torch.as_tensor(list(s), dtype=torch.long)
            mask[i, n - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    return ids, mask, pos


class PIConditionalTeacher:
    """Frozen teacher with one LoRA adapter per PI source.

    ``model`` is an HF causal LM (optionally a PEFT model whose adapters are named after the PI
    sources).  ``interface='prompt'`` implements ablation (g): all PI blocks are concatenated
    into one prompt prefix of the base model and no adapter is used.
    """

    def __init__(self, model, pad_id: int = 0, interface: str = "lora_prefix",
                 adapter_context: Callable[[str], contextlib.AbstractContextManager] | None = None):
        self.model = model
        self.pad_id = pad_id
        self.interface = interface
        self._peft = hasattr(model, "peft_config")
        self.adapter_context = adapter_context or self._default_context

    # ------------------------------------------------------------------ plumbing
    def _default_context(self, name: str):
        if not self._peft:
            if name != BASE:
                raise ValueError("non-PEFT teacher can only run the base variant")
            return contextlib.nullcontext()
        if name == BASE:
            return self.model.disable_adapter()
        self.model.set_adapter(name)
        return contextlib.nullcontext()

    def _variants(self, prompt_ids, response_ids, pi_prefix: dict[str, list[int]], sources: list[str]):
        variants = [(BASE, list(prompt_ids) + list(response_ids))]
        if not sources:
            return variants
        if self.interface == "prompt":
            prefix = [t for s in sources for t in pi_prefix[s]]
            variants.append((BASE, prefix + list(prompt_ids) + list(response_ids)))
        else:
            variants += [(s, list(pi_prefix[s]) + list(prompt_ids) + list(response_ids)) for s in sources]
        return variants

    @torch.no_grad()
    def _forward(self, names: list[str], ids, mask, pos, past=None):
        if self._peft and len(set(names)) > 1:
            return self.model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=past,
                              use_cache=past is not None, adapter_names=names)
        # homogeneous batch or non-PEFT model
        with self.adapter_context(names[0]):
            return self.model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=past,
                              use_cache=past is not None)

    @torch.no_grad()
    def _logits_per_variant(self, variants) -> list[torch.Tensor]:
        names = [n for n, _ in variants]
        ids, mask, pos = left_pad([s for _, s in variants], self.pad_id)
        if self._peft or len(set(names)) == 1:
            return list(self._forward(names, ids, mask, pos).logits.float())
        out = []
        for i, name in enumerate(names):
            with self.adapter_context(name):
                out.append(self.model(input_ids=ids[i:i + 1], attention_mask=mask[i:i + 1],
                                      position_ids=pos[i:i + 1]).logits[0].float())
        return out

    # --------------------------------------------------------------------- API
    @torch.no_grad()
    def score(self, prompt_ids: Sequence[int], response_ids: Sequence[int], pi_prefix: dict[str, list[int]],
              sources: list[str]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Teacher logits at every response position (predicting response token j).

        Returns (base [R, V], composite [R, V] or None when ``sources`` is empty).
        """
        r = len(response_ids)
        variants = self._variants(prompt_ids, response_ids, pi_prefix, sources)
        logits = self._logits_per_variant(variants)
        # with left padding all variants end at the same column; response token j is predicted
        # by column (n - r - 1 + j)
        sl = [lg[lg.shape[0] - r - 1: lg.shape[0] - 1] for lg in logits]
        base = sl[0]
        if len(sl) == 1:
            return base, None
        return base, compose_logits(base, sl[1:])

    @torch.no_grad()
    def sample(self, context_ids: Sequence[int], pi_prefix: dict[str, list[int]], sources: list[str],
               num_samples: int, max_new_tokens: int, temperature: float = 0.7,
               eos_id: int | None = None,
               generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample continuations from the (composite) teacher at state ``context_ids``.

        Every decoding step runs the |P|+1 variants for each sample with a KV cache, composes their
        last-position logits and feeds the sampled token back to all variants of that sample.
        Returns (tokens [n, L], lengths [n]); tokens after EOS are ``pad_id``.
        """
        variants = self._variants(context_ids, [], pi_prefix, sources)
        nv = len(variants)
        names = [n for n, _ in variants for _ in range(num_samples)]
        ids, mask, pos = left_pad([s for _, s in variants for _ in range(num_samples)], self.pad_id)
        tokens = torch.full((num_samples, max_new_tokens), self.pad_id, dtype=torch.long)
        done = torch.zeros(num_samples, dtype=torch.bool)
        lengths = torch.full((num_samples,), max_new_tokens, dtype=torch.long)
        past = self._new_cache()
        step_ids, step_mask, step_pos = ids, mask, pos
        for t in range(max_new_tokens):
            out = self._forward_cached(names, step_ids, step_mask, step_pos, past)
            past = out.past_key_values
            last = out.logits[:, -1].float().view(nv, num_samples, -1)
            comp = compose_logits(last[0], list(last[1:])) if nv > 1 else last[0]
            probs = torch.softmax(comp / max(temperature, 1e-6), dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            tokens[:, t] = nxt
            if eos_id is not None:
                newly = (~done) & (nxt == eos_id)
                lengths[newly] = t + 1
                done |= newly
            if bool(done.all()):
                break
            step_ids = nxt.repeat(nv).unsqueeze(-1)
            step_mask = torch.cat([step_mask, torch.ones(step_mask.shape[0], 1, dtype=torch.long)], dim=-1)
            step_pos = step_pos[:, -1:] + 1
        return tokens, lengths

    def _new_cache(self):
        try:
            from transformers import DynamicCache

            return DynamicCache()
        except ImportError:  # pragma: no cover
            return None

    @torch.no_grad()
    def _forward_cached(self, names, ids, mask, pos, past):
        if self._peft and len(set(names)) > 1:
            return self.model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=past,
                              use_cache=True, adapter_names=names)
        if len(set(names)) == 1:
            with self.adapter_context(names[0]):
                return self.model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=past,
                                  use_cache=True)
        raise NotImplementedError("heterogeneous variants need a PEFT model with mixed-adapter batches")

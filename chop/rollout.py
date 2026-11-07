"""Single-process sampling with rolling left truncation (reference path; verl uses vLLM).

Decoding uses temperature/top-p with a KV cache.  :func:`generate_response` follows the chunk plan
of :func:`chop.truncation.rolling_chunks`: at every chunk boundary the context is re-prefilled
with the truncated prefix, so each token is conditioned exactly as the scoring pass conditions it.
:func:`sample_segment` continues a fixed context without re-prefill (used for the L_a-token
segment branches, for which the chunk plan reserves window space).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from chop.truncation import rolling_chunks


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(-1)
    remove = probs.cumsum(-1) - probs > top_p   # keep the smallest prefix with mass >= top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)


def sample_token(logits: torch.Tensor, temperature: float, top_p: float,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1)
    probs = _top_p_filter(logits.float() / temperature, top_p).softmax(-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def sample_segment(model, context: Sequence[int], max_new_tokens: int, temperature: float = 0.7,
                   top_p: float = 1.0, eos_id: int | None = None,
                   generator: torch.Generator | None = None) -> list[int]:
    """Sample one continuation of ``context`` (stops after EOS)."""
    from transformers import DynamicCache

    past = DynamicCache()
    ids = torch.tensor([list(context)])
    pos = torch.arange(ids.shape[1])[None]
    out: list[int] = []
    for _ in range(max_new_tokens):
        o = model(input_ids=ids, position_ids=pos, past_key_values=past, use_cache=True)
        past = o.past_key_values
        tok = int(sample_token(o.logits[0, -1], temperature, top_p, generator))
        out.append(tok)
        if eos_id is not None and tok == eos_id:
            break
        ids = torch.tensor([[tok]])
        pos = pos[:, -1:] + 1
    return out


@torch.no_grad()
def generate_response(model, prompt_ids: Sequence[int], max_new_tokens: int, temperature: float = 0.7,
                      top_p: float = 0.95, eos_id: int | None = None, window: int | None = None,
                      chunk: int = 4096, reserve: int = 0, generator: torch.Generator | None = None) -> list[int]:
    """Generate a response under the rolling chunk plan (plain decoding when it fits the window)."""
    prompt = list(prompt_ids)
    if window is None or len(prompt) + max_new_tokens + reserve <= window:
        return sample_segment(model, prompt, max_new_tokens, temperature, top_p, eos_id, generator)
    resp: list[int] = []
    for c in rolling_chunks(len(prompt), max_new_tokens, window, chunk, reserve):
        got = sample_segment(model, prompt + resp[c.ctx_start:c.gen_start], c.gen_end - c.gen_start,
                             temperature, top_p, eos_id, generator)
        resp.extend(got)
        if eos_id is not None and got and got[-1] == eos_id:
            break
    return resp

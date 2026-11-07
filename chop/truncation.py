"""Rolling left truncation for cumulative generation caps beyond the native window.

The math sweep goes to a 60K cumulative generated-token budget while every model keeps its
native 32,768-token context.  Rollout and scoring must condition each token on the *same*
truncated context, so both use the chunk plan below: generation proceeds in chunks of
``chunk`` tokens, and before chunk [g0, g0 + chunk) the context is re-prefilled as

    prompt + response[c : g0],   c = max(0, g0 + chunk - (window - reserve - prompt_len))

which guarantees prompt + response[c : g0 + chunk] + ``reserve`` tokens fit the window.  The
plan depends only on g0 (not on where generation stops), so scoring re-runs each chunk with the
same c and reproduces the rollout's conditioning exactly.  ``reserve`` leaves room for segment
branches of length L_a sampled inside a chunk without re-prefill.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    ctx_start: int   # first response token kept in the context
    gen_start: int   # first response position generated / scored in this chunk
    gen_end: int     # exclusive


def rolling_chunks(prompt_len: int, response_len: int, window: int, chunk: int, reserve: int = 0) -> list[Chunk]:
    budget = window - reserve - prompt_len
    if budget <= 0:
        raise ValueError(f"prompt ({prompt_len}) + reserve ({reserve}) do not fit the native window ({window})")
    chunk = min(chunk, budget)
    return [Chunk(max(0, g0 + chunk - budget), g0, min(g0 + chunk, response_len))
            for g0 in range(0, response_len, chunk)]


def chunk_of(position: int, chunks: list[Chunk]) -> Chunk:
    """The chunk whose generated range contains response position ``position``."""
    for c in chunks:
        if c.gen_start <= position < c.gen_end:
            return c
    raise IndexError(f"position {position} outside the planned response")


def needs_truncation(prompt_len: int, response_len: int, window: int) -> bool:
    return prompt_len + response_len > window


def chunk_inputs(prompt_ids: list[int], response_ids: list[int], c: Chunk) -> tuple[list[int], int]:
    """Input ids for scoring chunk ``c`` and the index of response position ``c.gen_start`` in them."""
    ids = list(prompt_ids) + list(response_ids[c.ctx_start:c.gen_end])
    return ids, len(prompt_ids) + (c.gen_start - c.ctx_start)

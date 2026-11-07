"""The PI-conditioned token interface (App. C.2).

For every PI-conditioned teacher call the instance block p_i(x) is supplied through a dedicated
``[PI]`` prefix placed before the prompt, and adapter theta_i is active.  The unconditional call
has no prefix and no adapter.  Markers are plain text so that no tokenizer resize is needed; the
audit condition replaces the content by the ``[NO_PI]`` sentinel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

PI_OPEN = "[PI source={name}]\n"
PI_CLOSE = "\n[/PI]\n"
NO_PI_SENTINEL = "[NO_PI]"


def format_pi_block(name: str, content: str, encode: Callable[[str], list[int]] | None = None,
                    decode: Callable[[Sequence[int]], str] | None = None, max_tokens: int | None = None) -> str:
    """Render one PI block, truncating ``content`` to ``max_tokens`` tokens when a tokenizer is given."""
    if max_tokens is not None and encode is not None and decode is not None:
        ids = encode(content)
        if len(ids) > max_tokens:
            content = decode(ids[:max_tokens])
    return PI_OPEN.format(name=name) + content + PI_CLOSE


def sentinel_block(name: str) -> str:
    return format_pi_block(name, NO_PI_SENTINEL)


def teacher_input(prefix_ids: list[int], prompt_ids: list[int], response_ids: list[int]) -> tuple[list[int], int]:
    """Concatenate [PI prefix] + prompt + response; returns (ids, index of the first response token)."""
    ids = list(prefix_ids) + list(prompt_ids) + list(response_ids)
    return ids, len(prefix_ids) + len(prompt_ids)

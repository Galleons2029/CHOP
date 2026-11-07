"""Leakage and content controls (tab:swebench_pi_main, tab:pi_gating_full).

shuffled      : PI blocks permuted within the batch (derangement: wrong content, same capacity)
wrong_repo    : each instance receives a block from an instance of a different repository
sentinel      : content replaced by the [NO_PI] sentinel (PI-removed teacher audit)
random_adapter: LoRA weights re-drawn at matched norm, no PI (capacity control)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from chop.pi.serialize import NO_PI_SENTINEL


def derangement(n: int, generator: torch.Generator | None = None) -> list[int]:
    """Uniform random permutation with no fixed points (n >= 2)."""
    if n < 2:
        raise ValueError("a derangement needs at least two items")
    while True:
        p = torch.randperm(n, generator=generator).tolist()
        if all(i != j for i, j in enumerate(p)):
            return p


def shuffle_within_batch(blocks: Sequence[str], generator: torch.Generator | None = None) -> list[str]:
    p = derangement(len(blocks), generator)
    return [blocks[j] for j in p]


def wrong_repo(blocks: Sequence[str], repos: Sequence[str], generator: torch.Generator | None = None) -> list[str]:
    """Replace each block by one drawn from an instance of a different repository."""
    out = []
    for repo in repos:
        pool = [j for j, r in enumerate(repos) if r != repo]
        if not pool:
            raise ValueError(f"no instance from a repository other than {repo!r} in the batch")
        j = pool[int(torch.randint(len(pool), (1,), generator=generator))]
        out.append(blocks[j])
    return out


def sentinel(blocks: Sequence[str]) -> list[str]:
    return [NO_PI_SENTINEL for _ in blocks]


def apply_content_control(mode: str, blocks: Sequence[str], repos: Sequence[str] | None = None,
                          generator: torch.Generator | None = None) -> list[str]:
    if mode in ("true", "random_adapter"):
        return list(blocks)
    if mode == "shuffled":
        return shuffle_within_batch(blocks, generator)
    if mode == "wrong_repo":
        if repos is None:
            raise ValueError("wrong_repo needs the repository of every instance")
        return wrong_repo(blocks, repos, generator)
    if mode == "sentinel":
        return sentinel(blocks)
    raise ValueError(f"unknown PI content control {mode!r}")


@torch.no_grad()
def randomize_lora_(model, adapter_name: str, generator: torch.Generator | None = None) -> int:
    """Re-draw every LoRA A/B matrix of ``adapter_name`` with the same Frobenius norm.

    Keeps adapter capacity and update scale while removing anything learned from PI.
    Returns the number of tensors changed.
    """
    changed = 0
    for name, p in model.named_parameters():
        if f".{adapter_name}." in name and ("lora_A" in name or "lora_B" in name):
            norm = p.norm()
            fresh = torch.randn(p.shape, generator=generator, dtype=torch.float32).to(p)
            p.copy_(fresh * (norm / fresh.norm().clamp_min(1e-12)))
            changed += 1
    return changed

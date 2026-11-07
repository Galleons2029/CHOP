"""PI source redundancy (App. pi_redundancy): pairwise mutual information and Shapley-style kappa.

f(S) = TV(pi_T, pi_T^(S)) for a source subset S, with the additive composition of Eq. 14. The Shapley
value of source i averages its marginal contribution to f over all orderings (exact for |P| <= 6).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from chop.divergences import total_variation
from chop.pi.composition import compose_logits


def kappa_subset(base: torch.Tensor, variants: dict[str, torch.Tensor], subset) -> torch.Tensor:
    if not subset:
        return torch.zeros(base.shape[:-1])
    return total_variation(base, compose_logits(base, [variants[s] for s in subset]))


def shapley_kappa(base: torch.Tensor, variants: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    names = list(variants)
    n = len(names)
    phi = {s: torch.zeros(base.shape[:-1]) for s in names}
    cache = {}

    def f(sub):
        key = tuple(sorted(sub))
        if key not in cache:
            cache[key] = kappa_subset(base, variants, key)
        return cache[key]

    for i, s in enumerate(names):
        others = names[:i] + names[i + 1:]
        for k in range(n):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for sub in itertools.combinations(others, k):
                phi[s] += w * (f(sub + (s,)) - f(sub))
    return phi


def mutual_information(a, b) -> float:
    """MI (nats) between two binary gate sequences."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    mi = 0.0
    for x in (False, True):
        for y in (False, True):
            pxy = np.mean((a == x) & (b == y))
            px, py = np.mean(a == x), np.mean(b == y)
            if pxy > 0:
                mi += pxy * math.log(pxy / (px * py))
    return float(mi)


def pairwise_mi(gates: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    return {(x, y): mutual_information(gates[x], gates[y]) for x, y in itertools.combinations(sorted(gates), 2)}

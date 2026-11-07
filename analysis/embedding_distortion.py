"""Assumption A3 audit: W_2(e#mu, e#nu) <= L_e sqrt(TV(mu, nu)) on paired segment clouds (App. D.2).

TV between the segment distributions is estimated by the overlap of their empirical first-token
distributions, as in the paper. W_2 between embedded clouds is computed exactly for equal-size clouds
(assignment over permutations, via scipy's linear_sum_assignment).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from chop.monitors import fit_lipschitz


def w2_equal(a: np.ndarray, b: np.ndarray) -> float:
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    return float(np.sqrt(cost[r, c].mean()))


def first_token_tv(segs_a: list[list[int]], segs_b: list[list[int]]) -> float:
    ca = Counter(s[0] for s in segs_a if s)
    cb = Counter(s[0] for s in segs_b if s)
    na, nb = max(sum(ca.values()), 1), max(sum(cb.values()), 1)
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))


def estimate_le(pairs: list[tuple[np.ndarray, np.ndarray, list[list[int]], list[list[int]]]],
                quantile: float = 0.95) -> dict:
    w2 = [w2_equal(ea, eb) for ea, eb, _, _ in pairs]
    tv = [first_token_tv(sa, sb) for _, _, sa, sb in pairs]
    return {"L_e": fit_lipschitz(w2, np.sqrt(tv), quantile), "pairs": len(pairs)}

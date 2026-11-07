"""Training-time monitors (App. D.2): embedding collapse guard and A3 distortion audit."""

from __future__ import annotations

from collections import deque

import numpy as np

from chop.segment import effective_rank

__all__ = ["CollapseGuard", "effective_rank", "fit_lipschitz"]


class CollapseGuard:
    """Active fallback for segment-embedding collapse.

    If the effective rank of the segment-embedding covariance falls below ``threshold`` (default
    200, half the [280, 410] floor observed in the paper), Cell C/D states revert to Cell A and the
    embedding encoder is frozen until every rank in the trailing ``window`` observations is back
    above the threshold.
    """

    def __init__(self, threshold: float = 200.0, window: int = 10):
        self.threshold = threshold
        self.history: deque[float] = deque(maxlen=window)
        self.active = False

    def observe(self, rank: float) -> bool:
        """Record one effective-rank measurement; returns True while the fallback is active."""
        self.history.append(rank)
        if not self.active and rank < self.threshold:
            self.active = True
        elif self.active and len(self.history) == self.history.maxlen and min(self.history) >= self.threshold:
            self.active = False
        return self.active

    def state_dict(self) -> dict:
        return {"history": list(self.history), "active": self.active}

    def load_state_dict(self, state: dict) -> None:
        self.history.clear()
        self.history.extend(state["history"])
        self.active = state["active"]


def fit_lipschitz(w2, sqrt_tv, quantile: float = 0.95) -> float:
    """L_e in W_2(e#mu, e#nu) <= L_e sqrt(TV(mu, nu)) (A3): the given quantile of W_2 / sqrt(TV).

    The paper fits a line through the origin and reports the 95% quantile of the ratio
    distribution (L_e = 1.4 on 20K SWE-bench segment pairs).
    """
    w = np.asarray(w2, dtype=np.float64).ravel()
    s = np.asarray(sqrt_tv, dtype=np.float64).ravel()
    ok = s > 1e-12
    return float(np.quantile(w[ok] / s[ok], quantile))

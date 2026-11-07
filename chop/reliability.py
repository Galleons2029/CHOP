"""Reliability proxy rho_t (Eq. rho) and its calibration (App. B).

rho_t = exp(-lambda_1 H(q_t) - lambda_2 TV(q_t, p_t)) is read from logits that OPD already
computes.  Routing only uses streaming quantiles of rho_t, which are invariant to monotone
transforms, so the routing decision depends on (lambda_1, lambda_2) only through their ratio;
the overall scale matters only for reporting rho on the paper's absolute scale.
:func:`calibrate_lambdas` therefore fits the ratio by rank correlation with the reference score
rho_ref and then the scale by least squares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import optimize, stats

from chop.config import ReliabilityConfig


def reliability(entropy: torch.Tensor, tv: torch.Tensor, cfg: ReliabilityConfig,
                rho_ref: torch.Tensor | None = None) -> torch.Tensor:
    """rho_t for the variant selected in ``cfg`` (all variants are monotone in -H and -TV)."""
    if cfg.variant == "calibrated":
        return torch.exp(-cfg.lambda_entropy * entropy - cfg.lambda_tv * tv)
    if cfg.variant == "free":
        return torch.exp(-entropy / cfg.entropy_max - tv / cfg.tv_max)
    if cfg.variant == "entropy_only":
        return torch.exp(-entropy / cfg.entropy_max)
    if cfg.variant == "tv_only":
        return torch.exp(-tv / cfg.tv_max)
    if cfg.variant == "reference":
        if rho_ref is None:
            raise ValueError("variant='reference' needs rho_ref (teacher self-rollout success rates)")
        return rho_ref
    raise ValueError(f"unknown reliability variant {cfg.variant!r}")


def logistic_reliability(entropy: torch.Tensor, tv: torch.Tensor, a: float, b: float, h_max: float) -> torch.Tensor:
    """Logistic-squash parameterisation of App. A.4; induces the same quantile routing."""
    return torch.sigmoid(a * (h_max - entropy) - b * tv)


def reference_reliability(successes: torch.Tensor, trials: int) -> torch.Tensor:
    """rho_ref: empirical success rate of teacher self-rollouts continued from s_t."""
    return successes.float() / float(trials)


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(stats.spearmanr(x, y).statistic)


@dataclass
class Calibration:
    lambda_entropy: float
    lambda_tv: float
    spearman: float
    mse: float


def calibrate_lambdas(entropy, tv, rho_ref, n_angles: int = 181) -> Calibration:
    """Fit (lambda_1, lambda_2) on a calibration set (the paper uses 1024 states, then freezes).

    Stage 1: direction (cos a, sin a), a in [0, pi/2], maximising Spearman(-(cos a H + sin a TV), rho_ref).
    Stage 2: scale s minimising mean (exp(-s (cos a H + sin a TV)) - rho_ref)^2.
    """
    h = np.asarray(entropy, dtype=np.float64).ravel()
    t = np.asarray(tv, dtype=np.float64).ravel()
    r = np.asarray(rho_ref, dtype=np.float64).ravel()
    best_a, best_rs = 0.0, -np.inf
    for a in np.linspace(0.0, math.pi / 2, n_angles):
        rs = spearman(-(math.cos(a) * h + math.sin(a) * t), r)
        if np.isfinite(rs) and rs > best_rs:
            best_a, best_rs = float(a), rs
    z = math.cos(best_a) * h + math.sin(best_a) * t

    def mse(log_s: float) -> float:
        return float(np.mean((np.exp(-math.exp(log_s) * z) - r) ** 2))

    res = optimize.minimize_scalar(mse, bounds=(-8.0, 5.0), method="bounded")
    s = math.exp(res.x)
    return Calibration(s * math.cos(best_a), s * math.sin(best_a), best_rs, float(res.fun))


def free_normalisers(entropy, tv) -> tuple[float, float]:
    """Dataset-pooled empirical maxima used by rho_free (no outcome labels involved)."""
    h = float(np.max(np.asarray(entropy)))
    t = float(np.max(np.asarray(tv)))
    return max(h, 1e-8), max(t, 1e-8)


def expected_calibration_error(proxy, reference, n_bins: int = 10) -> float:
    """ECE of the proxy against the reference over proxy-decile bins (App. B.1 calibration audit)."""
    p = np.asarray(proxy, dtype=np.float64).ravel()
    r = np.asarray(reference, dtype=np.float64).ravel()
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - r[m].mean())
    return float(ece)

"""Statistics for the result tables.

* SWE-bench: 95% CIs from 10,000 instance-level bootstrap resamples stratified by seed (instances are
  resampled independently within each seed; the statistic is the mean over seeds of the resolve rate).
  Paired tests reuse the same resampled instances for both methods.
* Math: paired bootstrap over problems and seeds (both resampled with replacement) of the avg@k difference.
* Holm-Bonferroni correction across the comparisons of one table.
* Gap recovery: (method - student) / (teacher - student) x 100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass
class Estimate:
    mean: float
    lo: float
    hi: float


def resolve_rate_ci(resolved: np.ndarray, n_resamples: int = 10_000, confidence: float = 0.95,
                    seed: int = 0) -> Estimate:
    """resolved: [seeds, instances] of 0/1. Mean over seeds of the per-seed resolve rate (percent)."""
    r = np.asarray(resolved, dtype=np.float64)
    rng = np.random.default_rng(seed)
    s, n = r.shape
    idx = rng.integers(0, n, size=(n_resamples, s, n))
    boot = np.take_along_axis(np.broadcast_to(r, (n_resamples, s, n)), idx, axis=2).mean(axis=(1, 2))
    a = (1 - confidence) / 2
    return Estimate(100 * r.mean(), 100 * np.quantile(boot, a), 100 * np.quantile(boot, 1 - a))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Mean difference (a - b, percent points) and two-sided p-value; a, b: [seeds, instances] of 0/1."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    s, n = a.shape
    idx = rng.integers(0, n, size=(n_resamples, s, n))
    d = (a - b)
    boot = np.take_along_axis(np.broadcast_to(d, (n_resamples, s, n)), idx, axis=2).mean(axis=(1, 2))
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return 100 * d.mean(), float(min(p, 1.0))


def paired_bootstrap_problems_seeds(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000,
                                    seed: int = 0) -> tuple[float, float]:
    """Math: a, b [seeds, problems, samples] of 0/1 (avg@k per problem); resample problems and seeds."""
    a = np.asarray(a, dtype=np.float64).mean(-1)
    b = np.asarray(b, dtype=np.float64).mean(-1)
    d = a - b                                     # [seeds, problems]
    rng = np.random.default_rng(seed)
    s, n = d.shape
    si = rng.integers(0, s, size=(n_resamples, s))
    pi = rng.integers(0, n, size=(n_resamples, n))
    boot = d[si[:, :, None], pi[:, None, :]].mean(axis=(1, 2))
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return 100 * d.mean(), float(min(p, 1.0))


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm-adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def gap_recovery(method: float, student: float, teacher: float) -> float:
    return 100.0 * (method - student) / (teacher - student)


def spearman(x, y) -> float:
    return float(sps.spearmanr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)).statistic)


def weighted_mean(values, weights) -> float:
    v, w = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    return float((v * w).sum() / w.sum())


def avg_at_k(correct: np.ndarray) -> float:
    """correct: [problems, k] of 0/1 -> avg@k in percent."""
    return 100.0 * float(np.asarray(correct, dtype=np.float64).mean())

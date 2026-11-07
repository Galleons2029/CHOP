"""SNR constants (App. snr_constants) and the SNR bound of Proposition 1.

sigma_R:    std over v ~ p_t of |g_t(v) . u| * |R_perp(v)|, with R_perp(v) = R_T(v) - alpha_t R*(v)
            and u the unit aligned direction
sigma_star: std over v ~ p_t of |g_t(v) . u| * |alpha_t R*(v)|
c_S:        coefficient of variation of SW segment discrepancies (median of means over projection groups)

The paper does not define how the vector-valued products are reduced to scalars. Projecting onto the
aligned direction u is our documented choice.
"""

from __future__ import annotations

import numpy as np


def snr_bound(alpha: float, sigma_star: float, sigma_r: float) -> float:
    """sup SNR <= alpha sigma* / sqrt(sigma_R^2 + (1 - alpha)^2 sigma*^2)."""
    return alpha * sigma_star / np.sqrt(sigma_r ** 2 + (1 - alpha) ** 2 * sigma_star ** 2)


def sigma_estimates(proj: np.ndarray, r_t: np.ndarray, r_star: np.ndarray, alpha: float) -> tuple[float, float]:
    """proj: |g(v).u| per sampled token; r_t, r_star: rewards of the same tokens."""
    proj, r_t, r_star = (np.asarray(x, dtype=float) for x in (proj, r_t, r_star))
    return float(np.std(proj * np.abs(r_t - alpha * r_star))), float(np.std(proj * np.abs(alpha * r_star)))


def c_s(discrepancies: np.ndarray, groups: int = 8) -> float:
    """Median-of-means coefficient of variation over per-projection SW discrepancies [states, projections]."""
    d = np.asarray(discrepancies, dtype=float)
    parts = np.array_split(np.arange(d.shape[1]), groups)
    means = np.stack([d[:, p].mean(1) for p in parts if len(p)])
    m = np.median(means, 0)
    return float(np.std(m) / max(np.mean(m), 1e-12))

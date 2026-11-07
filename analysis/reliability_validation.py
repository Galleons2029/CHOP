"""Reliability proxy validation (App. B): rho vs rho_ref, calibration, failure regimes, depth curves.

rho_ref(s_t) is the empirical success rate of M teacher self-rollouts continued from s_t.
F1 (confidently wrong teacher): rho > 0.7 and rho_ref < 0.3.
F2 (reliable teacher, drifted student): rho_ref > 0.7 and rho in the middle tercile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chop.reliability import expected_calibration_error, spearman


def validate(rho, rho_ref) -> dict:
    r, ref = np.asarray(rho, dtype=float), np.asarray(rho_ref, dtype=float)
    lo, hi = np.quantile(r, [1 / 3, 2 / 3])
    return {"spearman": spearman(r, ref), "ece": expected_calibration_error(r, ref), "n": int(len(r)),
            "F1_confidently_wrong": float(((r > 0.7) & (ref < 0.3)).mean()),
            "F2_drifted_student": float(((ref > 0.7) & (r > lo) & (r <= hi)).mean())}


def depth_curves(rho_by_step: dict[int, list[tuple[int, float]]], bin_tokens: int = 1000, max_k: int = 30) -> dict:
    """Median rho per 1K-token depth bin for each training checkpoint (Fig. 2a input)."""
    depth_k = list(range(1, max_k + 1))
    out = {"depth_k": depth_k, "median_rho": {}}
    for step, pairs in rho_by_step.items():
        pos = np.array([p for p, _ in pairs])
        val = np.array([v for _, v in pairs])
        curve = []
        for k in depth_k:
            m = (pos >= (k - 1) * bin_tokens) & (pos < k * bin_tokens)
            curve.append(float(np.median(val[m])) if m.any() else None)
        out["median_rho"][str(step)] = curve
    return out


def write_depth(path: str | Path, curves: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(curves))

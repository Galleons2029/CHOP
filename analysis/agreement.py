"""Annotation export and agreement statistics for the criticality validation (App. annotation_protocol)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_states(states: list[dict], kappa: np.ndarray, path: str | Path, top_fraction: float = 0.1,
                  context_chars: int = 2000, seed: int = 0, n_controls: int | None = None) -> int:
    """Write the top-kappa decile plus an equal number of random controls, shuffled, labels hidden."""
    kappa = np.asarray(kappa)
    cut = np.quantile(kappa, 1 - top_fraction)
    top = [i for i in range(len(states)) if kappa[i] >= cut]
    rng = np.random.default_rng(seed)
    rest = [i for i in range(len(states)) if kappa[i] < cut]
    ctrl = list(rng.choice(rest, size=min(n_controls or len(top), len(rest)), replace=False)) if rest else []
    items = [(i, "top") for i in top] + [(int(i), "control") for i in ctrl]
    rng.shuffle(items)
    with open(path, "w") as f:
        for i, bucket in items:
            s = states[i]
            f.write(json.dumps({"item": int(i), "context": s["context"][-context_chars:], "next": s.get("next", ""),
                                "_bucket": bucket}) + "\n")
    return len(items)


def fleiss_kappa(ratings: np.ndarray) -> float:
    """ratings: [items, categories] counts of raters choosing each category."""
    r = np.asarray(ratings, dtype=float)
    n = r.sum(1)[0]
    p_j = r.sum(0) / r.sum()
    p_i = ((r * r).sum(1) - n) / (n * (n - 1))
    pe = (p_j ** 2).sum()
    return float((p_i.mean() - pe) / (1 - pe))


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0

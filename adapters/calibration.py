"""Adapter calibration diagnostics (App. C.5): assumption (A2) with delta_B = 0.13 in the paper.

* symmetric KL between the LoRA PI teacher and a full fine-tuning reference on held-out states;
* composition consistency: additive multi-LoRA composition vs a jointly trained adapter;
* per-source ECE of the next-action argmax.
All functions take per-state logits so they apply to any pair of models / compositions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from chop.pi.composition import compose_logits


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    la, lb = F.log_softmax(logits_a.float(), -1), F.log_softmax(logits_b.float(), -1)
    return 0.5 * ((la.exp() * (la - lb)).sum(-1) + (lb.exp() * (lb - la)).sum(-1))


def composition_error(base: torch.Tensor, adapters: list[torch.Tensor], joint: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between the additive composition and a jointly trained reference."""
    return symmetric_kl(compose_logits(base, adapters), joint)


def argmax_ece(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """Expected calibration error of max-probability confidence for next-action argmax correctness."""
    p = F.softmax(logits.float(), -1)
    conf, pred = p.max(-1)
    correct = (pred == targets).float()
    c, a = conf.numpy(), correct.numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (c > lo) & (c <= hi)
        if m.any():
            ece += m.mean() * abs(c[m].mean() - a[m].mean())
    return float(ece)

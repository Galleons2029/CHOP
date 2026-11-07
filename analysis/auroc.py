"""Sequence-AUROC: does the teacher rank successful student rollouts above failed ones?"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def teacher_sequence_score(teacher, prompt: list[int], response: list[int]) -> float:
    """Length-normalised teacher log-likelihood of a student response."""
    ids = torch.tensor([prompt + response])
    logp = F.log_softmax(teacher(input_ids=ids).logits[0, len(prompt) - 1:-1].float(), -1)
    return float(logp.gather(-1, torch.tensor(response).unsqueeze(-1)).mean())


def auroc(pos_scores, neg_scores) -> float:
    """Mann-Whitney AUROC with ties counted as 1/2."""
    pos, neg = np.asarray(pos_scores, dtype=float), np.asarray(neg_scores, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))

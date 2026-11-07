"""Alignment coefficient alpha_t (Eq. alpha_t) and its trajectory diagnostic alpha_diag (App. alpha_defs).

    alpha_t = cos( E_v[g_t(v) R_T(v|s_t)] , E_v[g_t(v) R*(v|s_t)] ),  v ~ p_t

R_T(v) = log q_t(v) - log p_t(v), and the first expectation is computed exactly over the vocabulary.
R*(v) is the sequence-level return after v, estimated from N = 64 teacher rollouts from s_t: rollouts are
grouped by their first token, and R_hat*(v) is the mean outcome of the group. Returns are centred by the
overall mean, and tokens that no rollout started with get the centred value 0. This is a documented
choice; the paper does not state how R* is extended to unsampled tokens.

alpha_diag averages alpha_t over 12 uniformly spaced reference states of a trajectory.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch

from analysis.gradients import cosine, flat_grad, last_logprobs, param_subset

RolloutFn = Callable[[list[int], int], list[tuple[int, float]]]   # (context, n) -> [(first token, outcome)]


def centred_returns(rollouts: list[tuple[int, float]], vocab: int) -> torch.Tensor:
    groups = defaultdict(list)
    for tok, r in rollouts:
        groups[int(tok)].append(float(r))
    base = float(np.mean([r for _, r in rollouts])) if rollouts else 0.0
    c = torch.zeros(vocab)
    for tok, rs in groups.items():
        c[tok] = float(np.mean(rs)) - base
    return c


def alpha_t(student, teacher, context: list[int], rollouts: list[tuple[int, float]],
            subset: str = "last_block", teacher_logq: torch.Tensor | None = None) -> float:
    """alpha_t at ``context``. ``teacher_logq`` overrides the teacher distribution: the gradient-alignment
    analysis (App. grad_alignment) passes the PI-conditioned composite, or the unconditional teacher."""
    params = param_subset(student, subset)
    logp = last_logprobs(student, context)
    with torch.no_grad():
        logq = teacher_logq if teacher_logq is not None else last_logprobs(teacher, context)
    p = logp.exp().detach()
    r_t = (logq - logp).detach()
    g_t = flat_grad((p * r_t * logp).sum(), params)
    c = centred_returns(rollouts, logp.shape[-1]).to(logp.device)
    g_s = flat_grad((p * c * logp).sum(), params)
    return cosine(g_t, g_s)


def reference_positions(length: int, n: int = 12) -> list[int]:
    if length <= 0:
        return []
    return sorted({int(round(x)) for x in np.linspace(0, length - 1, min(n, length))})


def alpha_diag(student, teacher, prompt: list[int], response: list[int], rollout: RolloutFn, n_rollouts: int = 64,
               n_states: int = 12, subset: str = "last_block") -> tuple[float, list[float]]:
    vals = []
    for pos in reference_positions(len(response), n_states):
        ctx = prompt + response[:pos]
        a = alpha_t(student, teacher, ctx, rollout(ctx, n_rollouts), subset)
        if not np.isnan(a):
            vals.append(a)
    return (float(np.mean(vals)) if vals else float("nan")), vals

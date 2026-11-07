"""Per-state divergences between student p_t and teacher q_t (App. A of the paper).

All functions take logits with an arbitrary leading shape ``[..., V]`` and return per-state
values of shape ``[...]`` in float32.  Student logits may carry gradients; teacher logits are
treated as constants.  For long trajectories call them on sequence chunks (see
:func:`chunked`) so that no ``[B, T, V]`` float32 tensor is ever materialised.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F


def _logp(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=-1)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """H(q) = -sum_v q(v) log q(v)."""
    logq = _logp(logits)
    return -(logq.exp() * logq).sum(-1)


def total_variation(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """TV(a, b) = 1/2 sum_v |a(v) - b(v)|."""
    return 0.5 * (_logp(logits_a).exp() - _logp(logits_b).exp()).abs().sum(-1)


def reverse_kl_full(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary KL(p || q) (Eq. opd_full); differentiable in the student."""
    logp = _logp(student_logits)
    logq = _logp(teacher_logits.detach())
    return (logp.exp() * (logp - logq)).sum(-1)


def reverse_kl_topk(student_logits: torch.Tensor, teacher_logits: torch.Tensor, k: int = 64,
                    support: str = "student") -> torch.Tensor:
    """Truncated reverse KL on a top-k support with both distributions renormalised.

    ``support='student'``: S_t = TopK(p_t, k) (Eq. opd_topk, CHOP's token branch).
    ``support='teacher'``: S_t = TopK(q_t, k) (local support matching of the Top-K LSM baseline).
    The support is selected without gradient (stop-gradient support, Sec. 4.5).
    """
    if support not in ("student", "teacher"):
        raise ValueError(f"unknown top-k support {support!r}")
    k = min(k, student_logits.shape[-1])
    ref = student_logits if support == "student" else teacher_logits
    idx = ref.detach().topk(k, dim=-1).indices
    s = student_logits.float().gather(-1, idx)
    t = teacher_logits.detach().float().gather(-1, idx)
    logp = F.log_softmax(s, dim=-1)
    logq = F.log_softmax(t, dim=-1)
    return (logp.exp() * (logp - logq)).sum(-1)


def generalized_jsd(student_logits: torch.Tensor, teacher_logits: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """GKD divergence (Agarwal et al., 2024): beta=0 forward KL, beta=1 reverse KL, else JSD(beta).

    Follows the TRL convention: M = (1 - beta) p + beta q and
    JSD(beta) = beta KL(q || M) + (1 - beta) KL(p || M).
    """
    logp = _logp(student_logits)
    logq = _logp(teacher_logits.detach())
    if beta <= 0.0:
        return (logq.exp() * (logq - logp)).sum(-1)
    if beta >= 1.0:
        return (logp.exp() * (logp - logq)).sum(-1)
    m = torch.logaddexp(logp + math.log(1 - beta), logq + math.log(beta))
    kl_q = (logq.exp() * (logq - m)).sum(-1)
    kl_p = (logp.exp() * (logp - m)).sum(-1)
    return beta * kl_q + (1 - beta) * kl_p


def token_logprobs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """log softmax(logits)[token] for every position."""
    return _logp(logits).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def sampled_reverse_kl_surrogate(student_logp: torch.Tensor, teacher_logp: torch.Tensor) -> torch.Tensor:
    """Sampled-token OPD (Eq. opd_sample) as a score-function surrogate.

    With y_t ~ p_t, l_t = log p_t(y_t) - log q_t(y_t) is an unbiased estimate of KL(p_t||q_t) and
    E[l_t * grad log p_t(y_t)] is its gradient, so the surrogate sg(l_t) * log p_t(y_t) has the
    right gradient.  The returned value is the surrogate; use ``(student_logp - teacher_logp)``
    for reporting the KL estimate itself.
    """
    adv = (student_logp - teacher_logp).detach()
    return adv * student_logp


def reliability_inputs(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(H(q_t), TV(q_t, p_t)) from logits already computed for OPD: no extra teacher pass."""
    with torch.no_grad():
        return entropy(teacher_logits), total_variation(teacher_logits, student_logits)


def topk_overlap(student_logits: torch.Tensor, teacher_logits: torch.Tensor, k: int = 64) -> torch.Tensor:
    """|TopK(p,k) ∩ TopK(q,k)| / k (App. A.4 diagnostic)."""
    with torch.no_grad():
        a = student_logits.topk(k, dim=-1).indices
        b = teacher_logits.topk(k, dim=-1).indices
        hits = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1)
        return hits / k


def chunked(fn: Callable[..., torch.Tensor], *tensors: torch.Tensor, chunk: int = 1024, dim: int = 0) -> torch.Tensor:
    """Apply ``fn`` to aligned slices of ``tensors`` along ``dim`` and concatenate the results."""
    n = tensors[0].shape[dim]
    outs = [fn(*(t.narrow(dim, i, min(chunk, n - i)) for t in tensors)) for i in range(0, n, chunk)]
    return torch.cat(outs, dim=dim)

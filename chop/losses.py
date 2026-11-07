"""Loss assembly: CHOP (Eq. chop_lite) and the baselines of Tables 1-3.

Token-level baselines are presets of the same router (``configs/method``):
  standard OPD   = rho_gate none, kappa_gate none, token estimator 'sampled'
  GKD            = rho_gate none, kappa_gate none, token estimator 'gkd'
  Top-K LSM      = rho_gate none, kappa_gate none, token estimator 'topk', support 'teacher', k=32
  OPSD-style     = rho_gate none, kappa_gate always (all-PI teacher), token estimator 'sampled'
  CHOP token-only= rho_gate none, kappa gate on
REOPOLD and GRPO have their own objectives below.

All losses are normalised by the number of eligible states in the batch (token mean).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from chop import divergences as D
from chop.config import MethodConfig, TokenLossConfig
from chop.router import Route
from chop.segment import advantages, segment_distance, segment_surrogate


@dataclass
class StateBatch:
    """Eligible states of a batch in stream order."""
    student_logits: torch.Tensor           # [N, V], carries gradient
    tokens: torch.Tensor                   # [N] realised tokens y_t
    teacher_logits: torch.Tensor           # [N, V] unconditional teacher
    teacher_pi_logits: torch.Tensor | None = None  # [N, V] composite PI-conditional teacher
    reward: torch.Tensor | None = None     # [N] outcome reward R(tau) of the state's trajectory
    special: torch.Tensor | None = None    # [N] bool, special tokens (Top-K LSM masking)


@dataclass
class SegmentTerms:
    """Segment-branch quantities for n anchor states.

    Either pass the stop-gradient embeddings (the distance and advantages are computed here) or
    precomputed advantages from :func:`segment_advantages` (the distributed trainer computes them
    in a no-grad pass so they can be normalised over the whole batch before micro-batching).
    """
    logp: torch.Tensor                       # [n, K', L_a] student log-probs of its segments (grad)
    mask: torch.Tensor                       # [n, K', L_a]
    weights: torch.Tensor                    # [n] routed states represented by each anchor
    teacher_emb: torch.Tensor | None = None  # [n, K, d]
    student_emb: torch.Tensor | None = None  # [n, K', d]
    adv: torch.Tensor | None = None          # [n, K']


@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


def token_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tokens: torch.Tensor,
                     cfg: TokenLossConfig) -> torch.Tensor:
    """Per-state token-branch loss (a surrogate whose gradient is the estimator's gradient)."""
    if cfg.estimator == "sampled":
        lp = D.token_logprobs(student_logits, tokens)
        lq = D.token_logprobs(teacher_logits.detach(), tokens)
        return D.sampled_reverse_kl_surrogate(lp, lq)
    if cfg.estimator == "full":
        return D.reverse_kl_full(student_logits, teacher_logits)
    if cfg.estimator == "topk":
        return D.reverse_kl_topk(student_logits, teacher_logits, cfg.topk, cfg.support)
    if cfg.estimator == "gkd":
        return D.generalized_jsd(student_logits, teacher_logits, cfg.gkd_beta)
    raise ValueError(f"unknown token estimator {cfg.estimator!r}")


def segment_advantages(teacher_emb: torch.Tensor, student_emb: torch.Tensor, cfg: MethodConfig,
                       generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(segment distance [n], REINFORCE advantages [n, K']) from stop-gradient embeddings."""
    s = cfg.segment
    dist, costs = segment_distance(teacher_emb.detach(), student_emb.detach(), s.distance, s.num_projections,
                                   generator)
    return dist, advantages(costs, s.estimator)


def _mean(x: torch.Tensor) -> float:
    return float(x.detach().float().mean()) if x.numel() else float("nan")


def chop_loss(states: StateBatch, route: Route, seg: SegmentTerms | None, cfg: MethodConfig,
              generator: torch.Generator | None = None, normalizer: int | None = None) -> LossOutput:
    """Eq. (chop_lite), summed over eligible states and divided by their number.

    trusted   (rho > tau):  KL(pi_theta || pi_T^(g P))                         [Cells A/B]
    untrusted (rho <= tau): SW_2^2(e#mu^(g P), e#nu) - beta R(tau) log pi(y_t)  [Cells C/D]
    Outside a configured rho stratum (rho-decile experiment) states use standard sampled OPD.
    ``normalizer`` (default: the number of states passed) lets micro-batches share the batch-level
    denominator.
    """
    n = states.tokens.numel()
    if route.cells.numel() != n:
        raise ValueError("route and state batch have different lengths")
    if states.teacher_pi_logits is not None:
        target = torch.where(route.pi_gate.to(states.teacher_logits.device).unsqueeze(-1),
                             states.teacher_pi_logits, states.teacher_logits)
    else:
        target = states.teacher_logits
    dev = states.student_logits.device
    trusted = (route.trusted & route.in_stratum).to(dev)
    untrusted = (~route.trusted & route.in_stratum).to(dev)
    outside = (~route.in_stratum).to(dev)
    keep = torch.ones(n, dtype=torch.bool, device=dev)
    if cfg.token.mask_special_tokens and states.special is not None:
        keep = ~states.special.to(dev)

    zero = states.student_logits.sum() * 0.0
    metrics: dict[str, float] = dict(route.fractions())

    tok_mask = trusted & keep
    tok = token_divergence(states.student_logits[tok_mask], target[tok_mask], states.tokens[tok_mask], cfg.token)
    loss_tok = tok.sum() if tok.numel() else zero
    metrics["token_loss"] = _mean(tok)

    loss_out = zero
    if cfg.beta_outcome and states.reward is not None and bool(untrusted.any()):
        lp = D.token_logprobs(states.student_logits[untrusted], states.tokens[untrusted])
        loss_out = -cfg.beta_outcome * (states.reward.to(dev)[untrusted].float() * lp).sum()

    loss_fb = zero
    if bool(outside.any()):
        lp = D.token_logprobs(states.student_logits[outside], states.tokens[outside])
        lq = D.token_logprobs(states.teacher_logits[outside], states.tokens[outside])
        loss_fb = D.sampled_reverse_kl_surrogate(lp, lq).sum()

    loss_seg = zero
    if seg is not None and seg.weights.numel():
        adv = seg.adv
        if adv is None:
            dist, adv = segment_advantages(seg.teacher_emb, seg.student_emb, cfg, generator)
            metrics["segment_distance"] = _mean(dist)
        loss_seg = cfg.segment.coef * segment_surrogate(seg.logp, seg.mask, adv.to(seg.logp.device),
                                                        seg.weights.to(seg.logp.device))
        metrics["num_anchors"] = float(seg.weights.numel())

    total = (loss_tok + loss_out + loss_fb + loss_seg) / max(normalizer or n, 1)
    metrics["loss"] = float(total.detach())
    return LossOutput(total, metrics)


def reopold_loss(states: StateBatch, cfg: MethodConfig, step: int) -> LossOutput:
    """Relaxed OPD: R = log q - log p (stop-gradient), clipped below at log(l/(1-l)).

    Phase I (step < switch): keep tokens with R >= log(l/(1-l)) (reward-based filtering).
    Phase II: keep the top-beta fraction of tokens by student entropy (entropy-guided sampling).
    Objective: token mean over kept tokens of sg(R_hat) * log p (maximised).
    """
    lp = D.token_logprobs(states.student_logits, states.tokens)
    lq = D.token_logprobs(states.teacher_logits.detach(), states.tokens)
    r = (lq - lp).detach()
    floor = math.log(cfg.reopold_lambda / (1 - cfg.reopold_lambda))
    r_hat = r.clamp_min(floor)
    if step < cfg.reopold_switch_step:
        m = r >= floor
    else:
        with torch.no_grad():
            h = D.entropy(states.student_logits)
            tau = torch.quantile(h.float(), 1 - cfg.reopold_entropy_top)
        m = h >= tau
    denom = m.sum().clamp_min(1)
    loss = -(r_hat * lp * m).sum() / denom
    return LossOutput(loss, {"loss": float(loss.detach()), "kept": float(m.float().mean()),
                             "reward_mean": _mean(r)})


def grpo_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-normalised outcome advantages (Shao et al., 2024) for the RLVR baseline."""
    adv = torch.zeros_like(rewards, dtype=torch.float32)
    for g in group_ids.unique():
        m = group_ids == g
        r = rewards[m].float()
        adv[m] = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return adv


def grpo_loss(student_logits: torch.Tensor, tokens: torch.Tensor, token_adv: torch.Tensor) -> LossOutput:
    """On-policy policy gradient with per-token copies of the sequence advantage (no KL term)."""
    lp = D.token_logprobs(student_logits, tokens)
    loss = -(token_adv.detach() * lp).mean()
    return LossOutput(loss, {"loss": float(loss.detach())})

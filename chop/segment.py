"""Segment-matching branch (Sec. 4.3; App. F.2-F.4).

At an untrusted state s_t the student is matched to K teacher anchor segments with K' student
segments of length L_a, embedded by the current student encoder (mean-pooled, stop-gradient),
under the sliced-Wasserstein distance

    SW_2^2(e#mu, e#nu) = 1/L sum_l W_2^2(Proj_l{e(a_k)}, Proj_l{e(u_k')}).

Because e is stop-gradient the transport cost is constant in theta; the gradient is estimated
with a REINFORCE surrogate over the student segments (leave-one-out baseline + per-batch
normalisation by default).

Student segments are scored by one *tree-packed* forward: branch tokens are appended after the
main trajectory and attend to the main prefix before their anchor plus their own earlier
tokens, with position ids continuing from the anchor.  Prefix activations are therefore stored
once and gradients flow through the shared prefix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# ----------------------------------------------------------------------------- anchor selection


def select_anchor_windows(
    segment_mask: torch.Tensor,
    window: int = 64,
    max_windows: int | None = 32,
    generator: torch.Generator | None = None,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparsify routed states into at most one anchor per L_a window.

    ``segment_mask[i]`` marks routed (untrusted) states in one trajectory.  Windows are aligned
    at ``offset`` (the first response position).  Each window with a routed state contributes one
    anchor (its first routed state) weighted by the number of routed states in the window, so the
    sum of per-state segment losses is preserved when neighbouring states share one estimate.
    If more than ``max_windows`` windows qualify, a uniform subset is kept and weights are scaled by
    n / max_windows (Horvitz-Thompson), keeping the estimate unbiased.
    Returns (anchor positions, weights), sorted by position.
    """
    idx = torch.nonzero(segment_mask.flatten(), as_tuple=False).flatten()
    if idx.numel() == 0:
        return idx, torch.empty(0)
    win = (idx - offset).div(window, rounding_mode="floor")
    first = torch.ones_like(win, dtype=torch.bool)
    first[1:] = win[1:] != win[:-1]
    anchors = idx[first]
    _, counts = torch.unique_consecutive(win, return_counts=True)
    weights = counts.float()
    n = anchors.numel()
    if max_windows is not None and n > max_windows:
        keep = torch.randperm(n, generator=generator)[:max_windows].sort().values
        anchors, weights = anchors[keep], weights[keep] * (n / max_windows)
    return anchors, weights


# ------------------------------------------------------------------------------- tree packing


@dataclass
class TreeLayout:
    input_ids: torch.Tensor      # [S] main trajectory followed by all branch tokens
    position_ids: torch.Tensor   # [S]
    main_len: int
    anchor: torch.Tensor         # [nb] main index a: branch b continues the context main[:a]
    start: torch.Tensor          # [nb] packed index of the first token of branch b
    length: torch.Tensor         # [nb]
    token_branch: torch.Tensor   # [S] branch id of every packed token, -1 for main tokens

    @property
    def num_branches(self) -> int:
        return int(self.anchor.numel())

    def allowed(self) -> torch.Tensor:
        """Dense boolean attention pattern [S, S] (True = may attend)."""
        s = self.input_ids.numel()
        q = torch.arange(s).unsqueeze(1)
        kv = torch.arange(s).unsqueeze(0)
        return self.mask_mod(0, 0, q, kv)

    def additive_mask(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """[1, 1, S, S] additive mask accepted by HF eager/sdpa attention."""
        m = torch.zeros(self.input_ids.numel(), self.input_ids.numel(), dtype=dtype)
        m.masked_fill_(~self.allowed(), torch.finfo(dtype).min)
        return m[None, None]

    def mask_mod(self, b, h, q_idx, kv_idx):
        """Tree attention pattern (True = may attend); usable on dense index grids."""
        return self.flex_mask_mod()(b, h, q_idx, kv_idx)

    def flex_mask_mod(self, device=None):
        """A plain mask_mod closure for ``torch.nn.attention.flex_attention.create_block_mask``."""
        tb = self.token_branch.to(device) if device is not None else self.token_branch
        anchor_tok = self._anchor_of_token().to(tb.device)

        def mask_mod(b, h, q_idx, kv_idx):
            qb, kb = tb[q_idx], tb[kv_idx]
            q_main, kv_main = qb < 0, kb < 0
            causal = kv_idx <= q_idx
            main_to_main = q_main & kv_main & causal
            branch_to_prefix = (~q_main) & kv_main & (kv_idx < anchor_tok[q_idx])
            branch_to_self = (~q_main) & (qb == kb) & causal
            return main_to_main | branch_to_prefix | branch_to_self

        return mask_mod

    def block_mask(self, device="cpu", block_size: int = 128):
        """FlexAttention BlockMask for HF models loaded with attn_implementation='flex_attention'."""
        from torch.nn.attention.flex_attention import create_block_mask

        s = self.input_ids.numel()
        return create_block_mask(self.flex_mask_mod(device), B=None, H=None, Q_LEN=s, KV_LEN=s, device=device,
                                 BLOCK_SIZE=block_size)

    def _anchor_of_token(self) -> torch.Tensor:
        if not hasattr(self, "_anchor_tok"):
            a = torch.zeros_like(self.token_branch)
            br = self.token_branch >= 0
            a[br] = self.anchor[self.token_branch[br]]
            self._anchor_tok = a
        return self._anchor_tok

    def branch_token_logprobs(self, logits: torch.Tensor, pad: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token log-probs of each branch under the packed forward.

        Token 0 of branch b is predicted by the main logits at index a-1 (the same distribution
        p_t as the realised token); token j >= 1 by the branch logits at start + j - 1.
        Returns (logp [nb, Lmax], mask [nb, Lmax]).
        """
        logp_all = torch.log_softmax(logits.float(), dim=-1)
        lmax = int(self.length.max()) if self.num_branches else 0
        out = logits.new_zeros(self.num_branches, lmax, dtype=torch.float32)
        mask = torch.zeros(self.num_branches, lmax, dtype=torch.bool)
        for b in range(self.num_branches):
            a, s, n = int(self.anchor[b]), int(self.start[b]), int(self.length[b])
            src = torch.cat([torch.tensor([a - 1]), torch.arange(s, s + n - 1)])
            tok = self.input_ids[s:s + n]
            out[b, :n] = logp_all[src, tok]
            mask[b, :n] = True
        return out, mask

    def branch_mean_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Mean-pooled hidden states of each branch's own tokens, [nb, d]."""
        spans = zip(self.start.tolist(), self.length.tolist(), strict=True)
        return torch.stack([hidden[s:s + n].mean(0) for s, n in spans])


def build_tree(main_ids: torch.Tensor, branches: list[tuple[int, torch.Tensor]]) -> TreeLayout:
    """Pack ``main_ids`` [T] and branches [(anchor a, tokens)] into one sequence."""
    main_ids = main_ids.flatten()
    t = main_ids.numel()
    ids, pos, tb = [main_ids], [torch.arange(t)], [torch.full((t,), -1, dtype=torch.long)]
    anchor, start, length = [], [], []
    cursor = t
    for b, (a, toks) in enumerate(branches):
        toks = toks.flatten()
        if not 1 <= a <= t:
            raise ValueError(f"anchor {a} outside main sequence of length {t}")
        n = toks.numel()
        ids.append(toks)
        pos.append(a + torch.arange(n))
        tb.append(torch.full((n,), b, dtype=torch.long))
        anchor.append(a)
        start.append(cursor)
        length.append(n)
        cursor += n
    return TreeLayout(
        input_ids=torch.cat(ids),
        position_ids=torch.cat(pos),
        main_len=t,
        anchor=torch.tensor(anchor, dtype=torch.long),
        start=torch.tensor(start, dtype=torch.long),
        length=torch.tensor(length, dtype=torch.long),
        token_branch=torch.cat(tb),
    )


def realised_segment(main_ids: torch.Tensor, logits: torch.Tensor, hidden: torch.Tensor, anchor: int,
                     length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probs [n] and mean hidden state [d] of the realised continuation main[a : a+L]."""
    end = min(anchor + length, main_ids.numel())
    logp = torch.log_softmax(logits[anchor - 1:end - 1].float(), dim=-1)
    tok = main_ids[anchor:end]
    return logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1), hidden[anchor:end].mean(0)


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid positions: hidden [..., L, d], mask [..., L] -> [..., d]."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(-2) / m.sum(-2).clamp_min(1.0)


# ------------------------------------------------------------------------------- distances


def random_directions(num_states: int, num_projections: int, dim: int,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniform unit directions [n, L, d] (fresh per state)."""
    d = torch.randn(num_states, num_projections, dim, generator=generator)
    return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def sliced_w2(teacher: torch.Tensor, student: torch.Tensor,
              directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sliced W_2^2 between empirical clouds and per-student-sample transport costs.

    teacher [n, K, d], student [n, K', d], directions [n, L, d].
    Per projection the student sample of rank r is matched to the teacher order statistic of index
    floor((r + 1/2) K / K') (the exact monotone coupling when K = K').  The per-sample cost
    c_k' = 1/L sum_l (proj_l(u_k') - proj_l(a_{sigma_l(k')}))^2 averages to SW_2^2.
    Returns (sw2 [n], costs [n, K']).
    """
    k, kp = teacher.shape[1], student.shape[1]
    px = torch.einsum("nkd,nld->nkl", teacher.float(), directions.float())
    py = torch.einsum("nkd,nld->nkl", student.float(), directions.float())
    sx = px.sort(dim=1).values
    ranks = py.argsort(dim=1).argsort(dim=1)
    idx = ((ranks.float() + 0.5) * k / kp).floor().long().clamp(max=k - 1)
    matched = sx.gather(1, idx)
    costs = (py - matched).pow(2).mean(-1)
    return costs.mean(-1), costs


def _sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.unsqueeze(-2) - b.unsqueeze(-3)).pow(2).sum(-1)


def mmd2_gaussian(teacher: torch.Tensor, student: torch.Tensor,
                  bandwidth: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Unbiased Gaussian-kernel MMD^2 with a per-state median-heuristic bandwidth.

    The per-student-sample cost is the score-function weight of d MMD^2 / d nu:
    c_k' = 2 mean_{j != k'} k(u_k', u_j) - 2 mean_i k(a_i, u_k').
    Returns (mmd2 [n], costs [n, K']).
    """
    x, y = teacher.float(), student.float()
    k, kp = x.shape[1], y.shape[1]
    dxx, dyy, dxy = _sqdist(x, x), _sqdist(y, y), _sqdist(x, y)
    if bandwidth is None:
        allp = torch.cat([x, y], dim=1)
        d = _sqdist(allp, allp)
        iu = torch.triu_indices(d.shape[-1], d.shape[-1], offset=1)
        sigma2 = d[:, iu[0], iu[1]].median(dim=-1).values.clamp_min(1e-8).view(-1, 1, 1)
    else:
        sigma2 = torch.full((x.shape[0], 1, 1), bandwidth ** 2)
    kxx, kyy, kxy = (torch.exp(-m / (2 * sigma2)) for m in (dxx, dyy, dxy))
    off_x = ~torch.eye(k, dtype=torch.bool)
    off_y = ~torch.eye(kp, dtype=torch.bool)
    mxx = kxx[:, off_x].mean(-1)
    myy = kyy[:, off_y].mean(-1)
    mxy = kxy.mean((-1, -2))
    mmd2 = mxx + myy - 2 * mxy
    kyy_off = kyy.masked_fill(~off_y, 0.0)
    costs = 2 * kyy_off.sum(-1) / max(kp - 1, 1) - 2 * kxy.mean(1)
    return mmd2, costs


# ----------------------------------------------------------------------- gradient estimator


def advantages(costs: torch.Tensor, estimator: str = "loo_norm", eps: float = 1e-6) -> torch.Tensor:
    """Centre/scale per-sample costs [n, K'] into REINFORCE weights (tab:sw_gradient).

    vanilla  : c
    mean     : c - mean_k c
    loo      : c - mean over the other K'-1 samples of the same state
    loo_norm : loo, divided by its standard deviation over the whole batch (default)
    """
    c = costs.detach().float()
    if estimator == "vanilla":
        return c
    if estimator == "mean":
        return c - c.mean(-1, keepdim=True)
    kp = c.shape[-1]
    if kp < 2:
        raise ValueError("leave-one-out baselines need K' >= 2")
    loo = c - (c.sum(-1, keepdim=True) - c) / (kp - 1)
    if estimator == "loo":
        return loo
    if estimator == "loo_norm":
        return loo / (loo.std(unbiased=False) + eps) if loo.numel() > 1 else loo
    raise ValueError(f"unknown estimator {estimator!r}")


def segment_surrogate(seg_logp: torch.Tensor, seg_mask: torch.Tensor, adv: torch.Tensor,
                      weights: torch.Tensor) -> torch.Tensor:
    """sum_n w_n * 1/K' sum_k' sg(A_nk') * sum_l log pi(u_l^{(k')} | ...).

    seg_logp/seg_mask [n, K', L_a]; adv [n, K']; weights [n].  Minimising it lowers the
    probability of student segments with above-baseline transport cost.
    """
    lp = (seg_logp * seg_mask).sum(-1)
    return (weights.unsqueeze(-1) * adv.detach() * lp).sum() / lp.shape[-1]


def segment_distance(teacher_emb: torch.Tensor, student_emb: torch.Tensor, distance: str = "sw",
                     num_projections: int = 32, generator: torch.Generator | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to SW_2^2 (default) or MMD^2; returns (distance [n], per-sample costs [n, K'])."""
    if distance == "sw":
        dirs = random_directions(teacher_emb.shape[0], num_projections, teacher_emb.shape[-1], generator)
        return sliced_w2(teacher_emb, student_emb, dirs.to(teacher_emb.device))
    if distance == "mmd":
        return mmd2_gaussian(teacher_emb, student_emb)
    raise ValueError(f"unknown distance {distance!r}")


def effective_rank(emb: torch.Tensor) -> float:
    """exp(entropy of the normalised covariance spectrum) of embeddings [N, d] (App. D.2)."""
    x = emb.detach().float()
    x = x - x.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh(x.T @ x / max(x.shape[0] - 1, 1)).clamp_min(0)
    p = ev / ev.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(math.exp(-(p * p.log()).sum()))

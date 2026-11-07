"""Per-rank CHOP computations used by the verl worker (and testable without Ray / verl).

The functions mirror phases B-E of :mod:`chop.reference_trainer` but scale to long trajectories:
* the student (FSDP-sharded under verl) is only called through its root forward, restricted to the
  logits rows that are needed; the frozen teacher is replicated and scored chunk-wise;
* teacher information is reduced to :class:`chop.engine.SparseTargets` at scoring time and cached
  on the worker until the update, so no vocabulary-sized tensor crosses process boundaries;
* segment branches are sampled with one shared prefix cache per chunk (:func:`chop.engine.tree_decode`)
  and scored with one tree-packed forward per chunk.

With ``lockstep=True`` every function issues a number of forward calls that depends only on the
configuration, never on the data, which keeps FSDP collectives aligned across ranks (dummy jobs
fill the gaps; see :mod:`chop_verl.lockstep`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from chop.config import MethodConfig
from chop.engine import (
    SparseTargets,
    StateStats,
    Variant,
    head_fn,
    root_forward,
    rows_fn,
    score_states,
    sparse_token_loss,
    tree_decode,
)
from chop.reliability import reliability
from chop.router import Route
from chop.segment import build_tree, mmd2_gaussian, random_directions, segment_surrogate, sliced_w2
from chop.truncation import Chunk, chunk_of


@dataclass
class TrajPayload:
    uid: str
    s_prompt: list[int]
    t_prompt: list[int]
    response: list[int]
    mask: list[int]                              # 1 = student-generated token (eligible state)
    reward: float
    pi_prefix: dict[str, list[int]]
    chunks: list[Chunk]
    # per-step sources (oracle next file): one prefix per chunk, computed from the edit log at the chunk start
    pi_prefix_chunks: dict[str, list[list[int]]] = field(default_factory=dict)

    @property
    def eligible(self) -> torch.Tensor:
        return torch.tensor(self.mask, dtype=torch.bool)

    def prefix(self, source: str, chunk_index: int) -> list[int]:
        per = self.pi_prefix_chunks.get(source)
        if per:
            return per[min(chunk_index, len(per) - 1)]
        return self.pi_prefix[source]


@dataclass
class AnchorSpec:
    pos: int                                     # response position of the anchor state
    gate: bool                                   # PI gate at the anchor
    weight: float                                # routed states represented


@dataclass
class AnchorSegments:
    spec: AnchorSpec
    chunk: Chunk
    student: list[list[int]] = field(default_factory=list)   # K'-1 sampled continuations
    teacher: list[list[int]] = field(default_factory=list)   # K anchors


class ChopModels:
    """Student + frozen teacher with per-source adapter activation.

    ``adapter_ctx(name)`` returns a context manager activating the PI adapter ``name`` on the teacher,
    or disabling adapters when ``name`` is None.  ``attention`` selects how tree masks are passed to
    the student: 'dense' (additive 4D mask; sdpa/eager) or 'flex' (FlexAttention BlockMask; GPU).
    """

    def __init__(self, student, teacher, sources: list[str], adapter_ctx: Callable | None = None,
                 attention: str = "dense", block_size: int = 128, pad_id: int = 0, eos_id: int | None = None,
                 lockstep: bool = False, logit_chunk: int = 1024):
        self.student, self.teacher, self.sources = student, teacher, list(sources)
        self.adapter_ctx = adapter_ctx or self._peft_ctx
        self.attention, self.block_size = attention, block_size
        self.pad_id, self.eos_id, self.lockstep, self.logit_chunk = pad_id, eos_id, lockstep, logit_chunk

    def _peft_ctx(self, name):
        if not hasattr(self.teacher, "peft_config"):
            if name is not None:
                raise ValueError("teacher has no PI adapters")
            return contextlib.nullcontext()
        if name is None:
            return self.teacher.disable_adapter()
        self.teacher.set_adapter(name)
        return contextlib.nullcontext()

    @property
    def device(self):
        return next(self.student.parameters()).device

    def tree_mask(self, layout):
        if self.attention == "flex":
            return layout.block_mask(self.device, self.block_size)
        return layout.additive_mask(torch.float32).to(self.device)

    @contextlib.contextmanager
    def attn_impl(self, model, impl: str | None):
        """Temporarily switch the attention implementation (tree masks need sdpa/eager/flex)."""
        if impl is None or not hasattr(model, "set_attn_implementation"):
            yield
            return
        prev = model.config._attn_implementation
        if prev == impl:
            yield
            return
        model.set_attn_implementation(impl)
        try:
            yield
        finally:
            model.set_attn_implementation(prev)

    def teacher_variants(self, tr: TrajPayload, gate: bool, chunk_index: int = 0) -> list[Variant]:
        out = [Variant([], lambda: self.adapter_ctx(None))]
        if gate:
            out += [Variant(tr.prefix(s, chunk_index), (lambda s=s: self.adapter_ctx(s))) for s in self.sources]
        return out


# ------------------------------------------------------------------------------------ phase B

@torch.no_grad()
def score_traj(m: ChopModels, tr: TrajPayload, cfg: MethodConfig, need_pi: bool
               ) -> tuple[StateStats, torch.Tensor, SparseTargets]:
    """(stats, rho, sparse targets) for the eligible states of one trajectory."""
    stats, targets = [], []
    dev = m.device
    for ci, c in enumerate(tr.chunks):
        n = c.gen_end - c.gen_start
        s_ids = tr.s_prompt + tr.response[c.ctx_start:c.gen_end]
        first_s = len(tr.s_prompt) + (c.gen_start - c.ctx_start)
        rows = list(range(first_s - 1, first_s - 1 + n))
        s_logits, _, _ = root_forward(m.student, torch.tensor(s_ids, device=dev), keep_rows=rows)
        t_ctx = tr.t_prompt + tr.response[c.ctx_start:c.gen_end]
        first_t = len(tr.t_prompt) + (c.gen_start - c.ctx_start)
        with m.adapter_ctx(None):
            _, th, _ = root_forward(m.teacher, torch.tensor(t_ctx, device=dev), keep_rows=[0], capture_hidden=True)
        head = m.teacher.get_output_embeddings()
        t_rows = th[first_t - 1:first_t - 1 + n]
        variants = []
        if need_pi and m.sources:
            for s in m.sources:
                pre = tr.prefix(s, ci)
                with m.adapter_ctx(s):
                    _, vh, _ = root_forward(m.teacher, torch.tensor(pre + t_ctx, device=dev), keep_rows=[0],
                                            capture_hidden=True)
                variants.append(head_fn(vh[len(pre) + first_t - 1:len(pre) + first_t - 1 + n], head))
        tok = torch.tensor(tr.response[c.gen_start:c.gen_end], device=dev)
        st, tg = score_states(rows_fn(s_logits), head_fn(t_rows, head), variants, tok, cfg.token.topk,
                              _teacher_k(cfg), m.logit_chunk)
        stats.append(st)
        targets.append(tg)
    ent = torch.cat([s.entropy for s in stats])
    tv = torch.cat([s.tv for s in stats])
    sent = torch.cat([s.student_entropy for s in stats])
    kap = torch.cat([s.kappa for s in stats]) if stats[0].kappa is not None else None
    elig = tr.eligible.to(ent.device)
    st = StateStats(ent[elig], tv[elig], None if kap is None else kap[elig], sent[elig])
    rho = reliability(st.entropy, st.tv, cfg.reliability)
    return st, rho, SparseTargets.cat(targets).select(elig)


def _teacher_k(cfg: MethodConfig) -> int:
    return cfg.token.topk if cfg.token.support == "teacher" else 32


# ------------------------------------------------------------------------------------ phase C

@torch.no_grad()
def sample_segments(m: ChopModels, tr: TrajPayload, specs: list[AnchorSpec], cfg: MethodConfig,
                    generator: torch.Generator | None = None) -> list[AnchorSegments]:
    """K'-1 student segments and K teacher anchors per anchor state (tree decoding per chunk)."""
    s = cfg.segment
    out = [AnchorSegments(sp, chunk_of(sp.pos, tr.chunks)) for sp in specs]
    groups: dict[tuple[Chunk, bool], list[int]] = {}
    for i, a in enumerate(out):
        groups.setdefault((a.chunk, a.spec.gate), []).append(i)
    for (c, gate), idx in groups.items():
        top = max(out[i].spec.pos for i in idx)
        s_ctx = tr.s_prompt + tr.response[c.ctx_start:top]
        s_anchor = [len(tr.s_prompt) + (out[i].spec.pos - c.ctx_start) for i in idx]
        with m.attn_impl(m.student, "sdpa"):
            stud = tree_decode(m.student, s_ctx, s_anchor, s.num_student_segments - 1, s.segment_length,
                               temperature=s.student_temperature, eos_id=m.eos_id, generator=generator,
                               pad_id=m.pad_id, lockstep=m.lockstep)
            if s.anchor_source == "student":
                teach = tree_decode(m.student, s_ctx, s_anchor, s.num_anchors, s.segment_length,
                                    temperature=s.teacher_temperature, eos_id=m.eos_id, generator=generator,
                                    pad_id=m.pad_id, lockstep=m.lockstep)
        if s.anchor_source != "student":
            t_ctx = tr.t_prompt + tr.response[c.ctx_start:top]
            t_anchor = [len(tr.t_prompt) + (out[i].spec.pos - c.ctx_start) for i in idx]
            with m.attn_impl(m.teacher, "sdpa"):
                teach = tree_decode(m.teacher, t_ctx, t_anchor, s.num_anchors, s.segment_length,
                                    m.teacher_variants(tr, gate, tr.chunks.index(c)), s.teacher_temperature,
                                    eos_id=m.eos_id,
                                    generator=generator, pad_id=m.pad_id, lockstep=m.lockstep)
        for j, i in enumerate(idx):
            out[i].student = [x if x else [m.pad_id] for x in stud[j]]
            out[i].teacher = [x if x else [m.pad_id] for x in teach[j]]
    return out


# ------------------------------------------------------------------------------------ phase D

def _layout(tr: TrajPayload, c: Chunk, segs: list[AnchorSegments], with_teacher: bool):
    main = tr.s_prompt + tr.response[c.ctx_start:c.gen_end]
    branches, owners = [], []
    for ai, a in enumerate(segs):
        idx = len(tr.s_prompt) + (a.spec.pos - c.ctx_start)
        for seg in a.student:
            branches.append((idx, torch.tensor(seg)))
            owners.append((ai, "s"))
        if with_teacher:
            for seg in a.teacher:
                branches.append((idx, torch.tensor(seg)))
                owners.append((ai, "t"))
    return build_tree(torch.tensor(main), branches), owners


@torch.no_grad()
def segment_costs(m: ChopModels, tr: TrajPayload, segs: list[AnchorSegments], cfg: MethodConfig,
                  generator: torch.Generator | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-student-sample costs [n, K'], distances [n], teacher [n, K, d] and student [n, K', d] embeddings."""
    s = cfg.segment
    L = s.segment_length
    teach, stud = [], []
    impl = "flex_attention" if m.attention == "flex" else "sdpa"
    for c in tr.chunks:
        group = [a for a in segs if a.chunk == c]
        if not group:
            continue
        lay, owners = _layout(tr, c, group, with_teacher=True)
        with m.attn_impl(m.student, impl):
            _, h, _ = root_forward(m.student, lay.input_ids.to(m.device), lay.position_ids.to(m.device),
                                   attention_mask=m.tree_mask(lay), keep_rows=[0], capture_hidden=True)
        emb = lay.branch_mean_hidden(h)
        for ai, a in enumerate(group):
            idx = len(tr.s_prompt) + (a.spec.pos - c.ctx_start)
            end = min(idx + L, lay.main_len)
            s_e = [h[idx:end].mean(0)] + [emb[b] for b, (o, k) in enumerate(owners) if o == ai and k == "s"]
            t_e = [emb[b] for b, (o, k) in enumerate(owners) if o == ai and k == "t"]
            stud.append(torch.stack(s_e))
            teach.append(torch.stack(t_e))
    t_emb, s_emb = torch.stack(teach).float(), torch.stack(stud).float()
    if s.distance == "sw":
        dirs = random_directions(t_emb.shape[0], s.num_projections, t_emb.shape[-1], generator).to(t_emb.device)
        dist, costs = sliced_w2(t_emb, s_emb, dirs)
    else:
        dist, costs = mmd2_gaussian(t_emb, s_emb)
    return costs, dist, t_emb, s_emb


# ------------------------------------------------------------------------------------ phase E

def update_traj(m: ChopModels, tr: TrajPayload, route: Route | None, segs: list[AnchorSegments],
                adv: torch.Tensor | None, targets: SparseTargets, cfg: MethodConfig, normalizer: int,
                backward: bool = True, extra: dict | None = None) -> tuple[float, dict[str, float]]:
    """Eq. (chop_lite) for one trajectory with sparse targets; gradients accumulate in the student.

    ``route`` covers the trajectory's eligible states in order; ``adv`` holds the batch-normalised
    advantages of this trajectory's anchors ([n, K'], same order as ``segs``).  For
    ``cfg.name == 'reopold'`` ``extra`` carries {'step', 'entropy_tau'} (batch-level threshold of
    the entropy-guided phase); for ``'grpo'`` it carries {'advantage'} of the trajectory.
    """
    if cfg.name in ("reopold", "grpo"):
        return _update_rl_style(m, tr, targets, cfg, normalizer, backward, extra or {})
    s = cfg.segment
    L = s.segment_length
    dev = m.device
    targets = targets.to(dev)
    elig = tr.eligible
    state_of_pos = torch.full((len(tr.response),), -1, dtype=torch.long)
    state_of_pos[elig] = torch.arange(int(elig.sum()))
    special = {m.eos_id, m.pad_id}
    total, metrics = 0.0, {"token_states": 0.0, "segment_anchors": 0.0}
    impl = "flex_attention" if m.attention == "flex" else "sdpa"
    for c in tr.chunks:
        group = [a for a in segs if a.chunk == c]
        lay, owners = _layout(tr, c, group, with_teacher=False)
        first = len(tr.s_prompt) + (c.gen_start - c.ctx_start)
        n = c.gen_end - c.gen_start
        resp_rows = list(range(first - 1, first - 1 + n))
        branch_rows = [int(st) + j for st, ln in zip(lay.start, lay.length, strict=True) for j in range(int(ln) - 1)]
        keep = resp_rows + branch_rows
        row_of = {r: i for i, r in enumerate(keep)}
        with m.attn_impl(m.student, impl if group else None):
            logits, _, _ = root_forward(m.student, lay.input_ids.to(dev), lay.position_ids.to(dev),
                                        attention_mask=m.tree_mask(lay) if group else None, keep_rows=keep)
        loss = logits.float().sum() * 0.0   # keeps the graph: every forward gets a backward (FSDP)
        # ---- token-level terms for the eligible states of this chunk
        pos = torch.arange(c.gen_start, c.gen_end)
        sel = elig[pos]
        if bool(sel.any()):
            sid = state_of_pos[pos[sel]]
            lg = logits[:n][sel.to(dev)]
            tok = torch.tensor(tr.response, device=dev)[pos[sel].to(dev)]
            tg = targets.select(sid.to(dev))
            trusted = route.trusted[sid] & route.in_stratum[sid]
            untrusted = ~route.trusted[sid] & route.in_stratum[sid]
            outside = ~route.in_stratum[sid]
            gate = route.pi_gate[sid]
            keep_tok = torch.tensor([t not in special for t in tok.tolist()]) if cfg.token.mask_special_tokens \
                else torch.ones_like(trusted)
            tmask = trusted & keep_tok
            if bool(tmask.any()):
                loss = loss + sparse_token_loss(lg[tmask.to(dev)], tok[tmask.to(dev)], tg.select(tmask), gate[tmask],
                                                cfg.token.estimator, cfg.token.support).sum()
            if cfg.beta_outcome and tr.reward and bool(untrusted.any()):
                lp = F.log_softmax(lg[untrusted.to(dev)].float(), -1).gather(
                    -1, tok[untrusted.to(dev)].unsqueeze(-1)).squeeze(-1)
                loss = loss - cfg.beta_outcome * tr.reward * lp.sum()
            if bool(outside.any()):
                o = outside.to(dev)
                loss = loss + sparse_token_loss(lg[o], tok[o], tg.select(outside),
                                                torch.zeros(int(outside.sum()), dtype=torch.bool), "sampled").sum()
            metrics["token_states"] += float(sel.sum())
        # ---- segment surrogate for the anchors of this chunk
        if group and adv is not None:
            lp_rows = []
            for ai, a in enumerate(group):
                idx = len(tr.s_prompt) + (a.spec.pos - c.ctx_start)
                end = min(idx + L, lay.main_len)
                real = [row_of[r] for r in range(idx - 1, end - 1)]
                tok = lay.input_ids[idx:end].to(dev)
                lp_real = F.log_softmax(logits[real].float(), -1).gather(-1, tok.unsqueeze(-1)).squeeze(-1)
                rows = [F.pad(lp_real, (0, L - lp_real.numel()))]
                masks = [torch.arange(L, device=dev) < lp_real.numel()]
                for b, (o, _k) in enumerate(owners):
                    if o != ai:
                        continue
                    st_b, ln = int(lay.start[b]), int(lay.length[b])
                    src = [row_of[idx - 1]] + [row_of[st_b + j] for j in range(ln - 1)]
                    btok = lay.input_ids[st_b:st_b + ln].to(dev)
                    lpb = F.log_softmax(logits[src].float(), -1).gather(-1, btok.unsqueeze(-1)).squeeze(-1)
                    rows.append(F.pad(lpb, (0, L - ln)))
                    masks.append(torch.arange(L, device=dev) < ln)
                lp_rows.append((torch.stack(rows), torch.stack(masks)))
            k0 = next(i for i, a in enumerate(segs) if a is group[0])
            a_adv = adv[k0:k0 + len(group)].to(dev)
            w = torch.tensor([a.spec.weight for a in group], device=dev)
            seg_lp = torch.stack([r for r, _ in lp_rows])
            seg_mask = torch.stack([mk for _, mk in lp_rows])
            loss = loss + s.coef * segment_surrogate(seg_lp, seg_mask, a_adv, w)
            metrics["segment_anchors"] += float(len(group))
        loss = loss / max(normalizer, 1)
        if backward:
            loss.backward()
        total += float(loss.detach())
    return total, metrics


# ------------------------------------------------------------------------- lockstep accounting

def student_forwards(phase: str, trajs: list[TrajPayload], cfg: MethodConfig,
                     segs: list[list[AnchorSegments]] | None = None) -> int:
    """Number of student forward calls a phase issues (the teacher is replicated: no collectives)."""
    if phase in ("score", "update"):
        return sum(len(t.chunks) for t in trajs)
    if phase == "sample":
        per_group = cfg.segment.segment_length * (2 if cfg.segment.anchor_source == "student" else 1)
        return per_group * sum(len({(a.chunk, a.spec.gate) for a in s}) for s in (segs or []))
    if phase == "costs":
        return sum(len({a.chunk for a in s}) for s in (segs or []))
    raise ValueError(f"unknown phase {phase!r}")


def dummy_forward(model, backward: bool = False) -> None:
    """A minimal forward (and zero-gradient backward) that issues the same FSDP collectives."""
    dev = next(model.parameters()).device
    ctx = contextlib.nullcontext() if backward else torch.no_grad()
    with ctx:
        out = model(input_ids=torch.zeros(1, 2, dtype=torch.long, device=dev), logits_to_keep=1)
        if backward:
            (out.logits.float().sum() * 0.0).backward()


def _update_rl_style(m: ChopModels, tr: TrajPayload, targets: SparseTargets, cfg: MethodConfig, normalizer: int,
                     backward: bool, extra: dict) -> tuple[float, dict[str, float]]:
    """REOPOLD (clipped log-ratio reward with token filtering) and GRPO (outcome advantage)."""
    import math

    dev = m.device
    targets = targets.to(dev)
    elig = tr.eligible
    state_of_pos = torch.full((len(tr.response),), -1, dtype=torch.long)
    state_of_pos[elig] = torch.arange(int(elig.sum()))
    total, kept = 0.0, 0.0
    for c in tr.chunks:
        ids = tr.s_prompt + tr.response[c.ctx_start:c.gen_end]
        first = len(tr.s_prompt) + (c.gen_start - c.ctx_start)
        n = c.gen_end - c.gen_start
        rows = list(range(first - 1, first - 1 + n))
        logits, _, _ = root_forward(m.student, torch.tensor(ids, device=dev), keep_rows=rows)
        pos = torch.arange(c.gen_start, c.gen_end)
        sel = elig[pos]
        loss = logits.float().sum() * 0.0   # keeps the graph: every forward gets a backward (FSDP)
        if bool(sel.any()):
            lg = logits[sel.to(dev)].float()
            tok = torch.tensor(tr.response, device=dev)[pos[sel].to(dev)]
            lp = F.log_softmax(lg, -1).gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            if cfg.name == "grpo":
                loss = -(float(extra["advantage"]) * lp).sum()
            else:
                r = (targets.select(state_of_pos[pos[sel]].to(dev)).base_tok - lp).detach()
                floor = math.log(cfg.reopold_lambda / (1 - cfg.reopold_lambda))
                if int(extra.get("step", 0)) < cfg.reopold_switch_step:
                    keep = r >= floor
                else:
                    with torch.no_grad():
                        h = -(F.softmax(lg, -1) * F.log_softmax(lg, -1)).sum(-1)
                    keep = h >= float(extra["entropy_tau"])
                loss = -(r.clamp_min(floor) * lp * keep).sum()
                kept += float(keep.sum())
        loss = loss / max(normalizer, 1)
        if backward:
            loss.backward()
        total += float(loss.detach())
    return total, {"token_states": float(elig.sum()), "segment_anchors": 0.0, "kept": kept}

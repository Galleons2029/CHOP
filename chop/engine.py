"""Scalable building blocks shared by the distributed trainer (chop_verl) and the tests.

The reference trainer keeps full ``[T, V]`` logits, which is exact but does not scale to 30K-token
trajectories with a 151,936-token vocabulary.  This module provides equivalent computations that
never materialise more than a ``[chunk, V]`` logits block:

* :func:`score_states` computes, per state, H(q), TV(q, p), kappa = TV(q, q^(P)) and the sparse
  teacher targets every supported token loss needs (student top-k support, teacher top-k support,
  realised-token log-probs).  Losses computed from :class:`SparseTargets` equal the dense ones.
* :func:`chunked_token_loss` evaluates a per-state loss from last hidden states with gradient,
  recomputing each logits chunk in backward (activation checkpointing), so the logits are never
  stored for the whole sequence.
* :func:`tree_decode` samples the segment branches of all anchors of one trajectory with one shared
  prefix KV cache: every decoding step appends one token per branch and a tree mask lets each branch
  attend to its own anchor prefix and its own earlier tokens.  Variants (base teacher + PI adapters
  with their own prefixes) run in lockstep and their logits are composed additively (Eq. 14).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from chop.rollout import sample_token

# ------------------------------------------------------------------------------ model access


def decoder(model):
    """The transformer body (without LM head) of an HF causal LM, PEFT-wrapped or not."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model


def lm_head(model) -> torch.nn.Module:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.get_output_embeddings()


def final_norm(model) -> torch.nn.Module:
    return decoder(model).norm


def root_forward(model, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None, attention_mask=None,
                 past_key_values=None, use_cache: bool = False, keep_rows=None, capture_hidden: bool = False):
    """Call the *root* module (required under FSDP2, where the root owns embeddings / norm / tied head).

    ``keep_rows`` (list/tensor of positions) restricts the LM head to those rows via ``logits_to_keep``;
    ``capture_hidden`` records the post-norm last hidden states [S, d] with a hook on the final norm.
    Returns (logits [len(keep_rows) or S, V], hidden or None, cache).
    """
    ids = input_ids[None] if input_ids.dim() == 1 else input_ids
    pos = None if position_ids is None else (position_ids[None] if position_ids.dim() == 1 else position_ids)
    captured = {}
    handle = None
    if capture_hidden:
        handle = final_norm(model).register_forward_hook(lambda _m, _i, out: captured.__setitem__("h", out))
    kwargs = {}
    if keep_rows is not None:
        kwargs["logits_to_keep"] = torch.as_tensor(keep_rows, dtype=torch.long, device=ids.device)
    try:
        out = model(input_ids=ids, position_ids=pos, attention_mask=attention_mask, past_key_values=past_key_values,
                    use_cache=use_cache, **kwargs)
    finally:
        if handle is not None:
            handle.remove()
    hidden = captured["h"][0] if capture_hidden else None
    return out.logits[0], hidden, out.past_key_values


def hidden_states(model, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
                  attention_mask=None, past_key_values=None, use_cache: bool = False):
    """Last-layer (post-norm) hidden states [S, d] and the cache, computing a single logits row."""
    _, h, cache = root_forward(model, input_ids, position_ids, attention_mask, past_key_values, use_cache,
                               keep_rows=[0], capture_hidden=True)
    return h, cache


def chunks(n: int, size: int):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


# --------------------------------------------------------------------------- sparse targets


@dataclass
class SparseTargets:
    """Per-state teacher information sufficient for every token loss in this repository."""
    student_ids: torch.Tensor            # [N, k] student top-k support (fixed at scoring time)
    base_at_student: torch.Tensor        # [N, k] teacher logits on that support
    teacher_ids: torch.Tensor            # [N, k2] teacher top-k2 support (Top-K LSM)
    base_at_teacher: torch.Tensor        # [N, k2]
    base_tok: torch.Tensor               # [N] teacher log-prob of the realised token
    pi_at_student: torch.Tensor | None = None   # composite PI teacher on the student support
    pi_tok: torch.Tensor | None = None

    def to(self, device) -> SparseTargets:
        return SparseTargets(**{k: (None if v is None else v.to(device)) for k, v in self.__dict__.items()})

    def select(self, idx) -> SparseTargets:
        f = {k: (None if v is None else v[idx]) for k, v in self.__dict__.items()}
        return SparseTargets(**f)

    @staticmethod
    def cat(parts: Sequence[SparseTargets]) -> SparseTargets:
        keys = parts[0].__dict__.keys()
        return SparseTargets(**{k: (None if getattr(parts[0], k) is None else
                                    torch.cat([getattr(p, k) for p in parts])) for k in keys})


@dataclass
class StateStats:
    entropy: torch.Tensor                # [N] H(q_base)
    tv: torch.Tensor                     # [N] TV(q_base, p)
    kappa: torch.Tensor | None           # [N] TV(q_base, q_composite)
    student_entropy: torch.Tensor | None = None  # [N] H(p) (REOPOLD entropy-guided sampling)


LogitsFn = Callable[[slice], torch.Tensor]


def head_fn(hidden: torch.Tensor, head: torch.nn.Module) -> LogitsFn:
    """Logits chunks from stored hidden states (used for the unsharded frozen teacher)."""
    return lambda sl: head(hidden[sl]).float()


def rows_fn(logits: torch.Tensor) -> LogitsFn:
    """Logits chunks from precomputed logits rows (used for the FSDP-sharded student)."""
    return lambda sl: logits[sl].float()


@torch.no_grad()
def score_states(student: LogitsFn, teacher: LogitsFn, variants: Sequence[LogitsFn], tokens: torch.Tensor,
                 k: int = 64, k_teacher: int = 32, chunk: int = 1024) -> tuple[StateStats, SparseTargets]:
    """Chunked per-state statistics and sparse targets.

    ``student``/``teacher``/``variants`` map a row slice to float logits for those states; the
    composite PI teacher is base + sum_v (logits_v - base).
    """
    ent, tv, kap, sent = [], [], [], []
    parts = []
    for sl in chunks(tokens.numel(), chunk):
        p, q = student(sl), teacher(sl)
        logp, logq = F.log_softmax(p, -1), F.log_softmax(q, -1)
        ent.append(-(logq.exp() * logq).sum(-1))
        sent.append(-(logp.exp() * logp).sum(-1))
        tv.append(0.5 * (logp.exp() - logq.exp()).abs().sum(-1))
        sid = p.topk(min(k, p.shape[-1]), -1).indices
        tid = q.topk(min(k_teacher, q.shape[-1]), -1).indices
        tok = tokens[sl].unsqueeze(-1).to(p.device)
        pi_s = pi_t = None
        if variants:
            comp = q.clone()
            for v in variants:
                comp += v(sl) - q
            logc = F.log_softmax(comp, -1)
            kap.append(0.5 * (logq.exp() - logc.exp()).abs().sum(-1))
            pi_s = comp.gather(-1, sid)
            pi_t = logc.gather(-1, tok).squeeze(-1)
        parts.append(SparseTargets(sid, q.gather(-1, sid), tid, q.gather(-1, tid), logq.gather(-1, tok).squeeze(-1),
                                   pi_s, pi_t))
    stats = StateStats(torch.cat(ent), torch.cat(tv), torch.cat(kap) if kap else None, torch.cat(sent))
    return stats, SparseTargets.cat(parts)


def _renorm_kl(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    logp, logq = F.log_softmax(s.float(), -1), F.log_softmax(t.float(), -1)
    return (logp.exp() * (logp - logq)).sum(-1)


def sparse_token_loss(student_logits: torch.Tensor, tokens: torch.Tensor, tgt: SparseTargets, use_pi: torch.Tensor,
                      estimator: str = "topk", support: str = "student") -> torch.Tensor:
    """Per-state token loss from full student logits [c, V] and sparse teacher targets.

    Equals :func:`chop.losses.token_divergence` with dense logits for the 'topk' (student or teacher
    support) and 'sampled' estimators; ``use_pi`` selects the composite teacher per state.
    """
    use_pi = use_pi.to(student_logits.device)
    if estimator == "sampled":
        lp = F.log_softmax(student_logits.float(), -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        lq = tgt.base_tok if tgt.pi_tok is None else torch.where(use_pi, tgt.pi_tok, tgt.base_tok)
        return (lp - lq).detach() * lp
    if estimator != "topk":
        raise ValueError(f"sparse targets support 'topk' and 'sampled' estimators, not {estimator!r}")
    if support == "student":
        t = tgt.base_at_student
        if tgt.pi_at_student is not None:
            t = torch.where(use_pi.unsqueeze(-1), tgt.pi_at_student, t)
        return _renorm_kl(student_logits.gather(-1, tgt.student_ids), t)
    if support == "teacher":
        if bool(use_pi.any()):
            raise ValueError("teacher-support top-k targets are stored for the base teacher only")
        return _renorm_kl(student_logits.gather(-1, tgt.teacher_ids), tgt.base_at_teacher)
    raise ValueError(f"unknown support {support!r}")


def chunked_token_loss(hidden: torch.Tensor, head: torch.nn.Module,
                       fn: Callable[[torch.Tensor, slice], torch.Tensor], chunk: int = 1024) -> torch.Tensor:
    """sum over chunks of fn(logits_chunk, slice).sum(), recomputing logits in backward."""
    total = hidden.new_zeros((), dtype=torch.float32)
    for sl in chunks(hidden.shape[0], chunk):
        def block(h, sl=sl):
            return fn(head(h), sl).sum()

        if hidden.requires_grad and torch.is_grad_enabled():
            total = total + checkpoint(block, hidden[sl], use_reentrant=False)
        else:
            total = total + block(hidden[sl])
    return total


# ----------------------------------------------------------------------------- tree decoding


@dataclass
class Variant:
    """One teacher view: its PI prefix ids and a context manager activating its adapter."""
    prefix: list[int]
    activate: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext


@torch.no_grad()
def tree_decode(model, context: Sequence[int], anchors: Sequence[int], num_samples: int, max_new_tokens: int,
                variants: Sequence[Variant] | None = None, temperature: float = 0.7, top_p: float = 1.0,
                eos_id: int | None = None, generator: torch.Generator | None = None,
                pad_id: int = 0, lockstep: bool = False) -> list[list[list[int]]]:
    """Sample ``num_samples`` continuations of length <= ``max_new_tokens`` at every anchor.

    With ``lockstep=True`` decoding never stops early, so every call runs exactly
    len(variants) * max_new_tokens forwards (required when the model is FSDP-sharded).

    ``context`` is the shared main sequence; anchor a means the branch continues ``context[:a]``.
    With several ``variants`` (base first) their logits are composed additively at every step.
    Returns segments[anchor][sample] as token lists (truncated after EOS).
    """
    variants = list(variants) if variants else [Variant([])]
    ctx = torch.as_tensor(list(context), dtype=torch.long)
    na, n = len(anchors), num_samples
    b = na * n                                   # branches, ordered anchor-major
    anchor_of = torch.tensor([a for a in anchors for _ in range(n)])
    if b == 0:
        return []
    from transformers import DynamicCache

    caches, first_logits = [], []
    for v in variants:
        with v.activate():
            ids = torch.cat([torch.as_tensor(v.prefix, dtype=torch.long), ctx])
            rows = len(v.prefix) + anchor_of - 1                 # last context position of each branch
            lg, _, cache = root_forward(model, ids, torch.arange(ids.numel()), past_key_values=DynamicCache(),
                                        use_cache=True, keep_rows=rows)
            first_logits.append(lg.float())
        caches.append(cache)

    def compose(logits: list[torch.Tensor]) -> torch.Tensor:
        out = logits[0].clone()
        for lg in logits[1:]:
            out += lg - logits[0]
        return out

    segs = torch.full((b, max_new_tokens), pad_id, dtype=torch.long)
    done = torch.zeros(b, dtype=torch.bool)
    lengths = torch.full((b,), max_new_tokens, dtype=torch.long)
    logits = compose(first_logits)
    neg = torch.finfo(torch.float32).min
    for j in range(max_new_tokens):
        tok = sample_token(logits, temperature, top_p, generator)
        tok = torch.where(done, torch.full_like(tok, pad_id), tok)
        segs[:, j] = tok
        if eos_id is not None:
            hit = (~done) & (tok == eos_id)
            lengths[hit] = j + 1
            done |= hit
        if j == max_new_tokens - 1 or (bool(done.all()) and not lockstep):
            break
        step_logits = []
        for v, cache in zip(variants, caches, strict=True):
            base_len = len(v.prefix) + ctx.numel()
            kv = base_len + (j + 1) * b
            mask = torch.full((b, kv), neg)
            prefix_ok = torch.arange(base_len).unsqueeze(0) < (len(v.prefix) + anchor_of).unsqueeze(1)
            mask[:, :base_len] = torch.where(prefix_ok, 0.0, neg)
            for s in range(j + 1):                          # own tokens of steps 0..j
                mask[torch.arange(b), base_len + s * b + torch.arange(b)] = 0.0
            pos = (len(v.prefix) + anchor_of + j).unsqueeze(0)
            with v.activate():
                lg, _, _ = root_forward(model, tok.unsqueeze(0), pos, attention_mask=mask[None, None],
                                        past_key_values=cache, use_cache=True)
                step_logits.append(lg.float())
        logits = compose(step_logits)
    return [[segs[a * n + i, :int(lengths[a * n + i])].tolist() for i in range(n)] for a in range(na)]

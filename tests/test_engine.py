import contextlib

import pytest
import torch

from chop import divergences as D
from chop import engine as E
from chop.config import TokenLossConfig
from chop.losses import token_divergence
from chop.pi.composition import PIConditionalTeacher
from chop.rollout import sample_segment
from chop.tiny import tiny_qwen3


def _heads(gen, v=40, d=8):
    s, t = torch.nn.Linear(d, v, bias=False), torch.nn.Linear(d, v, bias=False)
    return s, t


def test_hidden_states_and_head_reproduce_logits(gen):
    m = tiny_qwen3()
    ids = torch.randint(3, 96, (12,), generator=gen)
    h, _ = E.hidden_states(m, ids)
    full = m(input_ids=ids[None]).logits[0]
    assert torch.allclose(E.lm_head(m)(h), full, atol=1e-5)
    rows, _, _ = E.root_forward(m, ids, keep_rows=[2, 7])
    assert torch.allclose(rows, full[[2, 7]], atol=1e-5)


def test_score_states_matches_dense(gen):
    torch.manual_seed(0)
    sh, th = _heads(gen)
    n, d = 50, 8
    hs, ht = torch.randn(n, d, generator=gen), torch.randn(n, d, generator=gen)
    hv = [torch.randn(n, d, generator=gen), torch.randn(n, d, generator=gen)]
    tok = torch.randint(40, (n,), generator=gen)
    stats, tgt = E.score_states(E.head_fn(hs, sh), E.head_fn(ht, th), [E.head_fn(h, th) for h in hv], tok,
                                k=6, k_teacher=5, chunk=16)
    with torch.no_grad():
        p, q = sh(hs), th(ht)
        comp = q + sum(th(h) - q for h in hv)
    assert torch.allclose(stats.entropy, D.entropy(q), atol=1e-5)
    assert torch.allclose(stats.tv, D.total_variation(q, p), atol=1e-5)
    assert torch.allclose(stats.kappa, D.total_variation(q, comp), atol=1e-5)
    use_pi = torch.rand(n, generator=gen) > 0.5
    target = torch.where(use_pi.unsqueeze(-1), comp, q)
    for est, sup, k in (("topk", "student", 6), ("sampled", "student", 6)):
        cfg = TokenLossConfig(estimator=est, topk=k, support=sup)
        logits = sh(hs).detach().requires_grad_(True)
        dense = token_divergence(logits, target, tok, cfg)
        (g1,) = torch.autograd.grad(dense.sum(), logits)
        sparse = E.sparse_token_loss(logits, tok, tgt, use_pi, est, sup)
        (g2,) = torch.autograd.grad(sparse.sum(), logits)
        assert torch.allclose(dense, sparse, atol=1e-5) and torch.allclose(g1, g2, atol=1e-5)
    cfg = TokenLossConfig(estimator="topk", topk=5, support="teacher")
    logits = sh(hs).detach()
    dense = token_divergence(logits, q, tok, cfg)
    sparse = E.sparse_token_loss(logits, tok, tgt, torch.zeros(n, dtype=torch.bool), "topk", "teacher")
    assert torch.allclose(dense, sparse, atol=1e-5)


def test_chunked_token_loss_matches_unchunked(gen):
    head = torch.nn.Linear(8, 30, bias=False)
    h = torch.randn(37, 8, generator=gen, requires_grad=True)
    t = torch.randn(37, 30, generator=gen)

    def fn(logits, sl):
        return D.reverse_kl_full(logits, t[sl])

    full = D.reverse_kl_full(head(h), t).sum()
    (g_full,) = torch.autograd.grad(full, h)
    chunked = E.chunked_token_loss(h, head, fn, chunk=8)
    (g_chunk,) = torch.autograd.grad(chunked, h)
    assert torch.allclose(full, chunked, atol=1e-5) and torch.allclose(g_full, g_chunk, atol=1e-5)


def test_tree_decode_student_matches_per_anchor_greedy(gen):
    m = tiny_qwen3()
    ctx = torch.randint(3, 96, (25,), generator=gen).tolist()
    anchors = [5, 12, 25]
    segs = E.tree_decode(m, ctx, anchors, num_samples=2, max_new_tokens=6, temperature=0.0)
    for a, per_anchor in zip(anchors, segs, strict=True):
        ref = sample_segment(m, ctx[:a], 6, temperature=0.0)
        assert per_anchor[0] == ref and per_anchor[1] == ref


def test_tree_decode_composite_matches_row_sampler(gen):
    peft = pytest.importorskip("peft")
    base = tiny_qwen3()
    cfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    model = peft.get_peft_model(base, cfg, adapter_name="a1")
    model.add_adapter("a2", cfg)
    model.eval()
    prefix = {"a1": [20, 21, 22], "a2": [30]}

    def act(name):
        def cm():
            if name is None:
                return model.disable_adapter()
            model.set_adapter(name)
            return contextlib.nullcontext()
        return cm

    ctx = torch.randint(3, 96, (15,), generator=gen).tolist()
    variants = [E.Variant([], act(None)), E.Variant(prefix["a1"], act("a1")), E.Variant(prefix["a2"], act("a2"))]
    segs = E.tree_decode(model, ctx, [7, 15], num_samples=1, max_new_tokens=5, variants=variants, temperature=0.0)
    teacher = PIConditionalTeacher(model, pad_id=0)
    for a, per_anchor in zip([7, 15], segs, strict=True):
        toks, _ = teacher.sample(ctx[:a], prefix, ["a1", "a2"], 1, 5, temperature=1e-6)
        assert per_anchor[0] == toks[0].tolist()


def test_tree_decode_stops_at_eos(gen):
    m = tiny_qwen3()
    ctx = torch.randint(3, 96, (10,), generator=gen).tolist()
    greedy = sample_segment(m, ctx, 4, temperature=0.0)
    segs = E.tree_decode(m, ctx, [10], 1, 4, temperature=0.0, eos_id=greedy[1])
    assert segs[0][0] == greedy[:2]

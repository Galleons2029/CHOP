import torch

from chop.config import MethodConfig, RouterConfig
from chop.losses import SegmentTerms, StateBatch, chop_loss, grpo_advantages, reopold_loss, token_divergence
from chop.router import CellRouter


def _states(gen, n=64, v=17, reward=1.0, pi=True):
    s = torch.randn(n, v, generator=gen, requires_grad=True)
    return StateBatch(
        student_logits=s,
        tokens=torch.randint(v, (n,), generator=gen),
        teacher_logits=torch.randn(n, v, generator=gen),
        teacher_pi_logits=torch.randn(n, v, generator=gen) if pi else None,
        reward=torch.full((n,), reward),
    )


def _route(cfg, n, gen):
    r = CellRouter(cfg.router, seed=0)
    r.rho_q.initialize(torch.rand(4096, generator=gen))
    r.kappa_q.initialize(torch.rand(4096, generator=gen))
    return r.route(torch.rand(n, generator=gen), torch.rand(n, generator=gen))


def test_untrusted_rows_only_get_outcome_gradient(gen):
    cfg = MethodConfig()
    st = _states(gen, reward=0.0)
    route = _route(cfg, 64, gen)
    out = chop_loss(st, route, None, cfg)
    out.loss.backward()
    g = st.student_logits.grad.abs().sum(-1)
    assert (g[~route.trusted] == 0).all()        # R = 0: no outcome term, segment handled separately
    assert (g[route.trusted] > 0).all()
    assert 0.2 < out.metrics["segment"] < 0.4


def test_token_baseline_preset_equals_plain_opd(gen):
    cfg = MethodConfig(router=RouterConfig(rho_gate="none", kappa_gate="none"))
    cfg.token.estimator = "sampled"
    st = _states(gen)
    route = _route(cfg, 64, gen)
    out = chop_loss(st, route, None, cfg)
    ref = token_divergence(st.student_logits, st.teacher_logits, st.tokens, cfg.token).mean()
    assert torch.allclose(out.loss, ref)


def test_pi_gate_selects_composite_teacher(gen):
    cfg = MethodConfig(router=RouterConfig(rho_gate="none", kappa_gate="always"))
    cfg.token.estimator = "full"
    st = _states(gen)
    out = chop_loss(st, _route(cfg, 64, gen), None, cfg)
    ref = token_divergence(st.student_logits, st.teacher_pi_logits, st.tokens, cfg.token).mean()
    assert torch.allclose(out.loss, ref)


def test_segment_terms_contribute_gradient(gen):
    cfg = MethodConfig()
    st = _states(gen)
    logp = torch.randn(3, 8, 4, generator=gen, requires_grad=True)
    seg = SegmentTerms(logp=logp, mask=torch.ones(3, 8, 4, dtype=torch.bool), weights=torch.tensor([1.0, 2.0, 3.0]),
                       teacher_emb=torch.randn(3, 8, 6, generator=gen),
                       student_emb=torch.randn(3, 8, 6, generator=gen))
    out = chop_loss(st, _route(cfg, 64, gen), seg, cfg, generator=gen)
    out.loss.backward()
    assert logp.grad.abs().sum() > 0 and out.metrics["num_anchors"] == 3


def test_reopold_phases(gen):
    cfg = MethodConfig(name="reopold", reopold_switch_step=10)
    st = _states(gen, n=200)
    early = reopold_loss(st, cfg, step=0)
    late = reopold_loss(st, cfg, step=20)
    assert abs(late.metrics["kept"] - 0.2) < 0.02
    assert 0 < early.metrics["kept"] <= 1


def test_grpo_advantages():
    adv = grpo_advantages(torch.tensor([1.0, 0.0, 1.0, 1.0]), torch.tensor([0, 0, 1, 1]))
    assert torch.allclose(adv, torch.tensor([1.0, -1.0, 0.0, 0.0]), atol=1e-4)

import numpy as np
import torch

from chop import divergences as D


def _np_softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def test_entropy_tv_match_numpy(gen):
    a, b = torch.randn(5, 11, generator=gen), torch.randn(5, 11, generator=gen)
    pa, pb = _np_softmax(a.numpy()), _np_softmax(b.numpy())
    np.testing.assert_allclose(D.entropy(a).numpy(), -(pa * np.log(pa)).sum(-1), rtol=1e-5)
    np.testing.assert_allclose(D.total_variation(a, b).numpy(), 0.5 * np.abs(pa - pb).sum(-1), rtol=1e-5)


def test_full_and_topk_kl(gen):
    s, t = torch.randn(4, 9, generator=gen), torch.randn(4, 9, generator=gen)
    p, q = _np_softmax(s.numpy()), _np_softmax(t.numpy())
    ref = (p * (np.log(p) - np.log(q))).sum(-1)
    np.testing.assert_allclose(D.reverse_kl_full(s, t).numpy(), ref, rtol=1e-5)
    # k = |V| recovers the full KL for either support
    np.testing.assert_allclose(D.reverse_kl_topk(s, t, k=9).numpy(), ref, rtol=1e-5)
    np.testing.assert_allclose(D.reverse_kl_topk(s, t, k=9, support="teacher").numpy(), ref, rtol=1e-5)


def test_topk_support_choice(gen):
    s, t = torch.randn(1, 20, generator=gen), torch.randn(1, 20, generator=gen)
    for support, ref in (("student", s), ("teacher", t)):
        idx = ref.topk(5).indices[0]
        p = torch.softmax(s[0, idx], -1)
        q = torch.softmax(t[0, idx], -1)
        manual = (p * (p.log() - q.log())).sum()
        assert torch.allclose(D.reverse_kl_topk(s, t, 5, support)[0], manual, atol=1e-6)


def test_gkd_limits(gen):
    s, t = torch.randn(3, 7, generator=gen), torch.randn(3, 7, generator=gen)
    assert torch.allclose(D.generalized_jsd(s, t, 1.0), D.reverse_kl_full(s, t), atol=1e-6)
    fwd = D.reverse_kl_full(t, s)  # KL(q || p)
    assert torch.allclose(D.generalized_jsd(s, t, 0.0), fwd, atol=1e-6)
    mid = D.generalized_jsd(s, t, 0.5)
    assert (mid >= 0).all() and (mid <= np.log(2) + 1e-6).all()


def test_sampled_surrogate_gradient_is_unbiased(gen):
    """sum_y p(y) grad[sg(l(y)) log p(y)] equals grad KL(p||q) exactly."""
    theta = torch.randn(6, generator=gen, requires_grad=True)
    t = torch.randn(6, generator=gen)
    kl = D.reverse_kl_full(theta, t)
    (g_true,) = torch.autograd.grad(kl, theta)
    p = torch.softmax(theta, -1).detach()
    total = 0.0
    for y in range(6):
        lp = D.token_logprobs(theta, torch.tensor(y))
        lq = D.token_logprobs(t, torch.tensor(y))
        total = total + p[y] * D.sampled_reverse_kl_surrogate(lp, lq)
    (g_est,) = torch.autograd.grad(total, theta)
    assert torch.allclose(g_est, g_true, atol=1e-6)


def test_chunked_equals_whole(gen):
    a, b = torch.randn(37, 13, generator=gen), torch.randn(37, 13, generator=gen)
    assert torch.allclose(D.chunked(D.total_variation, a, b, chunk=8), D.total_variation(a, b))


def test_topk_overlap_bounds(gen):
    a = torch.randn(4, 30, generator=gen)
    assert torch.allclose(D.topk_overlap(a, a, 5), torch.ones(4))

import numpy as np
import pytest
import torch

from chop import segment as S


def test_anchor_windows_first_state_and_weights():
    mask = torch.zeros(300, dtype=torch.bool)
    mask[[3, 10, 70, 71, 72, 200]] = True
    a, w = S.select_anchor_windows(mask, window=64, max_windows=None)
    assert a.tolist() == [3, 70, 200]
    assert w.tolist() == [2.0, 3.0, 1.0]


def test_anchor_subsampling_is_horvitz_thompson(gen):
    mask = torch.zeros(64 * 40, dtype=torch.bool)
    mask[::64] = True
    a, w = S.select_anchor_windows(mask, window=64, max_windows=10, generator=gen)
    assert a.numel() == 10 and torch.allclose(w, torch.full((10,), 4.0))
    assert torch.equal(a, a.sort().values)


def test_sliced_w2_matches_explicit_and_pot(gen):
    x = torch.randn(3, 8, 5, generator=gen)
    y = torch.randn(3, 8, 5, generator=gen)
    dirs = S.random_directions(3, 16, 5, gen)
    sw, costs = S.sliced_w2(x, y, dirs)
    for n in range(3):
        px = (x[n] @ dirs[n].T).numpy()
        py = (y[n] @ dirs[n].T).numpy()
        explicit = np.mean((np.sort(px, 0) - np.sort(py, 0)) ** 2)
        assert abs(explicit - float(sw[n])) < 1e-5
        ot = pytest.importorskip("ot")
        pot = ot.sliced_wasserstein_distance(x[n].numpy(), y[n].numpy(), projections=dirs[n].T.numpy(), p=2)
        assert abs(float(pot) ** 2 - float(sw[n])) < 1e-5
    assert torch.allclose(costs.mean(-1), sw)


def test_sliced_w2_unequal_sizes_zero_for_duplicated_cloud(gen):
    x = torch.randn(1, 4, 3, generator=gen)
    y = torch.cat([x, x], dim=1)
    sw, _ = S.sliced_w2(x, y, S.random_directions(1, 8, 3, gen))
    assert float(sw) < 1e-10


def test_mmd_per_sample_costs_give_exact_gradient(gen):
    """Exact enumeration: E[(1/K') sum_k c_k grad log nu(y_k)] == grad E[MMD^2_U]."""
    pts = torch.randn(3, 2, generator=gen)            # student support
    teacher = torch.randn(1, 4, 2, generator=gen)
    theta = torch.randn(3, generator=gen, requires_grad=True)
    logp = torch.log_softmax(theta, -1)
    exp_mmd, exp_sur = 0.0, 0.0
    for i in range(3):
        for j in range(3):
            y = torch.stack([pts[i], pts[j]])[None]
            mmd, c = S.mmd2_gaussian(teacher, y, bandwidth=1.0)
            prob = (logp[i] + logp[j]).exp()
            exp_mmd = exp_mmd + prob * mmd[0]
            exp_sur = exp_sur + prob.detach() * (c[0, 0] * logp[i] + c[0, 1] * logp[j]) / 2
    (g_true,) = torch.autograd.grad(exp_mmd, theta, retain_graph=True)
    (g_est,) = torch.autograd.grad(exp_sur, theta)
    assert torch.allclose(g_true, g_est, atol=1e-6)


def test_advantage_estimators():
    c = torch.tensor([[1.0, 2.0, 3.0, 6.0]])
    assert torch.equal(S.advantages(c, "vanilla"), c)
    assert torch.allclose(S.advantages(c, "mean"), c - 3.0)
    loo = S.advantages(c, "loo")
    assert torch.allclose(loo[0, 0], torch.tensor(1.0 - 11.0 / 3))
    norm = S.advantages(c, "loo_norm")
    assert abs(float(norm.std(unbiased=False)) - 1.0) < 1e-4


def test_surrogate_descent_reduces_expected_sw(gen):
    """Following the LOO surrogate moves a categorical student toward the teacher cloud."""
    support = torch.linspace(-3, 3, 13).unsqueeze(-1)
    teacher = (torch.randn(1, 8, 1, generator=gen) * 0.3 + 1.5)
    theta = torch.zeros(13, requires_grad=True)
    opt = torch.optim.SGD([theta], lr=0.5)

    def expected_sw():
        p = torch.softmax(theta.detach(), -1)
        return float((p * (support[:, 0] - 1.5) ** 2).sum())

    before = expected_sw()
    for _ in range(200):
        idx = torch.multinomial(torch.softmax(theta.detach(), -1), 8, replacement=True, generator=gen)
        y = support[idx].unsqueeze(0)
        _, costs = S.sliced_w2(teacher, y, S.random_directions(1, 4, 1, gen))
        adv = S.advantages(costs, "loo_norm")
        logp = torch.log_softmax(theta, -1)[idx].view(1, 8, 1)
        loss = S.segment_surrogate(logp, torch.ones_like(logp, dtype=torch.bool), adv, torch.ones(1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert expected_sw() < 0.3 * before


def test_effective_rank():
    x = torch.randn(2000, 6) * torch.tensor([1, 1, 1, 1, 1e-4, 1e-4])
    assert 3.5 < S.effective_rank(x) < 4.5


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_tree_packed_forward_equals_separate_forwards(impl, gen):
    from chop.tiny import tiny_qwen3

    model = tiny_qwen3(attn_implementation=impl)
    main = torch.randint(3, 96, (30,), generator=gen)
    branches = [(12, torch.randint(3, 96, (5,), generator=gen)),
                (12, torch.randint(3, 96, (5,), generator=gen)),
                (25, torch.randint(3, 96, (3,), generator=gen))]
    lay = S.build_tree(main, branches)
    with torch.no_grad():
        out = model(input_ids=lay.input_ids[None], attention_mask=lay.additive_mask(),
                    position_ids=lay.position_ids[None], output_hidden_states=True)
        logits, hidden = out.logits[0], out.hidden_states[-1][0]
        ref_main = model(input_ids=main[None]).logits[0]
        assert torch.allclose(logits[:30], ref_main, atol=1e-5)
        lp, mask = lay.branch_token_logprobs(logits)
        emb = lay.branch_mean_hidden(hidden)
        for b, (a, toks) in enumerate(branches):
            seq = torch.cat([main[:a], toks])[None]
            o = model(input_ids=seq, output_hidden_states=True)
            ref_lp = torch.log_softmax(o.logits[0, a - 1:-1], -1).gather(-1, toks[:, None])[:, 0]
            assert torch.allclose(lp[b, :toks.numel()], ref_lp, atol=1e-5)
            assert torch.allclose(emb[b], o.hidden_states[-1][0, a:].mean(0), atol=1e-5)
    # realised continuation from the main sequence
    lp_r, e_r = S.realised_segment(main, logits, hidden, 12, 5)
    o = model(input_ids=main[None, :17], output_hidden_states=True)
    ref = torch.log_softmax(o.logits[0, 11:16], -1).gather(-1, main[12:17, None])[:, 0]
    assert torch.allclose(lp_r, ref, atol=1e-5)
    assert torch.allclose(e_r, o.hidden_states[-1][0, 12:17].mean(0), atol=1e-5)


def test_tree_gradients_flow_through_shared_prefix(gen):
    from chop.tiny import tiny_qwen3

    model = tiny_qwen3()
    main = torch.randint(3, 96, (12,), generator=gen)
    lay = S.build_tree(main, [(6, torch.randint(3, 96, (4,), generator=gen))])
    logits = model(input_ids=lay.input_ids[None], attention_mask=lay.additive_mask(),
                   position_ids=lay.position_ids[None]).logits[0]
    lp, mask = lay.branch_token_logprobs(logits)
    (lp * mask).sum().backward()
    emb_grad = model.model.embed_tokens.weight.grad
    assert emb_grad[main[:6]].abs().sum() > 0          # prefix tokens receive gradient
    only_after = set(main[6:].tolist()) - set(main[:6].tolist()) - set(lay.input_ids[12:].tolist())
    for tok in only_after:
        assert emb_grad[tok].abs().sum() == 0          # main tokens after the anchor do not

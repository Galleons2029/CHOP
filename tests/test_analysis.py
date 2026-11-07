import numpy as np
import torch

from analysis import (
    agreement,
    alpha,
    auroc,
    embedding_distortion,
    legacy_mlp,
    pi_redundancy,
    reliability_validation,
    reset_rollout,
    snr,
)
from analysis.gradients import last_logprobs
from chop.tiny import tiny_qwen3


def test_alpha_is_one_when_returns_equal_teacher_reward():
    s, t = tiny_qwen3(seed=0), tiny_qwen3(seed=1)
    ctx = [5, 6, 7, 8]
    with torch.no_grad():
        r_t = (last_logprobs(t, ctx) - last_logprobs(s, ctx))
    # one rollout per vocabulary token, outcome = R_T(v): the centred return is R_T minus a constant
    rollouts = [(v, float(r_t[v])) for v in range(r_t.numel())]
    a = alpha.alpha_t(s, t, ctx, rollouts)
    assert a > 0.99


def test_alpha_diag_uses_12_reference_states():
    assert alpha.reference_positions(100) == sorted(set(round(x) for x in np.linspace(0, 99, 12)))
    s, t = tiny_qwen3(seed=0), tiny_qwen3(seed=1)
    rng = np.random.default_rng(0)
    val, per = alpha.alpha_diag(s, t, [5, 6], list(range(10, 40)),
                                lambda ctx, n: [(int(rng.integers(96)), float(rng.random())) for _ in range(n)],
                                n_rollouts=16)
    assert len(per) <= 12 and -1 <= val <= 1


def test_auroc():
    assert auroc.auroc([3, 4], [1, 2]) == 1.0 and auroc.auroc([1], [1]) == 0.5
    t = tiny_qwen3()
    assert np.isfinite(auroc.teacher_sequence_score(t, [5, 6], [7, 8, 9]))


def test_reliability_validation_and_depth():
    rng = np.random.default_rng(0)
    ref = rng.random(500)
    out = reliability_validation.validate(ref + 0.05 * rng.random(500), ref)
    assert out["spearman"] > 0.95 and 0 <= out["F1_confidently_wrong"] <= 1
    curves = reliability_validation.depth_curves({50: [(100, 0.8), (1500, 0.4)]}, max_k=3)
    assert curves["median_rho"]["50"] == [0.8, 0.4, None]


def test_reset_rollout_restores_parameters_and_reduces_kl():
    s, t = tiny_qwen3(seed=0), tiny_qwen3(seed=1)
    before = [p.detach().clone() for p in s.parameters()]
    g = torch.Generator().manual_seed(0)
    norms, reds = [], []
    for c in ([5, 6, 7], [9, 10, 11, 12], [20, 21]):
        n, r = reset_rollout.grad_norm_and_kl_reduction(s, t, c, lr=1e-2, generator=g)
        norms.append(n)
        reds.append(r)
    assert all(torch.equal(a, b) for a, b in zip(before, s.parameters(), strict=True))
    assert all(n > 0 for n in norms)
    res = reset_rollout.run(s, t, {"original": [[5, 6, 7], [9, 10], [20, 21, 22], [30, 31]]}, lr=1e-2)
    assert "r" in res["pearson"]["original"]
    frag = reset_rollout.random_prefix([[1, 2], [3]], 5, np.random.default_rng(0))
    assert len(frag) == 5


def test_shapley_efficiency_and_mi():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(20, 11, generator=g)
    variants = {k: torch.randn(20, 11, generator=g) for k in ("a", "b", "c")}
    phi = pi_redundancy.shapley_kappa(base, variants)
    total = pi_redundancy.kappa_subset(base, variants, ("a", "b", "c"))
    assert torch.allclose(sum(phi.values()), total, atol=1e-5)
    x = np.array([True, False] * 50)
    assert abs(pi_redundancy.mutual_information(x, x) - np.log(2)) < 1e-9
    assert pi_redundancy.mutual_information(x, np.ones(100, bool)) == 0.0


def test_snr_and_embedding_distortion():
    assert abs(snr.snr_bound(1.0, 0.74, 0.41) - 0.74 / 0.41) < 1e-9 and snr.snr_bound(0.0, 0.74, 0.41) == 0.0
    d = np.random.default_rng(0).random((100, 32)) + 1
    assert 0 < snr.c_s(d) < 1
    a = np.random.default_rng(1).normal(size=(8, 4))
    assert embedding_distortion.w2_equal(a, a[::-1]) < 1e-9
    assert embedding_distortion.first_token_tv([[1], [2]], [[1], [1]]) == 0.5
    est = embedding_distortion.estimate_le([(a, a + 1.0, [[1]] * 8, [[2]] * 8)])
    assert abs(est["L_e"] - 2.0) < 1e-9


def test_agreement_statistics(tmp_path):
    assert agreement.cohen_kappa([1, 0, 1, 1], [1, 0, 1, 1]) == 1.0
    r = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
                  [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    assert abs(agreement.fleiss_kappa(r) - 0.210) < 1e-3      # Fleiss (1971) worked example
    n = agreement.export_states([{"context": f"c{i}"} for i in range(50)], np.arange(50), tmp_path / "a.jsonl")
    assert n == 10


def test_legacy_mlp_learns_kappa():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(400, 8, generator=g)
    kappa = torch.sigmoid(h[:, 0])
    m = legacy_mlp.fit(h, kappa, epochs=300, lr=1e-2)
    assert legacy_mlp.route_stability(m(h).detach(), kappa) > 0.9


def test_run_analysis_cli_on_tiny_config(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_analysis", "scripts/run_analysis.py")
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)
    base = ["--config", "configs/smoke/tiny_math.yaml", "--set", f"output_dir={tmp_path}", "--out", str(tmp_path / "a"),
            "--max-tokens", "8"]
    ra.main(["calibrate", *base, "--states", "8", "--rollouts", "2"])
    assert (tmp_path / "a" / "calibration_states.pt").exists()
    ra.main(["alpha", *base, "--trajectories", "2", "--rollouts", "4"])
    ra.main(["auroc", *base, "--trajectories", "4"])
    ra.main(["reset", *base, "--states", "3"])
    assert (tmp_path / "a" / "reset_rollout.json").exists()

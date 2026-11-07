import numpy as np
import torch

from chop.config import ReliabilityConfig, load_config
from chop.monitors import CollapseGuard, fit_lipschitz
from chop.reliability import calibrate_lambdas, expected_calibration_error, reliability, spearman
from chop.truncation import chunk_inputs, needs_truncation, rolling_chunks


def test_calibration_recovers_ratio():
    rng = np.random.default_rng(0)
    h, tv = rng.uniform(0, 3, 1024), rng.uniform(0, 1, 1024)
    ref = np.exp(-(0.5 * h + 1.5 * tv)) + rng.normal(0, 0.01, 1024)
    cal = calibrate_lambdas(h, tv, ref)
    assert abs(cal.lambda_tv / cal.lambda_entropy - 3.0) < 0.3
    assert cal.spearman > 0.95
    rho = reliability(torch.tensor(h), torch.tensor(tv),
                      ReliabilityConfig(lambda_entropy=cal.lambda_entropy, lambda_tv=cal.lambda_tv))
    assert spearman(rho.numpy(), ref) > 0.95


def test_ece_zero_for_identity():
    x = np.linspace(0, 1, 1000)
    assert expected_calibration_error(x, x) < 1e-12


def test_rolling_chunks_respect_window():
    prompt, resp, window = 100, 1000, 400
    chunks = rolling_chunks(prompt, resp, window, chunk=128)
    assert chunks[0].gen_start == 0 and chunks[-1].gen_end == resp
    for c in chunks:
        assert prompt + (c.gen_end - c.ctx_start) <= window
    assert needs_truncation(prompt, resp, window)
    no_trunc = rolling_chunks(10, 50, 400, chunk=16)
    assert all(c.ctx_start == 0 for c in no_trunc)
    ids, first = chunk_inputs(list(range(prompt)), list(range(resp)), chunks[-1])
    assert len(ids) <= window and ids[first] == chunks[-1].gen_start


def test_collapse_guard():
    g = CollapseGuard(threshold=200, window=3)
    assert not g.observe(300)
    assert g.observe(150)
    assert g.observe(250)
    assert g.observe(260)               # 150 is still inside the trailing window
    assert not g.observe(270)


def test_fit_lipschitz():
    s = np.linspace(0.1, 1, 100)
    assert abs(fit_lipschitz(1.4 * s, s) - 1.4) < 1e-9


def test_config_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("method:\n  segment:\n    num_anchors: 4\ntrain:\n  steps: 10\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nname: c\ntrain:\n  steps: 20\n")
    cfg = load_config(tmp_path / "child.yaml", ["method.beta_outcome=0.5"])
    assert cfg.name == "c" and cfg.train.steps == 20 and cfg.method.segment.num_anchors == 4
    assert cfg.method.beta_outcome == 0.5 and cfg.method.router.window == 8192

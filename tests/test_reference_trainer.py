import math

import pytest

from chop.config import load_config
from chop.reference_trainer import ReferenceTrainer

SMOKE = "configs/smoke/tiny_math.yaml"
pytestmark = pytest.mark.slow


def _trainer(tmp_path, *overrides):
    cfg = load_config(SMOKE, [f"output_dir={tmp_path}", *overrides])
    return ReferenceTrainer(cfg)


def test_rollout_conditioning_matches_scoring(tmp_path):
    """Greedy rollouts under rolling truncation are reproduced token-by-token by the scoring pass."""
    tr = _trainer(tmp_path, "train.temperature=0.0")
    trajs = tr.rollout(tr.task.train_instances()[:2])
    for t in trajs:
        assert any(c.ctx_start > 0 for c in t.chunks), "smoke config must exercise truncation"
        sc = tr.score(t, need_pi=True)
        assert sc.student_logits.argmax(-1).tolist() == t.response


def test_step_hits_all_cells_and_is_finite(tmp_path):
    tr = _trainer(tmp_path)
    rec = [tr.train_step() for _ in range(2)][-1]
    assert math.isfinite(rec["loss"])
    for cell in "ABCD":
        assert rec[cell] > 0
    assert rec["num_anchors"] > 0 and rec["segment_distance"] > 0


def test_segment_branch_alone_produces_gradient(tmp_path):
    tr = _trainer(tmp_path, "method.router.rho_gate=all_segment", "method.beta_outcome=0.0")
    trajs = tr.rollout(tr.task.train_instances()[:2])
    tr.opt.zero_grad()
    out = tr._chop_step(trajs)
    grad = sum(p.grad.abs().sum() for p in tr.student.parameters() if p.grad is not None)
    assert out.metrics["segment"] == 1.0 and float(grad) > 0


def test_collapse_guard_reverts_to_token_branch(tmp_path):
    tr = _trainer(tmp_path, "method.segment.rank_fallback_threshold=1e9")
    rec = tr.train_step()
    rec = tr.train_step()
    assert rec.get("collapse_fallback") == 1.0


@pytest.mark.parametrize("overrides", [
    ["method.name=opd", "method.router.rho_gate=none", "method.router.kappa_gate=none",
     "method.token.estimator=sampled"],
    ["method.name=topk_lsm", "method.router.rho_gate=none", "method.router.kappa_gate=none",
     "method.token.support=teacher", "method.token.topk=32", "method.token.mask_special_tokens=true"],
    ["method.name=opsd", "method.router.rho_gate=none", "method.router.kappa_gate=always",
     "method.token.estimator=sampled"],
    ["method.name=gkd", "method.router.rho_gate=none", "method.router.kappa_gate=none",
     "method.token.estimator=gkd"],
    ["method.name=reopold", "method.reopold_switch_step=1"],
    ["method.name=grpo", "method.grpo_group_size=2"],
    ["method.name=li2026", "method.sft_warmup_prompts=2", "method.router.rho_gate=none",
     "method.router.kappa_gate=none", "method.token.estimator=sampled"],
    ["method.segment.encoder=frozen_student"],
    ["method.segment.encoder=segment_only"],
    ["method.segment.anchor_source=student"],
    ["method.segment.distance=mmd"],
    ["method.segment.anchor_cache=true"],
    ["method.segment.estimator=vanilla"],
    ["method.router.stratum=[0.0,0.3]"],
    ["method.router.three_cell=true"],
    ["method.pi.interface=prompt"],
    ["method.pi.content=shuffled"],
    ["method.pi.content=random_adapter"],
    ["method.reliability.variant=free"],
])
def test_methods_and_ablations_run(tmp_path, overrides):
    tr = _trainer(tmp_path, "train.steps=2", *overrides)
    tr.run()
    lines = (tr.log_path).read_text().strip().splitlines()
    assert lines

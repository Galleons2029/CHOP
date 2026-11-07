"""End-to-end CHOP step through the driver with emulated ranks (no Ray / verl needed)."""

import math

import pytest
import torch

from chop.config import load_config
from chop.monitors import CollapseGuard
from chop.reference_trainer import ReferenceTrainer
from chop.router import CellRouter
from chop_verl import compute as C
from chop_verl import driver as Dr
from chop_verl.local import LocalWorkers

pytestmark = pytest.mark.slow


def _env(tmp_path, *overrides):
    cfg = load_config("configs/smoke/tiny_math.yaml", [f"output_dir={tmp_path}", *overrides])
    tr = ReferenceTrainer(cfg)
    trajs = tr.rollout(tr.task.train_instances()[:4])
    models = C.ChopModels(tr.student, tr.teacher.model, tr.sources, pad_id=0, eos_id=tr.tok.eos_token_id)
    payloads = [Dr.build_payload(str(i), t.s_prompt, t.response, [1] * len(t.response), t.reward,
                                 {"teacher_prompt_ids": t.t_prompt, "pi_prefix_ids": t.pi_prefix}, cfg.method,
                                 cfg.model.native_context, cfg.train.rollout_chunk) for i, t in enumerate(trajs)]
    workers = LocalWorkers(models, cfg.method, torch.optim.AdamW(tr.student.parameters(), lr=1e-4))
    router = CellRouter(cfg.method.router, seed=0)
    return tr, cfg, payloads, workers, router


def test_payload_chunks_match_reference_plan(tmp_path):
    tr, cfg, payloads, *_ = _env(tmp_path)
    trajs = tr.rollout(tr.task.train_instances()[:1])
    t = trajs[0]
    p = Dr.build_payload("x", t.s_prompt, t.response, [1] * len(t.response), t.reward,
                         {"teacher_prompt_ids": t.t_prompt, "pi_prefix_ids": t.pi_prefix}, cfg.method,
                         cfg.model.native_context, cfg.train.rollout_chunk)
    assert p.chunks == t.chunks


def test_sharding_balances_tokens(tmp_path):
    _, _, payloads, *_ = _env(tmp_path)
    shards = Dr.shard(payloads, 3)
    assert sorted(p.uid for s in shards for p in s) == sorted(p.uid for p in payloads)
    assert all(len(s) >= 1 for s in shards)


@pytest.mark.parametrize("world", [1, 2])
def test_chop_step_runs_and_updates(tmp_path, world):
    tr, cfg, payloads, workers, router = _env(tmp_path)
    Dr.chop_step(workers, router, CollapseGuard(0.0), payloads, cfg.method, step=0, world=world, need_pi=True)
    before = {n: q.detach().clone() for n, q in tr.student.named_parameters()}
    m = Dr.chop_step(workers, router, CollapseGuard(0.0), payloads, cfg.method, step=1, world=world, need_pi=True)
    assert math.isfinite(m["chop/loss"]) and m["chop/anchors"] > 0
    for cell in "ABCD":
        assert m[f"chop/cell_{cell}"] > 0
    assert any(not torch.equal(before[n], q) for n, q in tr.student.named_parameters())


def test_world_size_does_not_change_the_update(tmp_path):
    """Sharding is a pure work partition: with deterministic sampling (greedy segments, MMD) the summed
    gradient is identical for 1 and 3 ranks.  (With stochastic sampling the per-rank generators differ.)"""
    grads = []
    for world in (1, 3):
        tr, cfg, payloads, workers, router = _env(tmp_path, "method.segment.student_temperature=0",
                                                  "method.segment.teacher_temperature=0",
                                                  "method.segment.distance=mmd", "method.router.min_history=1")
        workers.opt = torch.optim.SGD(tr.student.parameters(), lr=0.0)
        workers.grad_clip = None
        Dr.chop_step(workers, router, CollapseGuard(0.0), payloads, cfg.method, step=0, world=world, need_pi=True)
        grads.append({n: q.grad.clone() for n, q in tr.student.named_parameters() if q.grad is not None})
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], atol=1e-5), n


@pytest.mark.parametrize("name", ["reopold", "grpo"])
def test_rl_style_methods_run(tmp_path, name):
    tr, cfg, payloads, workers, router = _env(tmp_path, f"method.name={name}")
    m = Dr.chop_step(workers, router, CollapseGuard(0.0), payloads, cfg.method, step=0, world=2, need_pi=False,
                     group_of={p.uid: "g" for p in payloads})
    assert math.isfinite(m["chop/loss"])


def test_collapse_fallback_drops_segments(tmp_path):
    tr, cfg, payloads, workers, router = _env(tmp_path)
    m = Dr.chop_step(workers, router, CollapseGuard(1e9), payloads, cfg.method, step=0, world=2, need_pi=True)
    assert m["chop/collapse_fallback"] == 1.0

"""The scalable per-rank path (chop_verl.compute) reproduces the dense reference trainer."""

import pytest
import torch

from chop.config import load_config
from chop.losses import SegmentTerms, StateBatch, chop_loss, segment_advantages
from chop.reference_trainer import ReferenceTrainer
from chop_verl import compute as C

pytestmark = pytest.mark.slow


def _setup(tmp_path, *overrides):
    cfg = load_config("configs/smoke/tiny_math.yaml", [f"output_dir={tmp_path}", "train.batch_size=2", *overrides])
    tr = ReferenceTrainer(cfg)
    trajs = tr.rollout(tr.task.train_instances()[:2])
    models = C.ChopModels(tr.student, tr.teacher.model, tr.sources, pad_id=0, eos_id=tr.tok.eos_token_id)
    payloads = [C.TrajPayload(str(i), t.s_prompt, t.t_prompt, t.response, [1] * len(t.response), t.reward,
                              t.pi_prefix, t.chunks) for i, t in enumerate(trajs)]
    return tr, trajs, models, payloads


@pytest.mark.parametrize("estimator", ["topk", "sampled"])
def test_scoring_and_update_match_reference(tmp_path, estimator):
    tr, trajs, models, payloads = _setup(tmp_path, f"method.token.estimator={estimator}")
    cfg = tr.cfg.method
    # --- phase B: identical statistics
    ref_scored = [tr.score(t, need_pi=True) for t in trajs]
    for p, ref in zip(payloads, ref_scored, strict=True):
        st, rho, _ = C.score_traj(models, p, cfg, need_pi=True)
        assert torch.allclose(st.entropy, ref.entropy, atol=1e-4)
        assert torch.allclose(st.tv, ref.tv, atol=1e-4)
        assert torch.allclose(st.kappa, ref.kappa, atol=1e-4)
        assert torch.allclose(rho, ref.rho, atol=1e-4)
    # prime the router with a stream so both segment and token states occur
    tr.router.update(torch.rand(512), torch.rand(512))
    routes = tr.route(trajs, ref_scored)
    anchors = [tr.anchors(t, r) for t, r in zip(trajs, routes, strict=True)]
    assert sum(len(a) for a in anchors) > 0
    segs = [[C.AnchorSegments(C.AnchorSpec(a.pos, a.gate, a.weight), a.chunk, a.student_segs, a.teacher_segs)
             for a in al] for al in anchors]
    # --- phase D: identical embeddings
    for t, al, sg, p in zip(trajs, anchors, segs, payloads, strict=True):
        if not al:
            continue
        te, se = tr.embed_anchors(t, al)
        _, _, te2, se2 = C.segment_costs(models, p, sg, cfg)
        assert torch.allclose(te, te2, atol=1e-4) and torch.allclose(se, se2, atol=1e-4)
    embs = [tr.embed_anchors(t, a) for t, a in zip(trajs, anchors, strict=True) if a]
    _, adv = segment_advantages(torch.cat([e[0] for e in embs]), torch.cat([e[1] for e in embs]), cfg,
                                torch.Generator().manual_seed(0))
    n_total = sum(len(t.response) for t in trajs)
    # --- phase E, reference (dense)
    tr.student.zero_grad()
    cursor = 0
    for t, sc, r, a in zip(trajs, ref_scored, routes, anchors, strict=True):
        logits, lp, mk = tr._grad_pass(t, a)
        seg = None
        if a:
            seg = SegmentTerms(lp, mk, torch.tensor([x.weight for x in a]), adv=adv[cursor:cursor + len(a)])
        states = StateBatch(logits, torch.tensor(t.response), sc.teacher_base, sc.teacher_pi,
                            torch.full((len(t.response),), t.reward))
        chop_loss(states, r, seg, cfg, normalizer=n_total).loss.backward()
        cursor += len(a)
    ref_grads = {n: q.grad.clone() for n, q in tr.student.named_parameters() if q.grad is not None}
    # --- phase E, scalable (sparse targets, one tree forward per chunk)
    tr.student.zero_grad()
    cursor = 0
    for p, r, sg in zip(payloads, routes, segs, strict=True):
        _, _, tgt = C.score_traj(models, p, cfg, need_pi=True)
        C.update_traj(models, p, r, sg, adv[cursor:cursor + len(sg)], tgt, cfg, n_total)
        cursor += len(sg)
    for n, q in tr.student.named_parameters():
        if n in ref_grads:
            assert torch.allclose(q.grad, ref_grads[n], atol=1e-4, rtol=1e-3), n


def test_tree_decoded_segments_have_expected_shapes(tmp_path):
    tr, trajs, models, payloads = _setup(tmp_path)
    specs = [C.AnchorSpec(pos, pos % 2 == 0, 1.0) for pos in (3, 20)]
    segs = C.sample_segments(models, payloads[0], specs, tr.cfg.method, torch.Generator().manual_seed(0))
    s = tr.cfg.method.segment
    for a in segs:
        assert len(a.student) == s.num_student_segments - 1 and len(a.teacher) == s.num_anchors
        assert all(1 <= len(x) <= s.segment_length for x in a.student + a.teacher)
    assert C.student_forwards("sample", payloads[:1], tr.cfg.method, [segs]) > 0


def test_lockstep_decode_issues_fixed_number_of_forwards(tmp_path):
    from chop.engine import tree_decode
    from chop.tiny import tiny_qwen3

    m = tiny_qwen3()
    calls = []
    h = m.register_forward_hook(lambda *_: calls.append(1))
    tree_decode(m, list(range(3, 13)), [5, 10], 2, 6, temperature=0.0, eos_id=None, lockstep=True)
    h.remove()
    assert len(calls) == 6

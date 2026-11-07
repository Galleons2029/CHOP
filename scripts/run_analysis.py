#!/usr/bin/env python3
"""Offline analyses on a (student, teacher, task) triple built from an experiment config.

  python scripts/run_analysis.py calibrate --config configs/base/math.yaml --states 1024 --rollouts 16
  python scripts/run_analysis.py alpha     --config ... --trajectories 32 --rollouts 64
  python scripts/run_analysis.py auroc     --config ... --trajectories 256
  python scripts/run_analysis.py reset     --config ... --states 400
Outputs go to results/analysis/<subcommand>.json. Model paths, PI adapters and the task come from the
config; ``model.student`` may point to a training checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from chop import divergences as D
from chop.config import load_config
from chop.reference_trainer import ReferenceTrainer
from chop.reliability import calibrate_lambdas, reliability
from chop.rollout import generate_response, sample_segment


def teacher_rollout_fn(tr: ReferenceTrainer, inst, max_tokens: int):
    """Teacher continuations from a student state, scored by the task reward (for rho_ref and R*)."""
    teacher = tr.teacher.model

    def fn(ctx: list[int], n: int) -> list[tuple[int, float]]:
        out = []
        with (teacher.disable_adapter() if hasattr(teacher, "peft_config") else torch.no_grad()):
            for _ in range(n):
                cont = sample_segment(teacher, ctx, max_tokens, 0.7, 0.95, tr.t_tok.eos_token_id, tr.gen)
                prompt_len = len(tr._prompt_ids(inst)[0])
                text = tr.tok.decode(ctx[prompt_len:] + cont)
                out.append((cont[0] if cont else -1, tr.task.reward(inst, text)))
        return [(v, r) for v, r in out if v >= 0]

    return fn


def sample_states(tr: ReferenceTrainer, n_states: int, seed: int = 0):
    rng = random.Random(seed)
    pool = tr.task.train_instances()
    states = []
    while len(states) < n_states:
        inst = pool[rng.randrange(len(pool))]
        traj = tr.rollout([inst])[0]
        if not traj.response:
            continue
        pos = rng.randrange(len(traj.response))
        states.append((inst, traj, pos))
    return states


@torch.no_grad()
def state_stats(tr: ReferenceTrainer, traj, pos: int) -> tuple[float, float]:
    s_ctx = traj.s_prompt + traj.response[:pos]
    t_ctx = traj.t_prompt + traj.response[:pos]
    p = tr.student(input_ids=torch.tensor([s_ctx])).logits[0, -1]
    q = tr.teacher.model(input_ids=torch.tensor([t_ctx])).logits[0, -1]
    return float(D.entropy(q)), float(D.total_variation(q, p))


def cmd_calibrate(tr, args) -> dict:
    states = sample_states(tr, args.states, args.seed)
    h, tv, ref = [], [], []
    for inst, traj, pos in states:
        e, t = state_stats(tr, traj, pos)
        rolls = teacher_rollout_fn(tr, inst, args.max_tokens)(traj.s_prompt + traj.response[:pos], args.rollouts)
        h.append(e)
        tv.append(t)
        ref.append(float(np.mean([r for _, r in rolls])) if rolls else 0.0)
    cal = calibrate_lambdas(h, tv, ref)
    cfg = tr.cfg.method.reliability
    cfg.lambda_entropy, cfg.lambda_tv = cal.lambda_entropy, cal.lambda_tv
    rho = reliability(torch.tensor(h), torch.tensor(tv), cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"rho": rho, "kappa": None}, out / "calibration_states.pt")
    return {"lambda_entropy": cal.lambda_entropy, "lambda_tv": cal.lambda_tv, "spearman_vs_ref": cal.spearman,
            "states": len(states), "entropy_max": max(h), "tv_max": max(tv)}


def cmd_alpha(tr, args) -> dict:
    from analysis.alpha import alpha_diag

    vals = []
    for inst in random.Random(args.seed).sample(tr.task.train_instances(), args.trajectories):
        traj = tr.rollout([inst])[0]
        a, _ = alpha_diag(tr.student, tr.teacher.model, traj.s_prompt, traj.response,
                          teacher_rollout_fn(tr, inst, args.max_tokens), args.rollouts)
        vals.append(a)
    return {"alpha_diag": float(np.nanmean(vals)), "per_trajectory": vals}


def cmd_auroc(tr, args) -> dict:
    from analysis.auroc import auroc, teacher_sequence_score

    pos, neg = [], []
    for inst in random.Random(args.seed).sample(tr.task.train_instances(), args.trajectories):
        traj = tr.rollout([inst])[0]
        if traj.response:
            s = teacher_sequence_score(tr.teacher.model, traj.t_prompt, traj.response)
            (pos if traj.reward > 0 else neg).append(s)
    return {"sequence_auroc": auroc(pos, neg), "successes": len(pos), "failures": len(neg)}


def cmd_reset(tr, args) -> dict:
    from analysis.reset_rollout import random_prefix, run

    states = sample_states(tr, args.states * 3, args.seed)
    rho = []
    for _, traj, pos in states:
        e, t = state_stats(tr, traj, pos)
        rho.append(float(reliability(torch.tensor(e), torch.tensor(t), tr.cfg.method.reliability)))
    lo, hi = np.quantile(rho, [0.3, 0.7])
    low = [s for s, r in zip(states, rho, strict=True) if r <= lo][:args.states]
    high = [s for s, r in zip(states, rho, strict=True) if r >= hi][:args.states]
    frags = [traj.response[i:i + 16] for _, traj, _ in states for i in range(0, len(traj.response), 16)]
    rng = np.random.default_rng(args.seed)
    teacher = tr.teacher.model
    cont = []
    for _inst, traj, pos in low:
        with (teacher.disable_adapter() if hasattr(teacher, "peft_config") else torch.no_grad()):
            prefix = generate_response(teacher, traj.t_prompt, max(pos, 1), 0.7, 0.95, None, generator=tr.gen)
        cont.append(traj.s_prompt + prefix[:pos])
    conditions = {
        "low_rho": [t.s_prompt + t.response[:p] for _, t, p in low],
        "high_rho": [t.s_prompt + t.response[:p] for _, t, p in high],
        "teacher_continuation": cont,
        "random_prefix": [t.s_prompt + random_prefix(frags, p, rng) for _, t, p in low],
    }
    res = run(tr.student, tr.teacher.model, conditions, lr=args.lr, out=Path(args.out) / "reset_rollout.json",
              seed=args.seed)
    return {"pearson": res["pearson"], "rho_cutoffs": [float(lo), float(hi)]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["calibrate", "alpha", "auroc", "reset"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="config override key=value")
    ap.add_argument("--states", type=int, default=1024)
    ap.add_argument("--trajectories", type=int, default=32)
    ap.add_argument("--rollouts", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    tr = ReferenceTrainer(cfg)
    res = {"calibrate": cmd_calibrate, "alpha": cmd_alpha, "auroc": cmd_auroc, "reset": cmd_reset}[args.command](
        tr, args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.command}.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "per_trajectory"}, indent=2, default=float))


if __name__ == "__main__":
    main()

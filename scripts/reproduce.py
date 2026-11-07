#!/usr/bin/env python3
"""Expand paper tables into runs: commands, seeds, and the estimated compute.

  python scripts/reproduce.py --table tab:swebench_main --dry-run      # print the commands
  python scripts/reproduce.py --table all --dry-run --summary           # per-table compute vs tab:budget
  python scripts/reproduce.py --table tab:anisotropy --emit outputs/sh  # one shell script per run and seed

Each training run writes its resolved experiment config to outputs/runs/<name>/experiment.yaml. chop_verl
reads that file through actor_rollout_ref.chop.method_file, so the table config is the single source of
truth for the method.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from chop.config import ExperimentConfig, load_config, to_yaml

PAPER_BUDGET_H100_DAYS = 3560


def configs(table: str) -> list[ExperimentConfig]:
    out = []
    for p in sorted(Path("configs/experiments").rglob("*.yaml")):
        c = load_config(p)
        if table == "all" or c.table == table:
            out.append(c)
    return out


def h100_days(c: ExperimentConfig) -> float:
    if c.eval_only or not c.est_hours_per_seed:
        return 0.0
    return c.gpus * c.est_hours_per_seed * len(c.seeds) / 24.0


def ckpt_dir(name: str, seed: int) -> str:
    return f"outputs/runs/{name}/seed{seed}"


def train_cmd(c: ExperimentConfig, seed: int) -> list[str]:
    run = f"outputs/runs/{c.name}"
    base = "chop_math" if c.kind == "math" else "chop_swe"
    data = []
    if c.kind == "swe" and c.data.manifest and "strict" in c.data.manifest:
        data.append("data.train_files=data/processed/swe/swe_train_strict.parquet")
    cmds = [f"mkdir -p {run} && python - <<'PY'\nfrom chop.config import load_config, to_yaml\n"
            f"open('{run}/experiment.yaml','w').write(to_yaml(load_config('{c._path}')))\nPY"]
    if c.method.sft_warmup_prompts:
        cmds.append(f"bash scripts/li2026_cold_start.sh {c.model.student} {c.model.teacher} "
                    f"{c.method.sft_warmup_prompts} {run}/sft")
    student = f"{run}/sft/hf" if c.method.sft_warmup_prompts else c.model.student
    cmds.append(" ".join([
        f"python -m chop_verl.main --config-name {base}",
        f"actor_rollout_ref.chop.method_file={run}/experiment.yaml",
        f"actor_rollout_ref.model.path={student}", f"actor_rollout_ref.chop.teacher_path={c.model.teacher}",
        f"data.max_response_length={c.train.max_response_tokens}", f"trainer.total_training_steps={c.train.steps}",
        f"actor_rollout_ref.actor.optim.lr={c.optim.lr}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={c.optim.warmup_steps}",
        f"data.train_batch_size={c.train.batch_size * c.train.grad_accum}", f"data.seed={seed}",
        f"actor_rollout_ref.chop.seed={seed}", f"trainer.n_gpus_per_node={c.gpus}",
        f"trainer.experiment_name={c.name}_s{seed}", f"trainer.default_local_dir={ckpt_dir(c.name, seed)}", *data]))
    cmds.append(f"python -m verl.model_merger merge --backend fsdp --local_dir {ckpt_dir(c.name, seed)}/"
                f"global_step_{c.train.steps}/actor --target_dir {ckpt_dir(c.name, seed)}/hf")
    return cmds


def eval_cmd(c: ExperimentConfig, seed: int, by_name: dict[str, ExperimentConfig]) -> list[str]:
    ev = c.eval or {}
    src = ev.get("checkpoint_of")
    if c.eval_only and src:
        ckpt = by_name[src].model.student if by_name[src].eval_only else f"{ckpt_dir(src, seed)}/hf"
    elif c.eval_only:
        ckpt = c.model.student
    else:
        ckpt = f"{ckpt_dir(c.name, seed)}/hf"
    if c.kind == "math":
        benches = " ".join(c.data.eval)
        return [f"python -m eval.math_eval --model {ckpt} --benchmarks {benches} --cap {c.train.max_response_tokens}"
                f" --seed {seed} --out results/math/{c.name}/seed{seed}"]
    harness = ev.get("harness", "openhands")
    if harness == "teacher_audit":
        return [f"python -m eval.teacher_audit --teacher {c.model.teacher} --adapters outputs/adapters/swe "
                f"--pi {ev.get('pi') or 'none'} --seed {seed} --out results/swe/{c.name}/seed{seed}"]
    if harness == "swe_agent":
        return [f"bash eval/swe_agent.sh {ckpt} {c.name}"]
    turns = ev.get("max_turns", 50)
    return [f"bash eval/swe_openhands.sh {ckpt} {c.name} {turns} {seed}",
            "# or, with this repository's CodeAct harness:",
            f"# python -m eval.swe_eval --model {ckpt} --max-turns {turns} --seed {seed} "
            f"--out results/swe/{c.name}/seed{seed}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--emit", default=None, help="directory for one shell script per run and seed")
    args = ap.parse_args()
    cfgs = []
    for p in sorted(Path("configs/experiments").rglob("*.yaml")):
        c = load_config(p)
        c._path = str(p)
        if args.table in ("all", c.table):
            cfgs.append(c)
    all_cfgs = {c.name: c for c in (load_config(p) for p in Path("configs/experiments").rglob("*.yaml"))}
    per_table: dict[str, float] = defaultdict(float)
    for c in cfgs:
        per_table[c.table] += h100_days(c)
        for seed in c.seeds if not c.eval_only else [0, 1, 2][:max(1, len(c.seeds))]:
            cmds = ([] if c.eval_only else train_cmd(c, seed)) + eval_cmd(c, seed, all_cfgs)
            if args.dry_run and not args.summary:
                print(f"### {c.table} | {c.row} | seed {seed}" + (f"\n# NOTE: {c.notes}" if c.notes else ""))
                print("\n".join(cmds) + "\n")
            if args.emit:
                d = Path(args.emit)
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{c.name}_s{seed}.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                                                        + (f"# NOTE: {c.notes}\n" if c.notes else "")
                                                        + "\n".join(cmds) + "\n")
    total = sum(per_table.values())
    if args.summary or not args.dry_run:
        for t, d in sorted(per_table.items()):
            print(f"{t:28s} {d:9.1f} H100-days")
        print(f"{'total':28s} {total:9.1f} H100-days (paper estimate: {PAPER_BUDGET_H100_DAYS}; the paper also counts "
              "PI adapters and calibration references)")
    _ = to_yaml  # re-exported for the emitted scripts


if __name__ == "__main__":
    main()

"""SWE-bench Verified pass@1 with the repository's CodeAct harness (token-in/token-out, vLLM).

  python -m eval.swe_eval --model outputs/ckpt --instances data/manifests/verified_500.txt --seed 0 \
      --max-turns 50 --out results/swe/<run>/seed0

Writes predictions.jsonl (SWE-bench format) and episodes.jsonl. It then grades every prediction with the
official harness (envs/swe/grading.py) into per_instance.jsonl. For the turn-cap sweep (tab:turncap),
re-run the same checkpoints with --max-turns 10 / 25 / 50. The headline table is produced with the
OpenHands harness (eval/swe_openhands.sh); comparing both harnesses on the same checkpoints separates
harness effects from training effects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


async def run_all(instances: list[dict], make_generate, renderer, make_sandbox, max_turns: int, time_limit: float,
                  window: int, chunk: int, budget: int, concurrency: int = 32) -> list[dict]:
    from envs.swe.codeact import run_episode

    sem = asyncio.Semaphore(concurrency)

    async def one(inst: dict) -> dict:
        async with sem:
            sb = await asyncio.to_thread(make_sandbox, inst)
            try:
                res = await run_episode(make_generate(inst), renderer, sb, inst["problem_statement"],
                                        base_commit=inst.get("base_commit"), max_turns=max_turns,
                                        time_limit=time_limit, window=window, chunk=chunk, budget=budget)
            finally:
                await asyncio.to_thread(sb.close)
            return {"instance_id": inst["instance_id"], "model_patch": res.patch, "turns": res.turns,
                    "finished": res.finished, "timed_out": res.timed_out,
                    "generated_tokens": sum(res.response_mask), "trajectory_tokens": len(res.response_ids)}

    return await asyncio.gather(*(one(x) for x in instances))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--instances", default="data/manifests/verified_500.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=50)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--window", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--budget", type=int, default=60000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from envs.swe.codeact import ChatRenderer
    from envs.swe.data import rows
    from envs.swe.grading import grade, instance_image
    from envs.swe.sandbox import DockerSandbox
    from eval.backends import VLLMAsyncBackend

    keep = set(Path(args.instances).read_text().split())
    data = [r for r in rows("verified") if r["instance_id"] in keep]
    tok = AutoTokenizer.from_pretrained(args.model)
    backend = VLLMAsyncBackend(args.model, max_model_len=args.window, seed=args.seed)
    eps = asyncio.run(run_all(data, lambda _inst: backend.generate, ChatRenderer(tok),
                              lambda inst: DockerSandbox(instance_image(inst["instance_id"])), args.max_turns,
                              args.time_limit, args.window, args.chunk, args.budget, args.concurrency))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.jsonl").open("w") as f, (out / "predictions.jsonl").open("w") as g:
        for e in eps:
            f.write(json.dumps(e) + "\n")
            g.write(json.dumps({"instance_id": e["instance_id"], "model_name_or_path": args.model,
                                "model_patch": e["model_patch"]}) + "\n")
    if args.no_grade:
        return
    by_id = {r["instance_id"]: r for r in data}
    with (out / "per_instance.jsonl").open("w") as f:
        for e in eps:
            rep = grade(by_id[e["instance_id"]], e["model_patch"], f"{out.name}-{args.seed}")
            f.write(json.dumps({"instance_id": e["instance_id"], "resolved": rep["resolved"],
                                "trajectory_tokens": e["trajectory_tokens"]}) + "\n")


if __name__ == "__main__":
    main()

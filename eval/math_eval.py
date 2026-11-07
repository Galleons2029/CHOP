"""AIME 2024/2025 and AMC 2023 avg@16 under a cumulative token cap with rolling left truncation.

  python -m eval.math_eval --model outputs/ckpt --benchmarks aime24 aime25 amc23 --cap 60000 --seed 0 \
      --out results/math/<run>

Every sample follows the same chunk plan as training rollouts (chop.truncation.rolling_chunks).
Samples are generated chunk by chunk in batched rounds, and a sample stops at EOS. The output is one
JSONL record per (problem, sample) plus a summary with avg@16 per benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chop.truncation import rolling_chunks
from eval.stats import avg_at_k


def generate_capped(backend, prompts: list[list[int]], k: int, cap: int, window: int, chunk: int, temperature: float,
                    top_p: float, eos_id: int | None, seed: int = 0, plan_lens: list[int] | None = None
                    ) -> list[list[list[int]]]:
    """k samples per prompt, each up to ``cap`` generated tokens under the rolling chunk plan."""
    plan_lens = plan_lens or [len(p) for p in prompts]
    jobs = [(i, j) for i in range(len(prompts)) for j in range(k)]
    resp = {job: [] for job in jobs}
    plans = {i: rolling_chunks(plan_lens[i], cap, window, chunk) for i in range(len(prompts))}
    live = set(jobs)
    rnd = 0
    while live:
        batch = sorted(live)
        ctxs, maxt, seeds = [], [], []
        for i, j in batch:
            r = resp[(i, j)]
            c = next(c for c in plans[i] if c.gen_start <= len(r) < c.gen_end)
            ctxs.append(prompts[i] + r[c.ctx_start:len(r)])
            maxt.append(c.gen_end - len(r))
            seeds.append(seed * 1_000_003 + (i * k + j) * 1009 + rnd)
        outs = backend.generate_batch(ctxs, maxt, temperature, top_p, seeds)
        for (i, j), out, m in zip(batch, outs, maxt, strict=True):
            resp[(i, j)].extend(out)
            if not out or (eos_id is not None and out[-1] == eos_id) or len(out) < m or len(resp[(i, j)]) >= cap:
                live.discard((i, j))
        rnd += 1
    return [[resp[(i, j)][:cap] for j in range(k)] for i in range(len(prompts))]


def evaluate(backend, task, tokenizer, benchmarks: list[str], k: int, cap: int, window: int, chunk: int,
             temperature: float, top_p: float, seed: int, render=None) -> tuple[list[dict], dict]:
    render = render or (lambda text: tokenizer.encode(text))
    records, summary = [], {}
    for bench in benchmarks:
        insts = task.eval_instances(bench)
        prompts = [render(task.prompt(x)) for x in insts]
        outs = generate_capped(backend, prompts, k, cap, window, chunk, temperature, top_p, tokenizer.eos_token_id,
                               seed)
        grid = []
        for inst, samples in zip(insts, outs, strict=True):
            row = []
            for j, ids in enumerate(samples):
                ok = task.reward(inst, tokenizer.decode(ids))
                row.append(ok)
                records.append({"benchmark": bench, "id": inst.id, "sample": j, "correct": ok, "tokens": len(ids)})
            grid.append(row)
        summary[bench] = {f"avg@{k}": avg_at_k(grid), "problems": len(insts)}
    return records, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25", "amc23"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--cap", type=int, default=30000)
    ap.add_argument("--window", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from envs.math import MathTask
    from eval.backends import VLLMBackend

    tok = AutoTokenizer.from_pretrained(args.model)
    backend = VLLMBackend(args.model, max_model_len=args.window, seed=args.seed)

    def render(text):
        return list(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                            tokenize=True))

    records, summary = evaluate(backend, MathTask(), tok, args.benchmarks, args.k, args.cap, args.window, args.chunk,
                                args.temperature, args.top_p, args.seed, render)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps({"args": vars(args), **summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

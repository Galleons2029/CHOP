#!/usr/bin/env python3
"""Teacher rollouts on DAPO-Math-17K prompts for the Li et al. off-policy cold start (SFT data)."""

from __future__ import annotations

import argparse
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from envs.math import MathTask

    task = MathTask()
    insts = random.Random(args.seed).sample(task.train_instances(), args.num_prompts)
    tok = AutoTokenizer.from_pretrained(args.teacher)
    llm = LLM(model=args.teacher, max_model_len=32768, seed=args.seed)
    prompts = [tok.apply_chat_template([{"role": "user", "content": task.prompt(x)}], add_generation_prompt=True,
                                       tokenize=False) for x in insts]
    outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=args.max_tokens))
    rows = [{"prompt": task.prompt(x), "response": o.outputs[0].text, "correct": task.reward(x, o.outputs[0].text)}
            for x, o in zip(insts, outs, strict=True)]
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"{len(rows)} rollouts, {sum(r['correct'] for r in rows)} correct -> {args.out}")


if __name__ == "__main__":
    main()

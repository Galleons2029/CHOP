#!/usr/bin/env python3
"""Write verl parquet files for the math experiments (DAPO-Math-17K train; AIME24/25, AMC23 eval).

Each row carries, in ``extra_info``, what the CHOP driver needs without tokenizers of its own:
  teacher_prompt_ids  the prompt rendered with the *teacher's* chat template
  pi_prefix_ids       {source: ids of the [PI] block} for the configured PI sources
  plan_prompt_len     longest context the rolling chunk plan must reserve on either side
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chop.pi.serialize import format_pi_block
from envs.math.task import DATASETS, MathTask


def build_rows(instances, task, student_render, teacher_render, teacher_encode, sources, data_source: str) -> list:
    rows = []
    for i, inst in enumerate(instances):
        text = task.prompt(inst)
        s_ids = student_render(text)
        t_ids = teacher_render(text)
        blocks = task.pi_blocks(inst)
        prefix = {s: teacher_encode(format_pi_block(s, blocks[s])) for s in sources}
        plan = max(len(s_ids), len(t_ids) + max((len(v) for v in prefix.values()), default=0))
        rows.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": text}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": inst.answer},
            "extra_info": {"index": i, "id": inst.id, "teacher_prompt_ids": t_ids, "pi_prefix_ids": prefix,
                           "plan_prompt_len": plan},
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/math")
    ap.add_argument("--student", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--teacher", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--sources", nargs="*", default=["final_answer"])
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-eval", type=int, default=None)
    args = ap.parse_args()
    import pandas as pd
    from transformers import AutoTokenizer

    s_tok, t_tok = AutoTokenizer.from_pretrained(args.student), AutoTokenizer.from_pretrained(args.teacher)

    def chat(tok):
        return lambda text: list(tok.apply_chat_template([{"role": "user", "content": text}],
                                                         add_generation_prompt=True, tokenize=True))

    task = MathTask(train_limit=args.limit_train)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"dapo_math_17k": task.train_instances()}
    for name in ("aime24", "aime25", "amc23"):
        insts = task.eval_instances(name)
        splits[name] = insts[:args.limit_eval] if args.limit_eval else insts
    for name, insts in splits.items():
        rows = build_rows(insts, task, chat(s_tok), chat(t_tok), lambda x: t_tok.encode(x, add_special_tokens=False),
                          args.sources, f"math_{name}")
        pd.DataFrame(rows).to_parquet(out / f"{name}.parquet")
        print(name, len(rows), "->", out / f"{name}.parquet")
    assert set(DATASETS) >= set(splits)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write verl parquet files for SWE-bench training (filtered train split) and evaluation (Verified).

extra_info per row:
  instance          SWE-bench fields needed by the episode and the grader
  image             instance image (Verified: published swebench images; train: --train-image-map)
  teacher_prompt_ids, pi_prefix_ids{source}, plan_prompt_len  (see scripts/prepare_math.py)
Static PI blocks: gold_patch (p1), test_manifest (p2), call_graph (p4, needs --repos-dir checkouts).
next_file (p3) is per-step and produced by the agent loop; plan_prompt_len reserves its token budget.
For training instances SWE-bench leaves FAIL_TO_PASS / PASS_TO_PASS empty; supply them with
--train-tests (JSON produced by running each instance's test patch in its reconstructed environment).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from chop.pi.serialize import format_pi_block
from envs.swe.codeact import SYSTEM_PROMPT, TOOLS, USER_TEMPLATE
from envs.swe.grading import instance_image
from pi_extraction import sources

INSTANCE_KEYS = ("instance_id", "repo", "base_commit", "patch", "test_patch", "problem_statement", "hints_text",
                 "version", "environment_setup_commit", "FAIL_TO_PASS", "PASS_TO_PASS", "created_at")


def messages(problem: str, workdir: str = "/testbed") -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(workdir=workdir, problem=problem)}]


def call_graph_for(row: dict, repos_dir: Path | None) -> str:
    """p4 at the base commit, extracted from a local clone (``repos_dir/<owner>__<name>``)."""
    if repos_dir is None:
        return ""
    from pi_extraction.call_graph import build_call_graph

    clone = repos_dir / row["repo"].replace("/", "__")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(f"git -C {clone} archive {row['base_commit']} | tar -x -C {tmp}", shell=True, check=True)
        return build_call_graph(tmp, row["problem_statement"])


def build_row(row: dict, render_student, render_teacher, encode_teacher, decode_teacher, source_names: list[str],
              max_tokens: int, call_graph: str = "", image: str | None = None, split: str = "verified") -> dict:
    blocks = {
        "gold_patch": lambda: sources.gold_patch(row["patch"]),
        "test_manifest": lambda: sources.test_manifest(row["FAIL_TO_PASS"], row["PASS_TO_PASS"]),
        "call_graph": lambda: call_graph,
    }
    prefix = {s: encode_teacher(format_pi_block(s, blocks[s](), encode_teacher, decode_teacher, max_tokens))
              for s in source_names if s != "next_file"}
    msgs = messages(row["problem_statement"])
    s_ids, t_ids = render_student(msgs), render_teacher(msgs)
    # the per-step next_file block is bounded by the same token budget (+ its markers)
    per_step = len(encode_teacher(format_pi_block("next_file", ""))) + max_tokens if "next_file" in source_names else 0
    longest = max([len(v) for v in prefix.values()] + [per_step, 0])
    return {
        "data_source": f"swe_{split}",
        "prompt": msgs,
        "ability": "swe",
        "reward_model": {"style": "harness", "ground_truth": ""},
        "extra_info": {"instance": {k: row.get(k) for k in INSTANCE_KEYS}, "image": image,
                       "teacher_prompt_ids": t_ids, "pi_prefix_ids": prefix,
                       "plan_prompt_len": max(len(s_ids), len(t_ids) + longest)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/swe")
    ap.add_argument("--student", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--sources", nargs="*", default=["gold_patch", "test_manifest", "next_file", "call_graph"])
    ap.add_argument("--max-tokens-per-source", type=int, default=2048)
    ap.add_argument("--repos-dir", type=Path, default=None, help="git clones named owner__repo (for call graphs)")
    ap.add_argument("--train-tests", type=Path, default=None, help="JSON {instance_id: {FAIL_TO_PASS, PASS_TO_PASS}}")
    ap.add_argument("--train-image-map", type=Path, default=None, help="JSON {instance_id: image} for train envs")
    ap.add_argument("--manifest", default="data/manifests/swe_train_filtered.txt")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    import pandas as pd
    from transformers import AutoTokenizer

    from envs.swe.data import rows

    s_tok, t_tok = AutoTokenizer.from_pretrained(args.student), AutoTokenizer.from_pretrained(args.teacher)

    def render(tok):
        return lambda msgs: list(tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=True))

    def enc(x: str) -> list[int]:
        return t_tok.encode(x, add_special_tokens=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = set(Path(args.manifest).read_text().split())
    tests = json.loads(args.train_tests.read_text()) if args.train_tests else {}
    images = json.loads(args.train_image_map.read_text()) if args.train_image_map else {}
    for split in ("verified", "train"):
        data = rows(split)
        if split == "train":
            data = [r for r in data if r["instance_id"] in keep and r["instance_id"] in images]
            for r in data:
                r.update(tests.get(r["instance_id"], {}))
        data = data[:args.limit] if args.limit else data
        built = [build_row(r, render(s_tok), render(t_tok), enc, t_tok.decode, args.sources, args.max_tokens_per_source,
                           call_graph_for(r, args.repos_dir) if "call_graph" in args.sources else "",
                           images.get(r["instance_id"]) or (instance_image(r["instance_id"]) if split == "verified"
                                                            else None), split)
                 for r in data]
        name = "verified_500" if split == "verified" else "swe_train_filtered"
        pd.DataFrame(built).to_parquet(out / f"{name}.parquet")
        print(split, len(built), "->", out / f"{name}.parquet")


if __name__ == "__main__":
    main()

"""PI-removed teacher audit (tab:pi_removed_audit): the teacher alone solves SWE-bench Verified.

  python -m eval.teacher_audit --teacher Qwen/Qwen3-8B-Base --adapters outputs/adapters/swe \
      --pi {none|sentinel|swe_deployable|swe_oracle} --seed 0 --out results/swe/<run>/seed0

--pi none: no adapters. sentinel: every adapter active with its [PI] block replaced by [NO_PI].
deployable / oracle: the real blocks of those sources. Multi-adapter logit composition runs in HF
(vLLM cannot compose several LoRA adapters that each have their own prefix), so this is slow. Use
--instances to shard the 500 instances over GPUs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--adapters", default="outputs/adapters/swe")
    ap.add_argument("--pi", default="none")
    ap.add_argument("--instances", default="data/manifests/verified_500.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from chop.pi.serialize import NO_PI_SENTINEL, format_pi_block
    from chop.pi.sources import PROTOCOLS
    from envs.swe.codeact import ChatRenderer
    from envs.swe.data import rows
    from envs.swe.grading import grade, instance_image
    from envs.swe.sandbox import DockerSandbox
    from eval.backends import HFCompositeBackend
    from eval.swe_eval import run_all
    from pi_extraction import sources as S

    tok = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16).cuda().eval()
    names = [] if args.pi == "none" else (PROTOCOLS["swe_oracle"] if args.pi == "sentinel" else PROTOCOLS[args.pi])
    names = [n for n in names if n != "next_file"]      # per-step source: needs the live edit log
    if names:
        model = PeftModel.from_pretrained(model, f"{args.adapters}/{names[0]}", adapter_name=names[0])
        for n in names[1:]:
            model.load_adapter(f"{args.adapters}/{n}", adapter_name=n)
    backend = HFCompositeBackend(model, tok.pad_token_id or tok.eos_token_id, tok.eos_token_id, seed=args.seed)
    keep = set(Path(args.instances).read_text().split())
    data = [r for r in rows("verified") if r["instance_id"] in keep]

    def blocks(r) -> dict[str, list[int]]:
        content = {"gold_patch": S.gold_patch(r["patch"]),
                   "test_manifest": S.test_manifest(r["FAIL_TO_PASS"], r["PASS_TO_PASS"]), "call_graph": ""}
        return {n: tok.encode(format_pi_block(n, NO_PI_SENTINEL if args.pi == "sentinel" else content[n]),
                              add_special_tokens=False) for n in names}

    eps = asyncio.run(run_all(data, lambda r: backend.for_instance(blocks(r), names), ChatRenderer(tok),
                              lambda r: DockerSandbox(instance_image(r["instance_id"])), args.max_turns, 600.0,
                              32768, 4096, 60000, concurrency=1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_id = {r["instance_id"]: r for r in data}
    with (out / "per_instance.jsonl").open("w") as f:
        for e in eps:
            rep = grade(by_id[e["instance_id"]], e["model_patch"], f"audit-{args.pi}-{args.seed}")
            f.write(json.dumps({"instance_id": e["instance_id"], "resolved": rep["resolved"],
                                "trajectory_tokens": e["trajectory_tokens"]}) + "\n")


if __name__ == "__main__":
    main()

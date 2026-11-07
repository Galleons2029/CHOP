#!/usr/bin/env python3
"""Check that a student/teacher pair can share token ids (OPD compares distributions over one vocabulary).

  python scripts/check_tokenizers.py Qwen/Qwen3-8B-Base deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

It reports tokens whose ids differ, tokens present on one side only, special-token differences, and
whether the chat templates differ. A differing chat template is expected (each side renders the prompt
with its own template); differing ids for ordinary tokens make the pair unusable without vocabulary
alignment.
"""

from __future__ import annotations

import argparse
import json


def compare_vocabs(a: dict[str, int], b: dict[str, int], special_a: set[str] = frozenset(),
                   special_b: set[str] = frozenset()) -> dict:
    shared = set(a) & set(b)
    mismatched = sorted(t for t in shared if a[t] != b[t])
    ordinary_mismatch = [t for t in mismatched if t not in special_a and t not in special_b]
    return {"shared": len(shared), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
            "id_mismatches": len(mismatched), "ordinary_id_mismatches": len(ordinary_mismatch),
            "examples": ordinary_mismatch[:10], "special_only_a": sorted(special_a - special_b)[:20],
            "special_only_b": sorted(special_b - special_a)[:20], "compatible": not ordinary_mismatch}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("student")
    ap.add_argument("teacher")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    s, t = AutoTokenizer.from_pretrained(args.student), AutoTokenizer.from_pretrained(args.teacher)
    rep = compare_vocabs(s.get_vocab(), t.get_vocab(), set(s.all_special_tokens), set(t.all_special_tokens))
    rep["chat_template_differs"] = (s.chat_template or "") != (t.chat_template or "")
    rep["eos"] = {"student": s.eos_token, "teacher": t.eos_token}
    print(json.dumps(rep, indent=2))
    raise SystemExit(0 if rep["compatible"] else 1)


if __name__ == "__main__":
    main()

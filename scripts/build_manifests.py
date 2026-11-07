#!/usr/bin/env python3
"""Build the CC-BY-4.0 data manifests from the public SWE-bench releases.

Outputs (data/manifests/):
  verified_500.txt            Verified instance ids (sorted)
  swe_agent_300_seed0.txt     fixed random 300-instance subset for the SWE-agent harness (random.Random(0))
  swe_train_filtered.txt      training ids after the default leakage filter (commit + non-generic path)
  swe_train_strict.txt        training ids after the strict filter (+ same repository)
  leakage_report.json         per-rule hit counts actually obtained
  categories.csv              heuristic issue categories for Verified (envs/swe/categories.py)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from envs.swe import categories, leakage
from envs.swe.data import rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/manifests")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    verified, train = rows("verified"), rows("train")
    ids = sorted(r["instance_id"] for r in verified)
    (out / "verified_500.txt").write_text("\n".join(ids) + "\n")
    (out / "swe_agent_300_seed0.txt").write_text("\n".join(sorted(random.Random(0).sample(ids, 300))) + "\n")
    kept, report = leakage.apply(train, verified, strict=False)
    kept_strict, report_strict = leakage.apply(train, verified, strict=True)
    (out / "swe_train_filtered.txt").write_text("\n".join(sorted(kept)) + "\n")
    (out / "swe_train_strict.txt").write_text("\n".join(sorted(kept_strict)) + "\n")
    report = {"default": report, "strict": report_strict,
              "paper_reported": {"default_removed": 27, "strict_additional_removed": 1823,
                                 "train_instances": 19008, "kept_default": 18981},
              "note": "Counts computed on princeton-nlp/SWE-bench (train) vs princeton-nlp/SWE-bench_Verified."}
    (out / "leakage_report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "categories.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "category"])
        cats = {r["instance_id"]: categories.categorize(r) for r in verified}
        for iid in ids:
            w.writerow([iid, cats[iid]])
    print(json.dumps(report["default"], indent=1), json.dumps(report["strict"]["rule_hits"]))
    print("categories:", dict(Counter(cats.values())))


if __name__ == "__main__":
    main()

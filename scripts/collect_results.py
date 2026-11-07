#!/usr/bin/env python3
"""Aggregate raw run outputs into results/summary.json (only measured values; nothing is filled in).

Layout read:
  results/math/<run>/seed<k>/summary.json        (eval.math_eval)
  results/math/<run>/seed<k>/samples.jsonl
  results/swe/<run>/seed<k>/per_instance.jsonl    (eval.swe_eval, or converted OpenHands/SWE-agent reports)
Summary written:
  math[run][benchmark] = {mean, sd, seeds, avg_tokens}
  swe[run]            = {resolve, ci: [lo, hi], seeds, instances, avg_len_k}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.stats import resolve_rate_ci


def _seed_dirs(run: Path) -> list[Path]:
    return sorted(p for p in run.iterdir() if p.is_dir() and p.name.startswith("seed"))


def collect_math(root: Path) -> dict:
    out = {}
    for run in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        per_bench = defaultdict(list)
        toks = defaultdict(list)
        for sd in _seed_dirs(run):
            summ = json.loads((sd / "summary.json").read_text())
            for bench, v in summ.items():
                if bench == "args":
                    continue
                key = next(k for k in v if k.startswith("avg@"))
                per_bench[bench].append(v[key])
            samples = sd / "samples.jsonl"
            if samples.exists():
                for line in samples.read_text().splitlines():
                    r = json.loads(line)
                    toks[r["benchmark"]].append(r["tokens"])
        out[run.name] = {b: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                             "seeds": len(v), "avg_tokens": float(np.mean(toks[b])) if toks[b] else None}
                         for b, v in per_bench.items()}
    return out


def swe_matrix(run: Path) -> tuple[np.ndarray, list[str], list[float]]:
    seeds, lens = [], []
    ids: list[str] | None = None
    for sd in _seed_dirs(run):
        rows = [json.loads(line) for line in (sd / "per_instance.jsonl").read_text().splitlines()]
        rows.sort(key=lambda r: r["instance_id"])
        if ids is None:
            ids = [r["instance_id"] for r in rows]
        elif ids != [r["instance_id"] for r in rows]:
            raise ValueError(f"{run}: seeds evaluated different instance lists")
        seeds.append([int(bool(r["resolved"])) for r in rows])
        lens += [r.get("trajectory_tokens", 0) for r in rows]
    return np.array(seeds), ids or [], lens


def collect_swe(root: Path) -> dict:
    out = {}
    for run in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        m, ids, lens = swe_matrix(run)
        if m.size == 0:
            continue
        est = resolve_rate_ci(m)
        out[run.name] = {"resolve": est.mean, "ci": [est.lo, est.hi], "seeds": int(m.shape[0]),
                         "instances": len(ids), "avg_len_k": float(np.mean(lens)) / 1000 if lens else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = Path(args.results)
    summary = {"math": collect_math(root / "math"), "swe": collect_swe(root / "swe")}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"math runs: {len(summary['math'])}, swe runs: {len(summary['swe'])}")


if __name__ == "__main__":
    main()

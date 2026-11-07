#!/usr/bin/env python3
"""Reported-vs-measured comparison for every paper table (markdown, one section per table).

Rows come from configs/experiments/<table>/*.yaml. Each config states its table label, row name,
the values the paper reports (``reported``), and the run name its outputs are stored under (``name``).
Measured values come from results/summary.json (scripts/collect_results.py). Missing runs print as "—".
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from chop.config import load_config


def measured(summary: dict, cfg) -> dict:
    """Measured values for a row; *_gain subtracts the reference run, gap_recovery uses cfg.gap."""
    run = cfg.name
    if run in summary.get("swe", {}):
        r = summary["swe"][run]
        return {"resolve": r["resolve"], "ci": r["ci"], "seeds": r["seeds"], "avg_len_k": r.get("avg_len_k")}
    if run not in summary.get("math", {}):
        return {}
    got = {b: v["mean"] for b, v in summary["math"][run].items()}
    got["seeds"] = min(v["seeds"] for v in summary["math"][run].values())
    ref = summary["math"].get(cfg.reference or "", {})
    for b in list(got):
        if b in ref:
            got[f"{b}_gain"] = got[b] - ref[b]["mean"]
    if cfg.gap and "aime24" in got:
        got["gap_recovery"] = 100.0 * (got["aime24"] - cfg.gap[0]) / (cfg.gap[1] - cfg.gap[0])
    return got


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:.1f}" for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def render(configs: list, summary: dict) -> str:
    by_table = defaultdict(list)
    for c in configs:
        by_table[c.table].append(c)
    lines = []
    for table in sorted(by_table):
        lines += [f"### {table}", "", "| row | metric | reported | measured | seeds |", "|---|---|---|---|---|"]
        for c in by_table[table]:
            got = measured(summary, c)
            for metric, rep in (c.reported or {"—": None}).items():
                val = got.get(metric)
                lines.append(f"| {c.row} | {metric} | {fmt(rep)} | {fmt(val)} | {got.get('seeds', 0)} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="configs/experiments")
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--out", default="results/tables.md")
    args = ap.parse_args()
    cfgs = [load_config(p) for p in sorted(Path(args.configs).rglob("*.yaml")) if not p.name.startswith("_")]
    cfgs = [c for c in cfgs if c.table]
    summary = json.loads(Path(args.summary).read_text()) if Path(args.summary).exists() else {}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render(cfgs, summary))
    print(f"{len(cfgs)} rows -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the paper's Figures 1 and 2 from measured results only.

Figure 1 reads results/summary.json (math sweep, SWE-bench main table, anisotropy gap recovery).
Figure 2 reads the reliability diagnostics written by analysis/reliability_validation.py and
analysis/reset_rollout.py.

A panel whose inputs are missing is left empty with a note. Nothing is simulated or copied from the
paper, so an empty panel means the corresponding runs have not been done yet. (The camera-ready Fig. 2
was drawn from parametric curves and random scatter that illustrate the reported correlations; a run of
this script replaces that with measured data.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"chop": "#1f3b73", "chop_light": "#5a7bc9", "topk": "#d97706", "opsd": "#0f766e", "opd": "#dc2626",
          "baseline": "#64748b", "teacher": "#111827", "grid": "#e5e7eb"}
CAPS = [1, 3, 7, 15, 30, 45, 60]
MATH_RUNS = {"Standard OPD": ("opd", COLORS["opd"]), "Top-K LSM": ("topk_lsm", COLORS["topk"]),
             "OPSD-style": ("opsd", COLORS["opsd"]), "CHOP": ("chop", COLORS["chop"])}
SWE_RUNS = [("Off-the-\nshelf", "offtheshelf"), ("Std.\nOPD", "opd"), ("Top-$K$", "topk_lsm"), ("OPSD", "opsd"),
            ("CHOP\ndep.", "chop_deployable"), ("CHOP\noracle", "chop_oracle")]
ANISO_RUNS = [("Std.\nOPD", "aniso_opd"), ("Top-$K$", "aniso_topk_lsm"), ("OPSD", "aniso_opsd"),
              ("Li\nfull", "aniso_li2026"), ("CHOP\ntoken", "aniso_chop_token_only"),
              ("CHOP\nno PI", "aniso_chop_no_pi"), ("CHOP\nSW", "aniso_chop_student_anchors"),
              ("CHOP\nfull", "aniso_chop_full")]


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": COLORS["grid"],
                         "grid.linewidth": 0.5, "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42})


def _missing(ax, what: str) -> None:
    ax.text(0.5, 0.5, f"no measured data yet\n({what})", ha="center", va="center", transform=ax.transAxes,
            color="#94a3b8", fontsize=8)


def fig1(summary: dict, out: Path, student: float = 38.4, teacher: float = 57.0) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.15))
    ax = axes[0]
    plotted = False
    for name, (prefix, color) in MATH_RUNS.items():
        xs, ys = [], []
        for cap in CAPS:
            r = summary.get("math", {}).get(f"{prefix}_{cap}k", {}).get("aime24")
            if r:
                xs.append(cap)
                ys.append(r["mean"])
        if xs:
            ax.plot(xs, ys, marker="o", ms=3.8, lw=1.35, color=color, label=name)
            plotted = True
    if plotted:
        ax.legend(loc="lower left", frameon=False, fontsize=6.7)
    else:
        _missing(ax, "results/math/<method>_<cap>k")
    ax.set_xlabel("Max response length (K tokens)")
    ax.set_ylabel("AIME 2024 avg@16")
    ax.set_title("(a) Long-horizon mathematical reasoning", pad=4)

    ax = axes[1]
    swe = summary.get("swe", {})
    vals = [swe.get(run, {}).get("resolve") for _, run in SWE_RUNS]
    if any(v is not None for v in vals):
        x = np.arange(len(SWE_RUNS))
        ax.bar(x, [v or 0 for v in vals], color=[COLORS["baseline"], COLORS["opd"], COLORS["topk"], COLORS["opsd"],
                                                  COLORS["chop_light"], COLORS["chop"]])
        ax.set_xticks(x, [lbl for lbl, _ in SWE_RUNS])
        if vals[0] is not None:
            ax.axhline(vals[0], color="#475569", lw=0.8, ls="--")
    else:
        _missing(ax, "results/swe/<run>")
    ax.set_ylabel("Resolve rate (%)")
    ax.set_title("(b) SWE-bench Verified (500 instances)", pad=4)

    ax = axes[2]
    gaps = []
    for _, run in ANISO_RUNS:
        r = summary.get("math", {}).get(run, {}).get("aime24")
        gaps.append(None if r is None else 100 * (r["mean"] - student) / (teacher - student))
    if any(g is not None for g in gaps):
        x = np.arange(len(ANISO_RUNS))
        ax.bar(x, [g or 0 for g in gaps], color=COLORS["chop_light"])
        ax.set_xticks(x, [lbl for lbl, _ in ANISO_RUNS])
    else:
        _missing(ax, "results/math/aniso_<method>")
    ax.set_ylabel("Teacher-student gap recovered (%)")
    ax.set_title("(c) Anisotropy probe", pad=4)
    fig.savefig(out / "fig1_intro_hero.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig2(analysis: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    depth = analysis / "reliability_depth.json"
    ax = axes[0]
    if depth.exists():
        d = json.loads(depth.read_text())
        for step, curve in sorted(d["median_rho"].items(), key=lambda kv: int(kv[0])):
            ax.plot(d["depth_k"], curve, label=f"step {step}")
        ax.legend(frameon=False, fontsize=7)
    else:
        _missing(ax, "analysis/reliability_validation.py --depth")
    ax.set_xlabel("Output position (K tokens)")
    ax.set_ylabel(r"Reliability $\rho_t$ (median)")
    ax.set_title("(a) Reliability vs depth", pad=4)
    rr = analysis / "reset_rollout.json"
    for ax, groups, title in ((axes[1], ("high_rho", "low_rho"), "(b) Gradient vs KL reduction"),
                              (axes[2], ("teacher_continuation", "low_rho"), "(c) Reset rollout")):
        if rr.exists():
            d = json.loads(rr.read_text())
            for g, color in zip(groups, (COLORS["chop"], COLORS["opd"]), strict=True):
                pts = d["conditions"][g]
                ax.scatter(pts["grad_norm"], pts["kl_reduction"], s=12, alpha=0.6, color=color,
                           label=f"{g} (r={d['pearson'][g]['r']:.2f})")
            ax.legend(frameon=False, fontsize=7)
        else:
            _missing(ax, "analysis/reset_rollout.py")
        ax.set_xlabel("OPD gradient norm")
        ax.set_ylabel("Subsequent KL reduction")
        ax.set_title(title, pad=4)
    fig.savefig(out / "fig2_reliability.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(args.summary).read_text()) if Path(args.summary).exists() else {}
    fig1(summary, out)
    fig2(Path(args.analysis), out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()

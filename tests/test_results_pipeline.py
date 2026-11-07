"""collect_results / make_tables / make_figures on a synthetic results tree (test data only)."""

import importlib.util
import json
from pathlib import Path

from chop.config import ExperimentConfig


def _load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(root: Path):
    for seed in (0, 1, 2):
        d = root / "swe" / "chop_deployable" / f"seed{seed}"
        d.mkdir(parents=True)
        with (d / "per_instance.jsonl").open("w") as f:
            for i in range(50):
                f.write(json.dumps({"instance_id": f"i{i:03d}", "resolved": (i + seed) % 7 == 0,
                                    "trajectory_tokens": 30000}) + "\n")
        m = root / "math" / "chop_60k" / f"seed{seed}"
        m.mkdir(parents=True)
        (m / "summary.json").write_text(json.dumps({"args": {}, "aime24": {"avg@16": 40.0 + seed}}))


def test_collect_and_tables(tmp_path):
    _tree(tmp_path)
    col = _load("scripts/collect_results.py")
    summary = {"math": col.collect_math(tmp_path / "math"), "swe": col.collect_swe(tmp_path / "swe")}
    r = summary["swe"]["chop_deployable"]
    assert r["seeds"] == 3 and r["instances"] == 50 and r["ci"][0] <= r["resolve"] <= r["ci"][1]
    assert summary["math"]["chop_60k"]["aime24"]["mean"] == 41.0
    tab = _load("paper/make_tables.py")
    cfg = ExperimentConfig(name="chop_deployable", table="tab:swebench_main", row="CHOP-deployable",
                           reported={"resolve": 13.2})
    missing = ExperimentConfig(name="not_run", table="tab:swebench_main", row="REOPOLD", reported={"resolve": 9.3})
    md = tab.render([cfg, missing], summary)
    assert "| CHOP-deployable | resolve | 13.2 |" in md and "| REOPOLD | resolve | 9.3 | — | 0 |" in md


def test_figures_render_without_simulated_data(tmp_path):
    figs = _load("paper/make_figures.py")
    figs.style()
    figs.fig1({}, tmp_path)
    figs.fig2(tmp_path / "no_analysis", tmp_path)
    assert (tmp_path / "fig1_intro_hero.pdf").stat().st_size > 0
    assert (tmp_path / "fig2_reliability.pdf").stat().st_size > 0


def test_reproduce_dry_run_and_budget(capsys):
    rep = _load("scripts/reproduce.py")
    import sys

    sys.argv = ["reproduce.py", "--table", "all", "--dry-run", "--summary"]
    rep.main()
    out = capsys.readouterr().out
    assert "tab:swebench_main" in out and "total" in out
    sys.argv = ["reproduce.py", "--table", "tab:anisotropy", "--dry-run"]
    rep.main()
    out = capsys.readouterr().out
    assert "chop_verl.main --config-name chop_math" in out and "--cap 30000" in out
    assert "trainer.total_training_steps=300" in out


def test_check_tokenizers_compare():
    chk = _load("scripts/check_tokenizers.py")
    rep = chk.compare_vocabs({"a": 0, "b": 1, "<s>": 2}, {"a": 0, "b": 1, "<s>": 5}, {"<s>"}, {"<s>"})
    assert rep["compatible"] and rep["id_mismatches"] == 1
    assert not chk.compare_vocabs({"a": 0}, {"a": 1})["compatible"]


def test_gain_and_gap_metrics():
    tab = _load("paper/make_tables.py")
    summary = {"math": {"x": {"aime24": {"mean": 44.3, "seeds": 3}}, "ref": {"aime24": {"mean": 40.0, "seeds": 3}}}}
    got = tab.measured(summary, ExperimentConfig(name="x", reference="ref", gap=[38.4, 57.0]))
    assert abs(got["aime24_gain"] - 4.3) < 1e-9 and round(got["gap_recovery"], 2) == 31.72

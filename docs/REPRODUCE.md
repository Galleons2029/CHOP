# Reproducing the paper

The commands for every table come from `scripts/reproduce.py`, which reads `configs/experiments/<table>/*.yaml`.
Each config records the paper's reported values, so `paper/make_tables.py` can show reported against measured
values side by side. Run the steps below in order. Steps 1–4 are shared by all tables.

## 1. Data and manifests

```bash
python scripts/build_manifests.py                 # data/manifests (already committed; re-running is deterministic)
python scripts/prepare_math.py                    # data/processed/math/*.parquet
python scripts/build_train_envs.py --workers 16   # SWE-bench training-split environments (Docker; best effort)
python scripts/prepare_swe.py --repos-dir /data/repos \
    --train-tests outputs/train_envs/tests.json --train-image-map outputs/train_envs/images.json
```

About the SWE-bench training environments:

- SWE-bench provides environments only for its evaluation repositories. `build_train_envs.py` reconstructs
  environments for the training split and records which instances got a working one
  (`outputs/train_envs/status.jsonl`).
- Only those instances enter training. **Report this coverage** with any SWE-bench result.
- The prepared rows also need call-graph PI. For that, `--repos-dir` must hold git clones named `owner__repo`.

## 2. Tokenizers

```bash
python scripts/check_tokenizers.py Qwen/Qwen3-8B-Base deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
python scripts/check_tokenizers.py Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B-Base
```

Both pairs share token ids over their common vocabulary (see [DESIGN.md](DESIGN.md) for the special-token
caveat).

## 3. PI adapters

For SWE-bench there is one rank-16 LoRA adapter per PI source. For math there is a single `final_answer` adapter.

1. Generate teacher trajectories on the training split with the SWE harness.
2. Turn them into triples (`adapters/triples.py`).
3. Train each adapter:

```bash
python -m adapters.train_lora --base Qwen/Qwen3-8B-Base --triples data/triples/swe.jsonl \
    --source call_graph --out outputs/adapters/swe/call_graph        # repeat per source
```

Calibration diagnostics (App. C.5) are computed with `adapters/calibration.py`.

## 4. Reliability calibration

```bash
python scripts/run_analysis.py calibrate --config configs/base/math.yaml --states 1024 --rollouts 16
```

This writes the fitted (λ₁, λ₂). Put them into `method.reliability` of the base config. It also writes
`results/analysis/calibration_states.pt`. Pass that file as `actor_rollout_ref.chop.calibration_states` so the
streaming quantiles start from the calibration corpus rather than a warm-up.

## 5. Training and evaluation

```bash
python scripts/reproduce.py --table tab:math_full --emit outputs/sh    # one script per (row, seed)
bash outputs/sh/chop_60k_s0.sh
```

Each emitted script:

1. writes the resolved experiment config;
2. trains with `chop_verl` (8×H100 by default);
3. merges the FSDP checkpoint to HF format;
4. evaluates:
   - math: avg@16 with the same rolling truncation used in training;
   - SWE-bench: pass@1. The default is the OpenHands harness: set `OPENHANDS_INFER_CMD`, see the header of
     `eval/swe_openhands.sh`. The repository's CodeAct harness (`eval/swe_eval.py`) is the alternative.

Rows marked with `notes` need an input that only an earlier run can provide, for example:

- the compute-parity rows need the teacher-token count;
- the frozen route needs routes from an OPD checkpoint;
- the oracle route needs ρ_ref for every state.

## 6. Tables and figures

```bash
python scripts/collect_results.py        # results/summary.json
python paper/make_tables.py              # results/tables.md: reported vs measured, row by row
python paper/make_figures.py             # results/figures/fig1_intro_hero.pdf, fig2_reliability.pdf
```

Figure 2 uses measured diagnostics. Generate the inputs with:

- `scripts/run_analysis.py reset` (reset rollout);
- `analysis/reliability_validation.py` (depth curves).

`make_figures.py` never simulates missing data. A missing input leaves its panel empty.

## Compute

`python scripts/reproduce.py --table all --dry-run --summary` prints H100-days per table. The training runs of all
tables come to about 3,190 H100-days. The paper's estimate is about 3,560, which also counts PI adapters, adapter
calibration references and pre-experimental sweeps.

Per-run costs follow tab:hparams:

- math: about 3.4 h per (method, cap) on 8×H100;
- SWE-bench: about 36 h per seed;
- each PI adapter: about 11 h.

## Rows that cannot be reproduced as stated

These rows are listed in [DESIGN.md](DESIGN.md) with the reasons.

- **tab:swebench_pi_main "stricter training filter".** It removes no instance on the public splits.
- **12-pair table rows with 70B / 8x22B teachers or an unpublished teacher (`Qwen3-14B-Distill`).**
- **tab:reliability_purity "learned reliability predictor" and tab:phi_refresh (legacy MLP routing).** The
  predictor code is in `analysis/legacy_mlp.py`, but routing on its output is not wired into the trainer.
- **The per-category breakdown.** The paper does not state how instances were assigned to categories.

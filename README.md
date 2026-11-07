# CHOP: Counterfactual Hierarchical On-Policy Distillation

This repository contains the code for **"When Token Supervision Fails: Segment-Level On-Policy Distillation for
Long-Horizon Agents"** (AACL 2026).

On-policy distillation (OPD) trains a student on its own trajectories using the teacher's next-token distribution.
On long tasks this stops working even when the teacher can still tell good trajectories from bad ones. CHOP makes
two decisions at every state the student visits:

1. **Granularity gate.** The reliability score ρ_t = exp(−λ₁H(q_t) − λ₂TV(q_t, p_t)) decides where to match.
   Where it is high, the student matches the teacher's next-token distribution (reverse KL). Where it falls in the
   lower tail (30th percentile over the preceding 8,192 states), the student instead matches a distribution of
   64-token segments, using a sliced-Wasserstein loss with REINFORCE gradients.
2. **PI gate.** The teacher can see privileged information (PI), such as a test manifest, a call graph or a
   reference answer. Each PI source has its own LoRA adapter, and the adapters are combined additively on the
   logits. The PI-conditioned teacher is used only where it changes the teacher's distribution: where
   κ̂_t = TV(π_T, π_T^(P)) is in the upper tail (70th percentile).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r environment/requirements-cpu.txt && pip install -e .     # CPU: tests, reference trainer
pip install -r environment/requirements-gpu.txt                           # GPU: verl + vLLM training/eval
```

## Quick start

```bash
make test                                   # CPU test suite (tiny random Qwen3 models, no downloads)
make smoke                                  # 3 CHOP steps of the reference trainer on a toy task (CPU)
bash scripts/smoke_gpu.sh 1                 # 10 steps of the verl recipe with Qwen3-0.6B / 1.7B (1 GPU)
python scripts/reproduce.py --table tab:swebench_main --dry-run   # commands for one paper table
```

## Repository map

```
chop/                 method core (framework-agnostic PyTorch)
  divergences, reliability, quantile, router, segment, losses, monitors, truncation, rollout
  pi/                 PI sources, [PI] serialisation, multi-LoRA composition, leakage/content controls
  engine.py           scalable pieces: sparse teacher targets, root-forward access, multi-anchor tree decoding
  reference_trainer.py  single-process trainer (correctness reference; CPU smoke runs)
chop_verl/            verl recipe: worker, driver step, trainer, AgentLoops (rolling math, OpenHands-style SWE)
envs/                 tasks: toy, math (DAPO-Math-17K, AIME 2024/2025, AMC 2023), SWE-bench (sandbox, editor,
                      CodeAct episodes, grading, leakage filter, training-split environments)
pi_extraction/        gold patch, test manifest, oracle next file, tree-sitter call graph
adapters/             PI adapter data (triples), LoRA training, calibration diagnostics
analysis/             alpha, AUROC, reliability validation, reset rollout, PI redundancy, SNR, A3 audit, agreement
eval/                 math avg@16, SWE-bench harnesses (CodeAct / OpenHands / SWE-agent), statistics
configs/              base recipes, method presets, experiments/<table>/<row>.yaml (generated)
data/manifests/       CC-BY-4.0 manifests built from the public SWE-bench releases
scripts/              data preparation, calibration, analyses, reproduce.py, result collection
paper/                reported-vs-measured tables and figures from measured results
```

## Reproducing the paper

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the order of steps, from data preparation and calibration through
PI adapters, training and evaluation to the tables and figures. It also gives the compute each table needs:
`scripts/reproduce.py --summary` estimates about 3,190 H100-days for the training runs of all tables. The PI
protocols (deployable vs oracle) are described in [docs/PI_PROTOCOL.md](docs/PI_PROTOCOL.md).

## License

- Code: Apache-2.0 ([LICENSE](LICENSE)).
- Data manifests: CC-BY-4.0 ([LICENSE-DATA](LICENSE-DATA)).
- The underlying datasets (SWE-bench, DAPO-Math-17K, AIME, AMC) keep their own licenses.

## Citation

See [CITATION.cff](CITATION.cff).

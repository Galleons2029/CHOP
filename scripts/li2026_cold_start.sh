#!/usr/bin/env bash
# Li et al. (2026) full recipe, stage 1: off-policy cold start (SFT on teacher rollouts of N prompts).
#   bash scripts/li2026_cold_start.sh <student> <teacher> <num_prompts> <out_dir>
# Stage 2 (OPD on teacher-aligned prompts) is the chop_verl run that reproduce.py emits after this script.
# The SFT stage uses verl's SFT trainer. Its config keys differ between verl releases, so check
# `python -m verl.trainer.sft_trainer --help` against the pinned version.
set -euo pipefail
STUDENT=$1; TEACHER=$2; N=$3; OUT=$4
mkdir -p "$OUT"
python scripts/teacher_rollouts.py --teacher "$TEACHER" --num-prompts "$N" --out "$OUT/teacher_rollouts.parquet"
torchrun --nproc_per_node=8 -m verl.trainer.sft_trainer \
  data.train_files="$OUT/teacher_rollouts.parquet" data.val_files="$OUT/teacher_rollouts.parquet" \
  data.prompt_key=prompt data.response_key=response model.partial_pretrain="$STUDENT" \
  trainer.default_local_dir="$OUT" trainer.total_epochs=1 trainer.project_name=chop \
  trainer.experiment_name="$(basename "$OUT")"
python -m verl.model_merger merge --backend fsdp --local_dir "$(ls -d "$OUT"/global_step_* | tail -1)" \
  --target_dir "$OUT/hf"

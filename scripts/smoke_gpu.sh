#!/usr/bin/env bash
# GPU smoke test for the verl recipe (1 node, 1-8 GPUs). Validates what cannot run on CPU:
# FSDP2 + vLLM wiring, FlexAttention tree masks, lockstep padding across ranks, weight sync.
#   bash scripts/smoke_gpu.sh [num_gpus]
set -euo pipefail
NGPU=${1:-1}
python scripts/prepare_math.py --out data/processed/math --limit-train 64 --limit-eval 8 \
  --student Qwen/Qwen3-0.6B-Base --teacher Qwen/Qwen3-1.7B
python -m chop_verl.main --config-name chop_math \
  data.train_files=data/processed/math/dapo_math_17k.parquet \
  data.val_files=data/processed/math/aime24.parquet \
  data.train_batch_size=8 data.max_response_length=1024 \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B-Base \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.chop.teacher_path=Qwen/Qwen3-1.7B \
  actor_rollout_ref.chop.rollout_chunk=256 actor_rollout_ref.chop.native_context=768 \
  actor_rollout_ref.rollout.max_model_len=1024 actor_rollout_ref.rollout.val_kwargs.n=2 \
  trainer.n_gpus_per_node="$NGPU" trainer.total_training_steps=10 trainer.test_freq=10 trainer.save_freq=-1 \
  trainer.experiment_name=smoke_gpu

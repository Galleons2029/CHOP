#!/usr/bin/env bash
# Second harness (App. second_harness): SWE-agent v1.1.0 default config, fixed 300-instance subset
# (data/manifests/swe_agent_300_seed0.txt), pass@1 graded by the SWE-bench harness.
# CLI flags are those of SWE-agent 1.1.0; check `sweagent run-batch --help` for your installed version.
#   bash eval/swe_agent.sh <checkpoint> <run_name>
set -euo pipefail
CKPT=$1; RUN=$2; PORT=${PORT:-8012}; OUT=results/swe_agent/$RUN
mkdir -p "$OUT"
vllm serve "$CKPT" --port "$PORT" --max-model-len 32768 --seed 0 > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!; trap 'kill $VLLM_PID' EXIT
until curl -sf "localhost:$PORT/v1/models" >/dev/null; do sleep 5; done
FILTER=$(paste -sd'|' data/manifests/swe_agent_300_seed0.txt)
sweagent run-batch --instances.type swe_bench --instances.subset verified --instances.split test \
  --instances.filter "^($FILTER)$" --agent.model.name "openai/$CKPT" \
  --agent.model.api_base "http://localhost:$PORT/v1" --agent.model.api_key EMPTY \
  --agent.model.per_instance_cost_limit 0 --output_dir "$OUT"
python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path "$OUT/preds.json" --run_id "$RUN" --max_workers 16

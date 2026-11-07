#!/usr/bin/env bash
# SWE-bench Verified pass@1 under the official OpenHands harness (tab:swebench_main, tab:turncap).
#   OPENHANDS_INFER_CMD='...' bash eval/swe_openhands.sh <checkpoint> <run_name> [max_iterations=50] [seed=0]
#
# The checkpoint is served with vLLM's OpenAI-compatible server. OpenHands' SWE-bench inference runs against
# it, and the SWE-bench harness grades the result. The inference entry point and its flags change between
# OpenHands releases, so this script does not hard-code them. Set OPENHANDS_INFER_CMD to the command of the
# pinned release (1.3.0 in the paper's reproducibility card). The command may use these placeholders:
#   {base_url} {model} {instances} {max_iterations} {output}
# The paper's reproducibility card fixes these settings, and your command must apply them:
#   50 iterations (turn-cap sweep: 10/25/50), 600 s per instance (timeout -> fail), no retries,
#   internet disabled during evaluation, test files hidden, temperature 0.7, top-p 0.95.
# The command must write SWE-bench predictions (instance_id, model_name_or_path, model_patch) to {output}.
set -euo pipefail
CKPT=$1; RUN=$2; ITERS=${3:-50}; SEED=${4:-0}
: "${OPENHANDS_INFER_CMD:?set OPENHANDS_INFER_CMD (see header)}"
PORT=${PORT:-8011}
OUT=results/swe_openhands/$RUN/iters$ITERS/seed$SEED
mkdir -p "$OUT"
vllm serve "$CKPT" --port "$PORT" --max-model-len 32768 --seed "$SEED" \
  --enable-auto-tool-choice --tool-call-parser hermes > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!; trap 'kill $VLLM_PID' EXIT
until curl -sf "localhost:$PORT/v1/models" >/dev/null; do sleep 5; done
CMD=${OPENHANDS_INFER_CMD//\{base_url\}/http://localhost:$PORT/v1}
CMD=${CMD//\{model\}/$CKPT}
CMD=${CMD//\{instances\}/$(realpath data/manifests/verified_500.txt)}
CMD=${CMD//\{max_iterations\}/$ITERS}
CMD=${CMD//\{output\}/$(realpath "$OUT")/predictions.jsonl}
echo "$CMD" > "$OUT/infer_cmd.txt"
bash -c "$CMD"
python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path "$OUT/predictions.jsonl" --run_id "$RUN-i$ITERS-s$SEED" --max_workers 16

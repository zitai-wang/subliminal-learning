#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=${SUBLEARN_GPU:-1}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=${SUBLEARN_PYTHON:-/home/wzy/anaconda3/envs/licv/bin/python}
CONFIG="$PROJECT_DIR/src/config/subliminal_mathvision.yaml"
RUN_DIR=${1:?Usage: run_subliminal_mathvision.sh RUN_DIR}
EVAL_BATCH_SIZE=${SUBLEARN_EVAL_BATCH_SIZE:-4}
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/provenance"
cp src/subliminal_mathvision.py "$RUN_DIR/provenance/subliminal_mathvision_execution.py"
cp scripts/run_subliminal_mathvision.sh "$RUN_DIR/provenance/run_subliminal_mathvision_execution.sh"
trap 'code=$?; printf "{\"exit_code\":%s,\"finished_at\":\"%s\"}\n" "$code" "$(date -Is)" > "$RUN_DIR/pipeline_exit.json"' EXIT

"$PYTHON" -u scripts/wait_for_gpu.py --devices "$CUDA_VISIBLE_DEVICES" --run-dir "$RUN_DIR" \
  --poll-seconds "${SUBLEARN_GPU_POLL_SECONDS:-5}" 2>&1 | tee -a "$RUN_DIR/logs/gpu_wait.log"

run_stage() {
    local label=$1
    shift
    echo "$(date -Is) Starting $label on physical CUDA $CUDA_VISIBLE_DEVICES"
    "$PYTHON" -u src/subliminal_mathvision.py evaluate --config "$CONFIG" --run-dir "$RUN_DIR" \
      --eval-batch-size "$EVAL_BATCH_SIZE" "$@" 2>&1 | tee -a "$RUN_DIR/logs/$label.log"
}

run_stage step1_baseline --conditions baseline_direct baseline_cot
run_stage step2_teacher --conditions teacher_cot
run_stage step6_student --conditions student_direct student_cot
"$PYTHON" -u scripts/summarize_subliminal_mathvision.py --run-dir "$RUN_DIR" \
  2>&1 | tee -a "$RUN_DIR/logs/step7_compare.log"
echo "$(date -Is) MathVision cross-dataset evaluation complete"

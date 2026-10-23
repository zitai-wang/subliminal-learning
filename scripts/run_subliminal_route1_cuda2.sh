#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=${SUBLEARN_GPU:-2}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=${SUBLEARN_PYTHON:-/home/wzy/anaconda3/envs/licv/bin/python}
CONFIG="$PROJECT_DIR/src/config/subliminal_route1.yaml"
RUN_DIR=${1:?Usage: run_subliminal_route1_cuda2.sh RUN_DIR}
EVAL_BATCH_SIZE=${SUBLEARN_EVAL_BATCH_SIZE:-4}
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/provenance"
cp src/subliminal_route1.py "$RUN_DIR/provenance/subliminal_route1_execution.py"
cp scripts/run_subliminal_route1_cuda2.sh "$RUN_DIR/provenance/run_subliminal_route1_execution.sh"
trap 'code=$?; printf "{\"exit_code\":%s,\"finished_at\":\"%s\"}\n" "$code" "$(date -Is)" > "$RUN_DIR/pipeline_exit.json"' EXIT

if [[ ${SUBLEARN_WAIT_FOR_GPU:-1} == 1 ]]; then
    "$PYTHON" -u scripts/wait_for_gpu.py --devices "$CUDA_VISIBLE_DEVICES" --run-dir "$RUN_DIR" --poll-seconds "${SUBLEARN_GPU_POLL_SECONDS:-15}" 2>&1 | tee -a "$RUN_DIR/logs/gpu_wait.log"
fi

run_stage() {
    local label=$1
    shift
    echo "$(date -Is) Starting $label on physical CUDA $CUDA_VISIBLE_DEVICES"
    "$PYTHON" -u src/subliminal_route1.py "$@" --config "$CONFIG" --run-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/logs/$label.log"
}

run_stage step1_baseline evaluate --conditions baseline_direct baseline_cot --eval-batch-size "$EVAL_BATCH_SIZE"
run_stage step2_teacher evaluate --conditions teacher_cot --eval-batch-size "$EVAL_BATCH_SIZE"
run_stage step3_4_generate_filter generate
run_stage step5_train train
run_stage step6_student evaluate --conditions student_direct student_cot --eval-batch-size "$EVAL_BATCH_SIZE"
"$PYTHON" -u scripts/summarize_subliminal_route1.py --run-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/logs/step7_compare.log"
echo "$(date -Is) All stages complete"

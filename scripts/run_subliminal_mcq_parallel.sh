#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
PYTHON=${SUBLEARN_PYTHON:-/home/wzy/anaconda3/envs/licv/bin/python}
CONFIG=${1:?Usage: run_subliminal_mcq_parallel.sh CONFIG RUN_DIR BASE_GPU STUDENT_GPU}
RUN_DIR=${2:?Usage: run_subliminal_mcq_parallel.sh CONFIG RUN_DIR BASE_GPU STUDENT_GPU}
BASE_GPU=${3:?Usage: run_subliminal_mcq_parallel.sh CONFIG RUN_DIR BASE_GPU STUDENT_GPU}
STUDENT_GPU=${4:?Usage: run_subliminal_mcq_parallel.sh CONFIG RUN_DIR BASE_GPU STUDENT_GPU}
EVAL_BATCH_SIZE=${SUBLEARN_EVAL_BATCH_SIZE:-4}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/provenance"
cp "$0" "$RUN_DIR/provenance/run_subliminal_mcq_parallel_execution.sh"
trap 'code=$?; printf "{\"exit_code\":%s,\"finished_at\":\"%s\"}\n" "$code" "$(date -Is)" > "$RUN_DIR/pipeline_exit.json"' EXIT

"$PYTHON" -u scripts/wait_for_gpu.py --devices "$BASE_GPU" --run-dir "$RUN_DIR" \
  --poll-seconds 5 --status-name base_gpu_wait.json 2>&1 | tee "$RUN_DIR/logs/base_gpu_wait.log"
env CUDA_VISIBLE_DEVICES="$BASE_GPU" "$PYTHON" -u src/subliminal_mcq.py evaluate \
  --config "$CONFIG" --run-dir "$RUN_DIR" --eval-batch-size "$EVAL_BATCH_SIZE" \
  --conditions baseline_direct baseline_cot teacher_cot 2>&1 | tee "$RUN_DIR/logs/base_teacher.log" &
base_pid=$!

(
  "$PYTHON" -u scripts/wait_for_gpu.py --devices "$STUDENT_GPU" --run-dir "$RUN_DIR" \
    --poll-seconds 5 --status-name student_gpu_wait.json 2>&1 | tee "$RUN_DIR/logs/student_gpu_wait.log"
  env CUDA_VISIBLE_DEVICES="$STUDENT_GPU" "$PYTHON" -u src/subliminal_mcq.py evaluate \
    --config "$CONFIG" --run-dir "$RUN_DIR" --eval-batch-size "$EVAL_BATCH_SIZE" \
    --conditions student_direct student_cot 2>&1 | tee "$RUN_DIR/logs/student.log"
) &
student_pid=$!

set +e
wait "$base_pid"; base_code=$?
wait "$student_pid"; student_code=$?
set -e
printf '{"exit_code":%s,"finished_at":"%s"}\n' "$base_code" "$(date -Is)" > "$RUN_DIR/base_worker_exit.json"
printf '{"exit_code":%s,"finished_at":"%s"}\n' "$student_code" "$(date -Is)" > "$RUN_DIR/student_worker_exit.json"
if [[ "$base_code" -ne 0 || "$student_code" -ne 0 ]]; then
  echo "MCQ evaluation failed: base=$base_code student=$student_code" >&2
  exit 1
fi
"$PYTHON" -u scripts/summarize_subliminal_mcq.py --run-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/logs/summarize.log"

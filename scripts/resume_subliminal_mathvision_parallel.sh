#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
PYTHON=${SUBLEARN_PYTHON:-/home/wzy/anaconda3/envs/licv/bin/python}
CONFIG="$PROJECT_DIR/src/config/subliminal_mathvision.yaml"
RUN_DIR=${1:?Usage: resume_subliminal_mathvision_parallel.sh RUN_DIR}
TEACHER_GPU=${SUBLEARN_TEACHER_GPU:-1}
STUDENT_GPU=${SUBLEARN_STUDENT_GPU:-2}
EVAL_BATCH_SIZE=${SUBLEARN_EVAL_BATCH_SIZE:-4}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/provenance"
cp "$0" "$RUN_DIR/provenance/resume_subliminal_mathvision_parallel_execution.sh"
trap 'code=$?; printf "{\"exit_code\":%s,\"finished_at\":\"%s\"}\n" "$code" "$(date -Is)" > "$RUN_DIR/pipeline_exit.json"' EXIT

if [[ "$TEACHER_GPU" == "$STUDENT_GPU" ]]; then
    echo "Teacher and Student require distinct physical GPUs." >&2
    exit 2
fi

"$PYTHON" -u scripts/wait_for_gpu.py --devices "$TEACHER_GPU,$STUDENT_GPU" \
  --run-dir "$RUN_DIR" --poll-seconds "${SUBLEARN_GPU_POLL_SECONDS:-2}" \
  --status-name parallel_gpu_wait.json 2>&1 | tee -a "$RUN_DIR/logs/parallel_gpu_wait.log"

echo "$(date -Is) Starting Teacher on physical CUDA $TEACHER_GPU"
env CUDA_VISIBLE_DEVICES="$TEACHER_GPU" "$PYTHON" -u src/subliminal_mathvision.py evaluate \
  --config "$CONFIG" --run-dir "$RUN_DIR" --eval-batch-size "$EVAL_BATCH_SIZE" \
  --conditions teacher_cot 2>&1 | tee -a "$RUN_DIR/logs/step2_teacher.log" &
teacher_pid=$!

echo "$(date -Is) Starting Student on physical CUDA $STUDENT_GPU"
env CUDA_VISIBLE_DEVICES="$STUDENT_GPU" "$PYTHON" -u src/subliminal_mathvision.py evaluate \
  --config "$CONFIG" --run-dir "$RUN_DIR" --eval-batch-size "$EVAL_BATCH_SIZE" \
  --conditions student_direct student_cot 2>&1 | tee -a "$RUN_DIR/logs/step6_student.log" &
student_pid=$!

set +e
wait "$teacher_pid"
teacher_code=$?
wait "$student_pid"
student_code=$?
set -e
printf '{"exit_code":%s,"finished_at":"%s"}\n' "$teacher_code" "$(date -Is)" \
  > "$RUN_DIR/teacher_worker_exit.json"
printf '{"exit_code":%s,"finished_at":"%s"}\n' "$student_code" "$(date -Is)" \
  > "$RUN_DIR/student_worker_exit.json"
if [[ "$teacher_code" -ne 0 || "$student_code" -ne 0 ]]; then
    echo "Parallel evaluation failed: teacher=$teacher_code student=$student_code" >&2
    exit 1
fi

"$PYTHON" -u scripts/summarize_subliminal_mathvision.py --run-dir "$RUN_DIR" \
  2>&1 | tee -a "$RUN_DIR/logs/step7_compare.log"
echo "$(date -Is) MathVision parallel resume complete"

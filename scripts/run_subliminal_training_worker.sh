#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
RUN_DIR=${1:?Usage: run_subliminal_training_worker.sh RUN_DIR}
PYTHON=${SUBLEARN_PYTHON:-/home/wzy/anaconda3/envs/licv/bin/python}
export CUDA_VISIBLE_DEVICES=${SUBLEARN_GPU:-1}
export SUBLEARN_AUXILIARY_TRAINING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN_DIR/logs"
if [[ -f "$RUN_DIR/logs/step5_train.log" ]]; then
    TRAIN_LOG_OFFSET=$(stat -c %s "$RUN_DIR/logs/step5_train.log")
else
    TRAIN_LOG_OFFSET=0
fi
trap 'code=$?; printf "{\"exit_code\":%s,\"finished_at\":\"%s\",\"log_start_bytes\":%s}\n" "$code" "$(date -Is)" "$TRAIN_LOG_OFFSET" > "$RUN_DIR/training_worker_exit.json"' EXIT
"$PYTHON" -u scripts/wait_for_gpu.py --devices "$CUDA_VISIBLE_DEVICES" \
    --run-dir "$RUN_DIR" --status-name gpu1_training_queue.json --poll-seconds 2 \
    2>&1 | tee -a "$RUN_DIR/logs/gpu1_training_queue.log"
"$PYTHON" -u src/subliminal_route1.py train --run-dir "$RUN_DIR" \
    --config "$PROJECT_DIR/src/config/subliminal_route1.yaml" \
    2>&1 | tee -a "$RUN_DIR/logs/step5_train.log"

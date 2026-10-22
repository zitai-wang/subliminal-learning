#!/usr/bin/env bash
set -euo pipefail
cd /data1/wzy/sub-learn
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/wzy/anaconda3/envs/licv/bin/python
CONFIG=/data1/wzy/sub-learn/src/config/subliminal_route1.yaml
RUN_DIR=${1:?Usage: run_subliminal_route1_cuda2.sh RUN_DIR}
mkdir -p "$RUN_DIR/logs"
for stage in generate train evaluate; do
    echo "$(date -Is) Starting $stage on physical CUDA 2"
    "$PYTHON" -u src/subliminal_route1.py "$stage" --config "$CONFIG" --run-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/logs/$stage.log"
done
echo "$(date -Is) All stages complete"

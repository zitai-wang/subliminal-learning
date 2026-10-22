#!/usr/bin/env bash
set -euo pipefail

cd /data1/wzy/sub-learn

LOG_DIR="results/rebuttal_support_size/eval_launches"
mkdir -p "${LOG_DIR}"

echo "[chain] waiting for QwenVL support-size evals to finish..."
while true; do
  qwen_json_count=$(find results/rebuttal_support_size/support*_from_500/evals/base_layerwise/qwen2.5-vl-7b-instruct -name 'use_base_vector_layers_*_direct_q_1.0.json' 2>/dev/null | wc -l)
  qwen_proc_count=$(ps -eo cmd | grep -E 'eval_base.py|run_eval_layers_base.py' | grep 'qwen2.5-vl-7b-instruct' | grep -v grep | wc -l)
  now=$(date '+%F %T')
  echo "[chain] ${now} qwen_json=${qwen_json_count}/504 qwen_proc=${qwen_proc_count}"

  if [[ "${qwen_json_count}" -ge 504 && "${qwen_proc_count}" -eq 0 ]]; then
    break
  fi
  sleep 300
done

echo "[chain] QwenVL complete. launching InternVL on cuda0,1,2..."
/home/wzy/anaconda3/envs/licv/bin/python scripts/run_rebuttal_support_size_base_layerwise_eval.py \
  --families intern \
  --support-sizes 100,200,300 \
  --devices 0,1,2 \
  2>&1 | tee "${LOG_DIR}/rebuttal_intern_s123_base_eval_launcher.log"

echo "[chain] InternVL launcher finished."

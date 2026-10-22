#!/usr/bin/env bash
set -euo pipefail

cd /data1/wzy/sub-learn

export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_home/datasets
export TRANSFORMERS_CACHE=/tmp/hf_home/transformers
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/data1/wzy/sub-learn:/data1/wzy/sub-learn/src

/home/wzy/anaconda3/envs/licv/bin/python /data1/wzy/sub-learn/scripts/run_static_pipeline.py \
  --model-name qwen2.5-7b-instruct \
  --dataset strategyqa \
  --method ffn \
  --devices 2 \
  --generate-batch-size 4 \
  --extract-batch-size 1 \
  --eval-batch-size 4 \
  --run-tag qwen2.5-7b-instruct_strategyqa_ffn_20260528_162020

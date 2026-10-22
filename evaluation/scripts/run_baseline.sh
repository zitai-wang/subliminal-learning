#!/usr/bin/env bash
set -euo pipefail


python evaluation/run_all.py \
  --config-name baseline_qwen25_vl_7b "$@"

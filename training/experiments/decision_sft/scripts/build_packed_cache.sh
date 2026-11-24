#!/bin/bash
set -euo pipefail

EXP_DIR=/path/to/experiments/decision_sft
FROZEN_STACK=${EXP_DIR}/deps/frozen_b8241e8_mg31e327_mb_c527134
SWIFT_HOME=${FROZEN_STACK}/ms-swift
MEGATRON_PATH=${FROZEN_STACK}/Megatron-LM
MCORE_BRIDGE_PATH=${FROZEN_STACK}/mcore-bridge/src
PYTHON=/path/to/storage/research/miniconda3/envs/swift/bin/python
MODEL_PATH=/path/to/checkpoints/workflow_sft
TRAIN_JSONL=${EXP_DIR}/data/dev_4000_800/train.jsonl
CACHE_ROOT=${EXP_DIR}/cache_packed_default_v1
OUTPUT_DIR=${CACHE_ROOT}/train

if [ -e "${OUTPUT_DIR}" ]; then
  echo "packed cache target already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

export AOSS_CONF=/path/to/storage/research/aoss.conf
export PYTHONNOUSERSITE=1
export PYTHONPATH=${MCORE_BRIDGE_PATH}:${MEGATRON_PATH}:${SWIFT_HOME}:${PYTHONPATH:-}
export IMAGE_MAX_TOKEN_NUM=2048
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=16
export MAX_PIXELS=200704

cd "${SWIFT_HOME}"
"${PYTHON}" -u swift/cli/export.py \
  --model "${MODEL_PATH}" \
  --dataset "${TRAIN_JSONL}" \
  --split_dataset_ratio 0 \
  --dataset_num_proc 8 \
  --dataset_shuffle false \
  --data_seed 20260824 \
  --to_cached_dataset true \
  --enable_thinking false \
  --add_non_thinking_prefix false \
  --response_prefix "" \
  --loss_scale default \
  --truncation_strategy delete \
  --max_length 8192 \
  --output_dir "${OUTPUT_DIR}"

test -d "${OUTPUT_DIR}/train"
echo "packed_cache_complete path=${OUTPUT_DIR}/train"

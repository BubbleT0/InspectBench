#!/bin/bash
set -euo pipefail

VARIANT=a2_balanced_contains_r1
EXP_DIR=/path/to/experiments/expanded_decision
SOURCE_EXP=/path/to/experiments/decision_sft
FROZEN_STACK=${SOURCE_EXP}/deps/frozen_b8241e8_mg31e327_mb_c527134
SWIFT_HOME=${FROZEN_STACK}/ms-swift
MEGATRON_PATH=${FROZEN_STACK}/Megatron-LM
MCORE_BRIDGE_PATH=${FROZEN_STACK}/mcore-bridge/src
PYTHON=/path/to/storage/research/miniconda3/envs/swift/bin/python
MODEL_PATH=/path/to/checkpoints/workflow_sft
DATA_DIR=${EXP_DIR}/data/${VARIANT}
CACHE_DIR=${EXP_DIR}/cache/${VARIANT}

if [ -e "${CACHE_DIR}" ]; then
  echo "cache target already exists: ${CACHE_DIR}" >&2
  exit 2
fi
mkdir -p "${CACHE_DIR}"

LOCAL_CACHE=$(mktemp -d "/tmp/critic_r1_full_${VARIANT}_cache.XXXXXX")
trap 'rm -rf "${LOCAL_CACHE}"' EXIT

export AOSS_CONF=/path/to/storage/research/aoss.conf
export PYTHONNOUSERSITE=1
export PYTHONPATH=${EXP_DIR}/scripts:${MCORE_BRIDGE_PATH}:${MEGATRON_PATH}:${SWIFT_HOME}:${PYTHONPATH:-}
export HF_HOME=${LOCAL_CACHE}/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export XDG_CACHE_HOME=${LOCAL_CACHE}/xdg
export TMPDIR=${LOCAL_CACHE}/tmp
export IMAGE_MAX_TOKEN_NUM=2048
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=16
export MAX_PIXELS=200704
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" "${TMPDIR}"

build_split() {
  local split=$1
  local input=${DATA_DIR}/${split}.jsonl
  local output=${CACHE_DIR}/${split}
  test -f "${input}"
  cd "${SWIFT_HOME}"
  "${PYTHON}" -u swift/cli/export.py \
    --model "${MODEL_PATH}" \
    --dataset "${input}" \
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
    --output_dir "${output}"
  test -d "${output}/train"
}

build_split train
build_split validation
echo "cache_complete variant=${VARIANT} path=${CACHE_DIR}"

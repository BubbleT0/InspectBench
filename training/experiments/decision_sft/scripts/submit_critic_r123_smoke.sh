#!/bin/bash
set -euo pipefail

EXP_DIR=/path/to/experiments/decision_sft
TRAIN_SCRIPT=${EXP_DIR}/scripts/train_critic_r123_8gpu.sh
WORKSPACE=YOUR_WORKSPACE_ID
PARTITION=neo1-edit
SPEC=N6lS.Iu.I80.8.64c1024g
CONTAINER=YOUR_TRAINING_IMAGE
MOUNT=YOUR_WORKSPACE_ID:/path/to/storage2,YOUR_WORKSPACE_ID:/path/to/storage1,YOUR_WORKSPACE_ID:/path/to/storage
SMOKE_DATA=${EXP_DIR}/audit/smoke_data

submit_one() {
  local variant=$1
  local stamp=$2
  local root=${EXP_DIR}/audit/gpu_smoke_${stamp}/${variant}
  local cache=${EXP_DIR}/audit/cpu_preflight_v1/${variant}/cache
  sco acp jobs create \
    --workspace-name "${WORKSPACE}" \
    --aec2-name "${PARTITION}" \
    --priority NORMAL \
    --container-image-url "${CONTAINER}" \
    --storage-mount "${MOUNT}" \
    --training-framework pytorch \
    --worker-spec "${SPEC}" \
    --worker-nodes 1 \
    -j "critic-smoke-${variant}-${stamp}" \
    --command="DATA_DIR_OVERRIDE=${SMOKE_DATA} CACHE_DIR_OVERRIDE=${cache} OUTPUT_DIR_OVERRIDE=${root}/output LOG_DIR_OVERRIDE=${root}/logs MAX_STEPS_OVERRIDE=2 SAVE_STEPS_OVERRIDE=100 EVAL_STEPS_OVERRIDE=1 DATASET_NUM_PROC_OVERRIDE=2 bash ${TRAIN_SCRIPT} ${variant}"
}

stamp=${1:?usage: $0 STAMP [r1|r2 ...]}
shift
if [ "$#" -eq 0 ]; then set -- r1 r2; fi
for variant in "$@"; do submit_one "$variant" "$stamp"; done

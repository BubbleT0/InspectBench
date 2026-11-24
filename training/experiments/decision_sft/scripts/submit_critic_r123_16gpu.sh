#!/bin/bash
set -euo pipefail

EXP_DIR=/path/to/experiments/decision_sft
TRAIN_SCRIPT=${EXP_DIR}/scripts/train_critic_r123_16gpu.sh
WORKSPACE=YOUR_WORKSPACE_ID
PARTITION=${PARTITION:-neo1-edit}
JOB_SUFFIX=${JOB_SUFFIX:-direct1}
SPEC=N6lS.Iu.I80.8.64c1024g
CONTAINER=YOUR_TRAINING_IMAGE
MOUNT=YOUR_WORKSPACE_ID:/path/to/storage2,YOUR_WORKSPACE_ID:/path/to/storage1,YOUR_WORKSPACE_ID:/path/to/storage

submit_one() {
  local variant=$1
  sco acp jobs create \
    --workspace-name "${WORKSPACE}" \
    --aec2-name "${PARTITION}" \
    --priority NORMAL \
    --container-image-url "${CONTAINER}" \
    --storage-mount "${MOUNT}" \
    --training-framework pytorch \
    --worker-spec "${SPEC}" \
    --worker-nodes 2 \
    -j "critic-r123-${variant}-16g-0824-${JOB_SUFFIX}" \
    --command="bash ${TRAIN_SCRIPT} ${variant}"
}

if [ "$#" -eq 0 ]; then
  set -- r2
fi

test -f "${TRAIN_SCRIPT}"
for variant in "$@"; do
  case "${variant}" in
    r1|r1_formal_packed|r1_np2p|r1_fixed|r1_packed|r2_packed|r2_formal_packed|r2|r3|r3_formal_packed) submit_one "${variant}" ;;
    *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
  esac
done

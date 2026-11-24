#!/bin/bash
set -euo pipefail

EXP_DIR=/path/to/experiments/expanded_decision
TRAIN_SCRIPT=${EXP_DIR}/scripts/train_r1_full_ratio_a2_16gpu.sh
WORKSPACE=YOUR_WORKSPACE_ID
PARTITION=${PARTITION:-neo1-edit}
SPEC=N6lS.Iu.I80.8.64c1024g
CONTAINER=YOUR_TRAINING_IMAGE
MOUNT=YOUR_WORKSPACE_ID:/path/to/storage2,YOUR_WORKSPACE_ID:/path/to/storage1,YOUR_WORKSPACE_ID:/path/to/storage

submit_one() {
  local variant=$1
  local short=$2
  sco acp jobs create \
    --workspace-name "${WORKSPACE}" \
    --aec2-name "${PARTITION}" \
    --priority NORMAL \
    --container-image-url "${CONTAINER}" \
    --storage-mount "${MOUNT}" \
    --training-framework pytorch \
    --worker-spec "${SPEC}" \
    --worker-nodes 2 \
    -j "critic-r1full-${short}-16g-0828" \
    --command="bash ${TRAIN_SCRIPT} ${variant}"
}

test -f "${TRAIN_SCRIPT}"
for variant in "$@"; do
  case "${variant}" in
    a_balanced) submit_one "${variant}" a1to1 ;;
    a2_balanced_contains_r1) submit_one "${variant}" a2r1 ;;
    b_natural_ratio_controlled) submit_one "${variant}" b215ctrl ;;
    c_all_unique) submit_one "${variant}" call ;;
    *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
  esac
done

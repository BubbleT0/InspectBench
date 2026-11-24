#!/bin/bash
set -euo pipefail

EXP_DIR=/path/to/experiments/decision_sft
TRAIN_SCRIPT=${EXP_DIR}/scripts/train_critic_r123_8gpu.sh
WORKSPACE=YOUR_WORKSPACE_ID
PARTITION=${PARTITION:-neo1-trajectory}
JOB_SUFFIX=${JOB_SUFFIX:-}
SPEC=N6lS.Iu.I80.8.64c1024g
CONTAINER=YOUR_TRAINING_IMAGE
MOUNT=YOUR_WORKSPACE_ID:/path/to/storage2,YOUR_WORKSPACE_ID:/path/to/storage1,YOUR_WORKSPACE_ID:/path/to/storage

submit_one() {
  local variant=$1
  local job_name=critic-sm-${variant}-8g-0824
  if [ -n "${JOB_SUFFIX}" ]; then
    job_name=${job_name}-${JOB_SUFFIX}
  fi
  sco acp jobs create \
    --workspace-name "${WORKSPACE}" \
    --aec2-name "${PARTITION}" \
    --priority NORMAL \
    --container-image-url "${CONTAINER}" \
    --storage-mount "${MOUNT}" \
    --training-framework pytorch \
    --worker-spec "${SPEC}" \
    --worker-nodes 1 \
    -j "${job_name}" \
    --command="bash ${TRAIN_SCRIPT} ${variant}"
}

if [ "$#" -eq 0 ]; then
  set -- r1 r2 r3
fi

test -f "${TRAIN_SCRIPT}"
for variant in "$@"; do
  case "${variant}" in
    r1|r2|r3) submit_one "${variant}" ;;
    *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
  esac
done

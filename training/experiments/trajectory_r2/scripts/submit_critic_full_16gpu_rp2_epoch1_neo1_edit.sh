#!/bin/bash
set -euo pipefail

EXP_DIR=/path/to/experiments/trajectory_r2
TRAIN_SCRIPT=${EXP_DIR}/scripts/train_critic_full_16gpu_rp2_epoch1_neo1_edit.sh
WORKSPACE=YOUR_WORKSPACE_ID
PARTITION=neo1-edit
SPEC=N6lS.Iu.I80.8.64c1024g
CONTAINER=YOUR_TRAINING_IMAGE
MOUNT=YOUR_WORKSPACE_ID:/path/to/storage2,YOUR_WORKSPACE_ID:/path/to/storage1,YOUR_WORKSPACE_ID:/path/to/storage
JOB_NAME=critic-full4src-rp2-16g-e1-edit-0820

test -f "${TRAIN_SCRIPT}"

sco acp jobs create \
    --workspace-name "${WORKSPACE}" \
    --aec2-name "${PARTITION}" \
    --priority NORMAL \
    --container-image-url "${CONTAINER}" \
    --storage-mount "${MOUNT}" \
    --training-framework pytorch \
    --worker-spec "${SPEC}" \
    --worker-nodes 2 \
    -j "${JOB_NAME}" \
    --command="bash ${TRAIN_SCRIPT}"

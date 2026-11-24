#!/bin/bash
set -euo pipefail

EXP=/path/to/experiments/trajectory_r1
SCRIPT=${EXP}/scripts/eval_three_checkpoints_8gpu.sh
WORKSPACE=YOUR_WORKSPACE_ID
PARTITION=neo1-trajectory
SPEC=N6lS.Iu.I80.8.64c1024g
CONTAINER=YOUR_TRAINING_IMAGE
MOUNT=YOUR_WORKSPACE_ID:/path/to/storage2,YOUR_WORKSPACE_ID:/path/to/storage1,YOUR_WORKSPACE_ID:/path/to/storage

test -f "${SCRIPT}"
sco acp jobs create \
  --workspace-name "${WORKSPACE}" \
  --aec2-name "${PARTITION}" \
  --priority NORMAL \
  --container-image-url "${CONTAINER}" \
  --storage-mount "${MOUNT}" \
  --training-framework pytorch \
  --worker-spec "${SPEC}" \
  --worker-nodes 1 \
  -j critic-rp1-val3ckpt-8g-0821 \
  --command="bash ${SCRIPT}"

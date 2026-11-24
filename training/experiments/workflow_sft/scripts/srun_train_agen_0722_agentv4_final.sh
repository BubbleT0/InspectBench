#!/bin/bash
# 8N pack+train — cleaned agentv4 0715/0716/0717/0721, date-aware RP.
set -euo pipefail

par_name=neo1-trajectory
num_nodes=${NUM_NODES:-8}

PARTITION=$par_name
WORKSPACE=YOUR_WORKSPACE_ID
CONTAINER=YOUR_TRAINING_IMAGE
MOUNT=YOUR_WORKSPACE_ID:/path/to/storage2,YOUR_WORKSPACE_ID:/path/to/storage1,YOUR_WORKSPACE_ID:/path/to/storage

spec_name="N6lS.Iu.I80.8.64c1024g"
TRAIN_SCRIPT=/path/to/storage/multimodal_ckpt/research/autotrain/ms-swift/shell/scripts/train_agen_0722_agentv4_final.sh

exp_name=neo_agentic_qwen36_27b_agen_0722_agentv4_final
LOG_DIR=/path/to/storage/multimodal_ckpt/research/autotrain/logs
LOG_SUB_FOLDER="$LOG_DIR/${exp_name}_$(date +%Y%m%d_%H%M%S)"
echo "日志: $LOG_SUB_FOLDER"

DATA_DIR_OVERRIDE=/path/to/storage/multimodal_data/research/agen/0715_0716_0717_0721_combined_rp_v1/data
CACHE_DIR_OVERRIDE=/path/to/storage/multimodal_ckpt/research/autotrain/cache/data/sft_agen_0722_agentv4_final
REPEAT_MAP=/path/to/storage/multimodal_data/research/agen/0715_0716_0717_0721_combined_rp_v1/repeat_map.tsv

RUN_CMD="export LOG_SUB_FOLDER=\"$LOG_SUB_FOLDER\"; \
         export EXP_SUFFIX=agen_0722_agentv4_final_rp_3ep; \
         export DATA_DIR_OVERRIDE=\"$DATA_DIR_OVERRIDE\"; \
         export CACHE_DIR_OVERRIDE=\"$CACHE_DIR_OVERRIDE\"; \
         export PACK_PARALLEL_SHARDS=8; \
         export PACK_NUM_PROC_PER_SHARD=8; \
         export REPEAT_MAP_TSV=\"$REPEAT_MAP\"; \
         bash $TRAIN_SCRIPT"

~/.sco/bin/sco acp jobs create \
--workspace-name "$WORKSPACE" \
--aec2-name "$PARTITION" \
--priority HIGHEST \
--container-image-url "$CONTAINER" \
--storage-mount "$MOUNT" \
--training-framework pytorch \
--worker-spec "$spec_name" \
--worker-nodes "$num_nodes" \
-j "$exp_name" \
--command="$RUN_CMD"

echo "Submitted agentv4 final: BASE Qwen3.6-27B, 128K/3ep, 8834 original / 12312 RP effective"
echo "  data=$DATA_DIR_OVERRIDE"
echo "  repeat_map=$REPEAT_MAP"
echo "  log=$LOG_SUB_FOLDER"

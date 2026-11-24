#!/bin/bash
set -euo pipefail

EXP=/path/to/experiments/expanded_decision
PYTHON=/path/to/storage/research/miniconda3/envs/swift/bin/python
CACHE=${EXP}/cache/a2_balanced_contains_r1

echo "[$(date '+%F %T %Z')] A2_CACHE_START"
bash "${EXP}/scripts/build_a2_cache.sh"
echo "[$(date '+%F %T %Z')] A2_CACHE_COMMAND_COMPLETE"

"${PYTHON}" -c "from datasets import load_from_disk; t=load_from_disk('${CACHE}/train/train'); v=load_from_disk('${CACHE}/validation/train'); assert len(t)==13026, len(t); assert len(v)==800, len(v); print(f'A2_CACHE_AUDIT train={len(t)} validation={len(v)}')"

echo "[$(date '+%F %T %Z')] A2_SUBMIT_START"
PARTITION=neo1-edit bash "${EXP}/scripts/submit_r1_full_ratio_a2_16gpu.sh" a2_balanced_contains_r1
echo "[$(date '+%F %T %Z')] A2_PIPELINE_COMPLETE"

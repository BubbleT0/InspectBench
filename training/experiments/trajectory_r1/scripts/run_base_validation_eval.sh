#!/bin/bash
set -euo pipefail

EXP=/path/to/experiments/trajectory_r1
EVAL=${EXP}/eval/stratified_validation_20260821
ENDPOINT=https://api.example.invalid

python3 "${EXP}/scripts/evaluate_stratified_validation.py" \
  --endpoint "${ENDPOINT}" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/negative_v41_flash_50000_validation.jsonl" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/negative_v41_validation.jsonl" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/positive_pinterest_validation.jsonl" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/positive_wikimedia_validation.jsonl" \
  --output-dir "${EVAL}/base" \
  --split validation \
  --workers 4 \
  --max-tokens 2048 \
  --retries 3

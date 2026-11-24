#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <checkpoint-name: checkpoint-250|checkpoint-500|checkpoint-740> <endpoint>" >&2
  exit 2
fi

CHECKPOINT=$1
ENDPOINT=${2%/}
case "${CHECKPOINT}" in
  checkpoint-250|checkpoint-500|checkpoint-740) ;;
  *) echo "unsupported checkpoint: ${CHECKPOINT}" >&2; exit 2 ;;
esac

EXP=/path/to/experiments/trajectory_r1
EVAL=${EXP}/eval/stratified_validation_20260821

MODEL_ROOT=$(python3 -c 'import json,sys,urllib.request; print(json.load(urllib.request.urlopen(sys.argv[1]+"/v1/models", timeout=20))["data"][0].get("root", ""))' "${ENDPOINT}")
case "${MODEL_ROOT}" in
  *"/${CHECKPOINT}") ;;
  *) echo "endpoint model root mismatch: expected */${CHECKPOINT}, got ${MODEL_ROOT}" >&2; exit 3 ;;
esac

python3 "${EXP}/scripts/evaluate_stratified_validation.py" \
  --endpoint "${ENDPOINT}" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/negative_v41_flash_50000_validation.jsonl" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/negative_v41_validation.jsonl" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/positive_pinterest_validation.jsonl" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/positive_wikimedia_validation.jsonl" \
  --output-dir "${EVAL}/${CHECKPOINT}" \
  --split validation \
  --workers 4 \
  --max-tokens 2048 \
  --retries 3

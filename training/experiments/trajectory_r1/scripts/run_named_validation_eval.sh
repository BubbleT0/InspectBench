#!/bin/bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <output-label> <endpoint> <expected-root-suffix>" >&2
  exit 2
fi

LABEL=$1
ENDPOINT=${2%/}
EXPECTED_SUFFIX=$3
EXP=/path/to/experiments/trajectory_r1
EVAL=${EXP}/eval/stratified_validation_20260821

MODEL_ROOT=$(python3 -c 'import json,sys,urllib.request; print(json.load(urllib.request.urlopen(sys.argv[1]+"/v1/models", timeout=20))["data"][0].get("root", ""))' "${ENDPOINT}")
case "${MODEL_ROOT}" in
  *"${EXPECTED_SUFFIX}") ;;
  *) echo "endpoint model root mismatch: expected *${EXPECTED_SUFFIX}, got ${MODEL_ROOT}" >&2; exit 3 ;;
esac

python3 "${EXP}/scripts/evaluate_stratified_validation.py" \
  --endpoint "${ENDPOINT}" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/negative_v41_flash_50000_validation.jsonl" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/negative_v41_validation.jsonl" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/positive_pinterest_validation.jsonl" \
  --dataset "${EXP}/data/stage5_swift/data_swift_format/positive_wikimedia_validation.jsonl" \
  --output-dir "${EVAL}/${LABEL}" \
  --split validation \
  --workers 4 \
  --max-tokens 2048 \
  --retries 3

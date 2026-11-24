#!/usr/bin/env bash
set -euo pipefail

: "${CRITIC_BASE_URL:?Set CRITIC_BASE_URL}"
: "${CRITIC_MODEL:?Set CRITIC_MODEL}"
: "${CRITIC_API_KEY:?Set CRITIC_API_KEY}"

input_root="${1:?Usage: run_v41.sh INPUT_ROOT OUTPUT_ROOT [INPUT_JSONL]}"
output_root="${2:?Usage: run_v41.sh INPUT_ROOT OUTPUT_ROOT [INPUT_JSONL]}"
input_args=()
if [[ $# -ge 3 && -n "$3" ]]; then
  input_args=(--input-jsonl "$3")
fi

u1-critic-v41-run \
  --input-root "$input_root" \
  --output-root "$output_root" \
  "${input_args[@]}" \
  --base-url "$CRITIC_BASE_URL" \
  --model "$CRITIC_MODEL" \
  --api-key-env CRITIC_API_KEY \
  --workflow adversarial_consensus \
  --coverage-mode harness_v41 \
  --all --confirm-run-all YES \
  --max-in-flight "${CRITIC_MAX_IN_FLIGHT:-48}" \
  --max-images-in-flight "${CRITIC_MAX_IMAGES_IN_FLIGHT:-8}" \
  --timeout "${CRITIC_TIMEOUT:-180}" \
  --max-tokens "${CRITIC_MAX_TOKENS:-16000}" \
  --retries 2 \
  --consolidation-retries 2 \
  --unit-retries 2

#!/bin/bash
set -uo pipefail

EXP=/path/to/experiments/trajectory_r2
TOOL=/path/to/storage/research/miniconda3/envs/data_pipeline/bin/agent-data-tools
META=${EXP}/data/stage2_input_v3/meta_middle_format.json
OUT=${EXP}/data/stage2_validate_v3

mkdir -p "${OUT}/logs"
"${TOOL}" validate-openai \
  -i "${META}" \
  -l "${OUT}/logs" \
  -w 8 \
  -e inf \
  >"${OUT}/validate.stdout.log" 2>&1
code=$?
printf '%s\n' "${code}" >"${OUT}/validate.exitcode"
exit "${code}"

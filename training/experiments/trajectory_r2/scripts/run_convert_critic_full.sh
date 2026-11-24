#!/bin/bash
set -uo pipefail

EXP=/path/to/experiments/trajectory_r2
DEFAULT_TOOL=/path/to/storage/research/miniconda3/envs/data_pipeline/bin/agent-data-tools
ISOLATED_TOOL=${EXP}/tooling/agent_data_venv/bin/agent-data-tools
META=${EXP}/data/stage4_rp/meta_middle_format.json
OUT=${EXP}/data/stage5_swift/data_swift_format
LOG=${EXP}/data/stage5_swift/convert_swift.stdout.log

mkdir -p "${OUT}"
if [ -x "${ISOLATED_TOOL}" ]; then
  TOOL=${ISOLATED_TOOL}
else
  TOOL=${DEFAULT_TOOL}
fi

"${TOOL}" convert-swift-messages \
  -i "${META}" \
  -o "${OUT}" \
  -w 8 \
  >"${LOG}" 2>&1
code=$?
printf '%s\n' "${code}" >"${EXP}/data/stage5_swift/convert_swift.exitcode"
exit "${code}"

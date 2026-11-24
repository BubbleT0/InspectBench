#!/bin/bash
set -euo pipefail

EXP=/path/to/experiments/trajectory_r1
RUN=${EXP}/output/full_16gpu_tp2_pp4_cp1_dp2_rp1_epoch1_gbs8_stratval/v0-20260820-172425
EVAL=${EXP}/eval/stratified_validation_20260821
LOG=${EVAL}/logs

mkdir -p "${LOG}" "${EVAL}/checkpoint-250" "${EVAL}/checkpoint-500" "${EVAL}/checkpoint-740"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_server() {
  local name=$1
  local devices=$2
  local port=$3
  CUDA_VISIBLE_DEVICES="${devices}" python3 -m vllm.entrypoints.openai.api_server \
    --model "${RUN}/${name}" \
    --served-model-name "${name}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    >"${LOG}/${name}_server.log" 2>&1 &
  pids+=("$!")
}

start_server checkpoint-250 0,1 8100
start_server checkpoint-500 2,3 8101
start_server checkpoint-740 4,5 8102

python3 - "${pids[@]}" <<'PY'
import json
import sys
import time
import urllib.request

ports = (8100, 8101, 8102)
pids = [int(x) for x in sys.argv[1:]]
deadline = time.time() + 1200
ready = set()
while time.time() < deadline:
    for port in ports:
        if port in ready:
            continue
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                payload = json.load(response)
            if payload.get("data"):
                ready.add(port)
                print(f"server_ready port={port} model={payload['data'][0]['id']}", flush=True)
        except Exception:
            pass
    if len(ready) == len(ports):
        break
    if any(not __import__("os").path.exists(f"/proc/{pid}") for pid in pids):
        raise SystemExit("a vLLM server exited before readiness")
    time.sleep(5)
if len(ready) != len(ports):
    raise SystemExit(f"server readiness timeout: {sorted(ready)}")
PY

eval_pids=()
for item in checkpoint-250:8100 checkpoint-500:8101 checkpoint-740:8102; do
  name=${item%:*}
  port=${item#*:}
  bash "${EXP}/scripts/run_checkpoint_validation_eval.sh" "${name}" "http://127.0.0.1:${port}" \
    >"${LOG}/${name}_validation.log" 2>&1 &
  eval_pids+=("$!")
done

failed=0
for pid in "${eval_pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done
if [ "${failed}" -ne 0 ]; then
  echo "checkpoint validation failed processes=${failed}" >&2
  exit 4
fi

echo "three-checkpoint validation completed"

#!/usr/bin/env bash
# Pipeline stage runner.
#
# Historical 0706 helper modes:
#   all | decode | convert
#
# Generic mandatory stage 1.5 (run after stage-1 shards are merged/deduped):
#   structured-qc <stage1-openai-dir> <new-empty-qc-dir> [structured_qc.py flags]
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=/path/to/storage/research/backflow/v0.3-0706
EXTRACTED="$ROOT/extracted"
DECODED="$ROOT/decoded"
OPENAI="$ROOT/openai_per_case"
mode="${1:-all}"   # all | decode | convert | structured-qc

if [ "$mode" = "structured-qc" ]; then
  [ "$#" -ge 3 ] || {
    echo "usage: $0 structured-qc <stage1-openai-dir> <new-empty-qc-dir> [structured_qc.py flags]" >&2
    exit 2
  }
  input_dir=$2
  output_dir=$3
  shift 3
  python3 "$SCRIPT_DIR/structured_qc.py" \
    --input "$input_dir" \
    --output "$output_dir" \
    "$@"
  [ -s "$output_dir/meta_openai.json" ]
  [ -s "$output_dir/structured_qc_summary.json" ]
  [ -s "$output_dir/structured_qc_audit.jsonl" ]
  echo "stage 1.5 complete -> $output_dir"
  exit 0
fi

mkdir -p "$DECODED" "$OPENAI"

if [ "$mode" = "decode" ] || [ "$mode" = "all" ]; then
  n=0
  for d in "$EXTRACTED"/*/; do
    name=$(basename "$d"); h="$d/.harness"
    [ -f "$h" ] || continue
    out="$DECODED/$name"
    [ -d "$out/.harness/sessions" ] && { continue; }   # 幂等
    rm -rf "$out"; mkdir -p "$out"
    tar -xzf "$h" -C "$out" 2>/dev/null && n=$((n+1))
  done
  echo "decoded $n cases -> $DECODED"
fi

if [ "$mode" = "convert" ] || [ "$mode" = "all" ]; then
  n=0; ok=0
  for d in "$DECODED"/*/; do
    name=$(basename "$d"); H="$d/.harness"
    [ -d "$H/sessions" ] || continue
    python3 "$ROOT/convert_agen.py" --harness "$H" --out "$OPENAI/$name" >/dev/null 2>>"$ROOT/convert.err" && ok=$((ok+1))
    n=$((n+1))
  done
  echo "converted $ok/$n cases -> $OPENAI"
fi

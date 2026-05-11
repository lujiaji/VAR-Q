#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/var/varq/base/VAR-VARQ-8.json}"
REF_BATCH="${2:-}"
OUT_DIR="${3:-$ROOT/Benchmark/output/VAR/eval/images}"

if [[ -z "$REF_BATCH" ]]; then
  echo "Usage: bash scripts/eval_VAR.sh [config.json] <reference_batch.npz> [output_dir]" >&2
  exit 1
fi

python "$ROOT/scripts/inference_multi_VAR.py" \
  --config "$CONFIG" \
  --save_path "$OUT_DIR"

python "$ROOT/Benchmark/OpenAI-tool/evaluator.py" \
  "$REF_BATCH" \
  "${OUT_DIR}.npz"

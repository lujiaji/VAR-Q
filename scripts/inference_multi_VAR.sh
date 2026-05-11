#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/var/varq/base/VAR-VARQ-8.json}"
SAVE_PATH="${2:-$ROOT/Benchmark/output/VAR/images}"

python "$ROOT/scripts/inference_multi_VAR.py" \
  --config "$CONFIG" \
  --total_iters 1000 \
  --batch_size 50 \
  --save_path "$SAVE_PATH"

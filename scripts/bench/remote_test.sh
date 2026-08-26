#!/usr/bin/env bash
# Sync the repo to the A100 box and run a command inside the kivi_bench container.
# Usage: scripts/bench/remote_test.sh '<command run from /work/VAR-Q>'
set -euo pipefail
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)/"
rsync -az -e 'ssh -o BatchMode=yes' \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='scripts/output' \
  "$LOCAL_DIR" A100:work/kv-quant-eval/VAR-Q/
ssh -o BatchMode=yes A100 \
  "docker exec -e CUDA_VISIBLE_DEVICES=0 kivi_bench bash -lc 'cd /work/VAR-Q && ${1}'"

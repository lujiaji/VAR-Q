#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT_DIR}/configs/var/varq/base/VAR-VARQ-8.json}"
SAVE_DIR="${2:-${ROOT_DIR}/scripts/output/var}"

if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/VAR:${PYTHONPATH:-}"

python "${ROOT_DIR}/scripts/inference_multi_VAR.py" \
  --config "${CONFIG}" \
  --save_path "${SAVE_DIR}" \
  "$@"

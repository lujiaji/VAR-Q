#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LONGLIVE_DIR="${ROOT_DIR}/third_party/LongLive"
CONFIG="${1:-${ROOT_DIR}/configs/longlive/varq/base/LL-VARQ-8.json}"

if [[ $# -gt 0 ]]; then shift; fi
if [[ "${1:-}" == "--" ]]; then shift; fi

if [[ ! -d "${LONGLIVE_DIR}" ]]; then
  echo "Missing third_party/LongLive. Clone https://github.com/NVlabs/LongLive to ${LONGLIVE_DIR}." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/inference_LongLive.sh [config] -- <upstream longlive inference command>" >&2
  echo "Example: bash scripts/inference_LongLive.sh configs/longlive/varq/base/LL-VARQ-4.json -- python sample.py ..." >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}:${LONGLIVE_DIR}:${PYTHONPATH:-}"
export LONGLIVE_VARQ_CONFIG="${CONFIG}"
export VARQ_CONFIG_FILE="${CONFIG}"
export VARQ_BACKEND="longlive"

cd "${LONGLIVE_DIR}"
"$@"

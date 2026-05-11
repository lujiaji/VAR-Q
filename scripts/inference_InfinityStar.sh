#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFINITYSTAR_DIR="${ROOT_DIR}/third_party/InfinityStar"
CONFIG="${1:-${ROOT_DIR}/configs/infinitystar/varq/base/InfinityStar-VARQ-8.json}"

if [[ $# -gt 0 ]]; then shift; fi

if [[ ! -d "${INFINITYSTAR_DIR}" ]]; then
  echo "Missing third_party/InfinityStar. Clone https://github.com/FoundationVision/InfinityStar to ${INFINITYSTAR_DIR}." >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${INFINITYSTAR_DIR}:${PYTHONPATH:-}"
export INFINITYSTAR_VARQ_CONFIG="${CONFIG}"

python "${ROOT_DIR}/scripts/run_infinitystar_inference.py" "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LONGLIVE_DIR="${LONGLIVE_DIR:-${ROOT_DIR}/third_party/LongLive}"
DEFAULT_CONFIG="${ROOT_DIR}/configs/longlive/varq/base/LL-VARQ-8.json"

if [[ "${1:-}" == "--" ]]; then
  CONFIG="${DEFAULT_CONFIG}"
  shift
elif [[ $# -gt 0 ]]; then
  CONFIG="$1"
  shift
  if [[ "${1:-}" == "--" ]]; then shift; fi
else
  CONFIG="${DEFAULT_CONFIG}"
fi

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${ROOT_DIR}/${CONFIG}"
fi

if [[ ! -d "${LONGLIVE_DIR}" ]]; then
  echo "Missing third_party/LongLive. Clone https://github.com/NVlabs/LongLive to ${LONGLIVE_DIR}." >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing VAR-Q config: ${CONFIG}" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${LONGLIVE_DIR}:${PYTHONPATH:-}"
export LONGLIVE_VARQ_CONFIG="${CONFIG}"
export VARQ_CONFIG_FILE="${CONFIG}"
export VARQ_BACKEND="longlive"
export LONGLIVE_VARQ_COMMIT_TIMESTEP="${LONGLIVE_VARQ_COMMIT_TIMESTEP:-0}"
export VARQ_FUSED_FLASH_EXT_PATH="${VARQ_FUSED_FLASH_EXT_PATH:-${ROOT_DIR}/build/varq_fused_flash}"

exec python "${ROOT_DIR}/scripts/run_longlive_inference.py" \
  --config "${CONFIG}" \
  --upstream-dir "${LONGLIVE_DIR}" \
  --entrypoint "${LONGLIVE_ENTRYPOINT:-inference.py}" \
  --pipeline-module "${LONGLIVE_PIPELINE_MODULE:-pipeline.causal_inference}" \
  --pipeline-class "${LONGLIVE_PIPELINE_CLASS:-CausalInferencePipeline}" \
  -- "$@"

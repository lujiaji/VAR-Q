#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_DIR="${SELF_FORCING_DIR:-${ROOT_DIR}/third_party/Self-Forcing}"
DEFAULT_CONFIG="${ROOT_DIR}/configs/self_forcing/varq/base/SF-VARQ-8.json"

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

if [[ ! -d "${SF_DIR}" ]]; then
  echo "Missing third_party/Self-Forcing. Clone https://github.com/guandeh17/Self-Forcing to ${SF_DIR}." >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing VAR-Q config: ${CONFIG}" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${SF_DIR}:${PYTHONPATH:-}"
export SELF_FORCING_VARQ_CONFIG="${CONFIG}"
export VARQ_CONFIG_FILE="${CONFIG}"
export VARQ_BACKEND="self_forcing"
export SELF_FORCING_VARQ_COMMIT_TIMESTEP="${SELF_FORCING_VARQ_COMMIT_TIMESTEP:-0}"
export VARQ_FUSED_FLASH_EXT_PATH="${VARQ_FUSED_FLASH_EXT_PATH:-${ROOT_DIR}/build/varq_fused_flash}"

exec python "${ROOT_DIR}/scripts/run_self_forcing_inference.py" \
  --config "${CONFIG}" \
  --upstream-dir "${SF_DIR}" \
  --entrypoint "${SELF_FORCING_ENTRYPOINT:-inference.py}" \
  --pipeline-module "${SELF_FORCING_PIPELINE_MODULE:-pipeline.causal_inference}" \
  --pipeline-class "${SELF_FORCING_PIPELINE_CLASS:-CausalInferencePipeline}" \
  -- "$@"

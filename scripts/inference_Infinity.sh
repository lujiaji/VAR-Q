#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFINITY_DIR="${ROOT_DIR}/third_party/Infinity"
CONFIG="${1:-${ROOT_DIR}/configs/infinity/varq/base/Infinity-VARQ-8.json}"
PROMPT="${2:-a high-resolution photograph of a corgi wearing sunglasses}"
SAVE_FILE="${3:-${ROOT_DIR}/scripts/output/infinity.png}"

if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi

if [[ ! -d "${INFINITY_DIR}" ]]; then
  echo "Missing third_party/Infinity. Clone https://github.com/FoundationVision/Infinity to ${INFINITY_DIR}." >&2
  exit 1
fi

if [[ -z "${INFINITY_MODEL_PATH:-}" ]]; then
  echo "Set INFINITY_MODEL_PATH to the upstream Infinity checkpoint path." >&2
  exit 1
fi
if [[ -z "${INFINITY_TEXT_ENCODER_CKPT:-}" ]]; then
  echo "Set INFINITY_TEXT_ENCODER_CKPT to the upstream text encoder checkpoint path." >&2
  exit 1
fi

mkdir -p "$(dirname "${SAVE_FILE}")"
export PYTHONPATH="${ROOT_DIR}:${INFINITY_DIR}:${PYTHONPATH:-}"
export VARQ_CONFIG_FILE="${CONFIG}"

ARGS=()
ARGS+=(--pn "${INFINITY_PN:-1M}")
ARGS+=(--model_path "${INFINITY_MODEL_PATH}")
ARGS+=(--text_encoder_ckpt "${INFINITY_TEXT_ENCODER_CKPT}")
if [[ -n "${INFINITY_VAE_PATH:-}" ]]; then ARGS+=(--vae_path "${INFINITY_VAE_PATH}"); fi
if [[ -n "${INFINITY_VAE_TYPE:-}" ]]; then ARGS+=(--vae_type "${INFINITY_VAE_TYPE}"); fi

python "${ROOT_DIR}/scripts/run_infinity_inference.py" \
  --varq_config "${CONFIG}" \
  --prompt "${PROMPT}" \
  --save_file "${SAVE_FILE}" \
  "${ARGS[@]}" \
  "$@"

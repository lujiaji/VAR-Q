#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FLASH_ATTN_SOURCE="${FLASH_ATTN_SOURCE:-${ROOT_DIR}/third_party/flash-attention}"
BUILD_DIR="${VARQ_FUSED_BUILD_DIR:-${ROOT_DIR}/build/varq_fused_flash}"

if [[ ! -f "${FLASH_ATTN_SOURCE}/csrc/flash_attn/src/flash_fwd_kernel.h" ]]; then
  echo "Missing flash-attention v2.7.3 source: ${FLASH_ATTN_SOURCE}" >&2
  echo "Clone it with submodules or set FLASH_ATTN_SOURCE to that checkout." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
python "${ROOT_DIR}/VAR_Q/csrc/fused_flash/build_extension.py" \
  --flash-attn-src "${FLASH_ATTN_SOURCE}" \
  --build-dir "${BUILD_DIR}" \
  "$@"

cd "${ROOT_DIR}"
VARQ_FUSED_FLASH_EXT_PATH="${BUILD_DIR}" python - <<'PY'
from VAR_Q.fused import flash_dequant_cuda

status = flash_dequant_cuda.availability()
print("loader_available", status.available)
print("loader_message", status.message)
if not status.available:
    raise SystemExit(1)
print("backend_info", flash_dequant_cuda.load_extension().backend_info())
PY

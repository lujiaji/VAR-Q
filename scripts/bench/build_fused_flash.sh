#!/usr/bin/env bash
# Build the optional VAR-Q fused FlashAttention CUDA backend on the A100 box.
#
# This script follows the project workflow: code lives locally, then
# remote_test.sh syncs the repo to /work/VAR-Q inside the kivi_bench container.
# The remote command prepares a cached flash-attention v2.7.3 source checkout
# for the CUDA/CUTLASS work and builds the local _varq_fused_flash bridge.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_TEST="$ROOT_DIR/scripts/bench/remote_test.sh"

if [[ ! -x "$REMOTE_TEST" ]]; then
  echo "missing executable remote helper: $REMOTE_TEST" >&2
  exit 1
fi

"$REMOTE_TEST" '
set -euo pipefail
SRC_DIR=/work/flash-attention-v2.7.3
BUILD_DIR=/work/VAR-Q/build/varq_fused_flash
mkdir -p /work /work/VAR-Q/build
if [ ! -d "$SRC_DIR/.git" ]; then
  git clone --depth 1 --branch v2.7.3 https://github.com/Dao-AILab/flash-attention.git "$SRC_DIR"
fi
cd "$SRC_DIR"
git submodule update --init csrc/cutlass
mkdir -p "$BUILD_DIR"
cd /work/VAR-Q
python VAR_Q/csrc/fused_flash/build_extension.py \
  --flash-attn-src "$SRC_DIR" \
  --build-dir "$BUILD_DIR"
VARQ_FUSED_FLASH_EXT_PATH="$BUILD_DIR" python - <<PY
from VAR_Q.fused import flash_dequant_cuda
status = flash_dequant_cuda.availability()
print("loader_available", status.available)
print("loader_message", status.message)
print("loader_module_path", status.module_path)
if not status.available:
    raise SystemExit(1)
print("backend_info", flash_dequant_cuda.load_extension().backend_info())
PY
python - <<PY
from pathlib import Path
src = Path("/work/flash-attention-v2.7.3")
build = Path("/work/VAR-Q/build/varq_fused_flash")
print("flash_attn_source", src)
print("build_dir", build)
print("has_flash_fwd_kernel", (src / "csrc/flash_attn/src/flash_fwd_kernel.h").exists())
print("has_cutlass", (src / "csrc/cutlass/include/cutlass/cutlass.h").exists())
PY
'

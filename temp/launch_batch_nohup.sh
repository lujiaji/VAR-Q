#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for VAR-Q batch experiments.
# Supports custom output path, GPU list for quality eval, and bench CUDA device.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARQ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_DIR="${VARQ_ROOT}/temp/infinity_varq_sageattn_sweep"
CHECKPOINTS_DIR="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar"
OUTPUT_DIR="${VARQ_ROOT}/results"
GPU_LIST="0,1,2,3"
BENCH_CUDA_DEVICE="0"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_FILE=""

WARMUP=1
RUNS=3
SEED=42
PROMPT="A cinematic shot of a cat walking in the rain"
ENABLE_MAX_BS=0
SKIP_BENCH=0
SKIP_QUALITY=0

usage() {
  cat <<EOF
Usage:
  bash temp/launch_batch_nohup.sh --output_dir <path> [options]

Required:
  --output_dir <path>               Output root for all experiment results

Optional:
  --config_dir <path>               Config folder (default: ${CONFIG_DIR})
  --checkpoints_dir <path>          InfinityStar checkpoints dir
  --gpu_list "0,1,2,3"              GPUs for VBench quality eval
  --bench_cuda_device "0"           CUDA_VISIBLE_DEVICES for bench only
  --python_bin <python>             Python executable (default: \$PYTHON_BIN or python)
  --log_file <path>                 nohup log file (default: <output_dir>/nohup_batch.log)
  --warmup <int>                    Throughput warmup runs
  --runs <int>                      Throughput measured runs
  --seed <int>                      Seed for bench
  --prompt <text>                   Prompt for bench
  --enable_max_bs                   Enable max_bs probing path
  --skip_bench                      Skip bench stage
  --skip_quality                    Skip quality stage
  -h, --help                        Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_dir) CONFIG_DIR="$2"; shift 2 ;;
    --checkpoints_dir) CHECKPOINTS_DIR="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpu_list) GPU_LIST="$2"; shift 2 ;;
    --bench_cuda_device) BENCH_CUDA_DEVICE="$2"; shift 2 ;;
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    --log_file) LOG_FILE="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --enable_max_bs) ENABLE_MAX_BS=1; shift ;;
    --skip_bench) SKIP_BENCH=1; shift ;;
    --skip_quality) SKIP_QUALITY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "ERROR: --output_dir is required"
  usage
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
if [[ -z "${LOG_FILE}" ]]; then
  LOG_FILE="${OUTPUT_DIR}/nohup_batch.log"
fi
mkdir -p "$(dirname "${LOG_FILE}")"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_batch_experiments.py"
  --config_dir "${CONFIG_DIR}"
  --checkpoints_dir "${CHECKPOINTS_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --gpu_list "${GPU_LIST}"
  --python_bin "${PYTHON_BIN}"
  --warmup "${WARMUP}"
  --runs "${RUNS}"
  --prompt "${PROMPT}"
  --seed "${SEED}"
  --bench_cuda_device "${BENCH_CUDA_DEVICE}"
)

if [[ "${ENABLE_MAX_BS}" -eq 1 ]]; then
  CMD+=(--enable_max_bs)
fi
if [[ "${SKIP_BENCH}" -eq 1 ]]; then
  CMD+=(--skip_bench)
fi
if [[ "${SKIP_QUALITY}" -eq 1 ]]; then
  CMD+=(--skip_quality)
fi

echo "Launching with nohup..."
echo "output_dir: ${OUTPUT_DIR}"
echo "gpu_list: ${GPU_LIST}"
echo "bench_cuda_device: ${BENCH_CUDA_DEVICE}"
echo "log_file: ${LOG_FILE}"
echo "command: ${CMD[*]}"

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Started. PID=${PID}"
echo "Tail log: tail -f \"${LOG_FILE}\""

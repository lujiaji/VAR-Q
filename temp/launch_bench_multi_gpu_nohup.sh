#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARQ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_DIR="${VARQ_ROOT}/temp/infinity_varq_sageattn_sweep"
CONFIGS=""
CHECKPOINTS_DIR="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar"
OUTPUT_DIR="${VARQ_ROOT}/results_bench_only"
GPU_LIST="0,1,2,3"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_FILE=""

WARMUP=1
RUNS=3
SEED=42
PROMPT="A cinematic shot of a cat walking in the rain"
ENABLE_MAX_BS=0
ONLY_MAX_BS=0

usage() {
  cat <<EOF
Usage:
  bash temp/launch_bench_multi_gpu_nohup.sh --output_dir <path> [options]

Required:
  --output_dir <path>                 Bench-only output root

Optional:
  --config_dir <path>                 Default config folder
  --configs "A,B,C"                   Run selected configs only (name/stem/path)
  --checkpoints_dir <path>            InfinityStar checkpoints dir
  --gpu_list "0,1,2,3"                Multi-GPU worker list
  --python_bin <python>               Python executable
  --log_file <path>                   nohup log file (default: <output_dir>/nohup_bench.log)
  --warmup <int>                      Warmup runs
  --runs <int>                        Measured runs
  --seed <int>                        Bench seed
  --prompt <text>                     Bench prompt
  --enable_max_bs                     Enable max_bs probing
  --only_max_bs                       Skip base single-bs bench and only run max_bs probing
  -h, --help                          Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_dir) CONFIG_DIR="$2"; shift 2 ;;
    --configs) CONFIGS="$2"; shift 2 ;;
    --checkpoints_dir) CHECKPOINTS_DIR="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpu_list) GPU_LIST="$2"; shift 2 ;;
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    --log_file) LOG_FILE="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --enable_max_bs) ENABLE_MAX_BS=1; shift ;;
    --only_max_bs) ONLY_MAX_BS=1; shift ;;
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
  LOG_FILE="${OUTPUT_DIR}/nohup_bench.log"
fi
mkdir -p "$(dirname "${LOG_FILE}")"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_bench_multi_gpu.py"
  --config_dir "${CONFIG_DIR}"
  --checkpoints_dir "${CHECKPOINTS_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --gpu_list "${GPU_LIST}"
  --python_bin "${PYTHON_BIN}"
  --warmup "${WARMUP}"
  --runs "${RUNS}"
  --prompt "${PROMPT}"
  --seed "${SEED}"
)

if [[ -n "${CONFIGS}" ]]; then
  CMD+=(--configs "${CONFIGS}")
fi
if [[ "${ENABLE_MAX_BS}" -eq 1 ]]; then
  CMD+=(--enable_max_bs)
fi
if [[ "${ONLY_MAX_BS}" -eq 1 ]]; then
  CMD+=(--only_max_bs)
fi

echo "Launching bench-only multi-GPU with nohup..."
echo "output_dir: ${OUTPUT_DIR}"
echo "gpu_list: ${GPU_LIST}"
echo "configs: ${CONFIGS:-<all in config_dir>}"
echo "log_file: ${LOG_FILE}"
echo "command: ${CMD[*]}"

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "Started. PID=${PID}"
echo "Tail log: tail -f \"${LOG_FILE}\""


# bash /home/jiaji_lu/AR/VAR-Q/temp/launch_bench_multi_gpu_nohup.sh \
#   --output_dir /data/jiaji_lu/VAR-Q/results_bench_maxbs_v2 \
#   --config_dir /home/jiaji_lu/AR/VAR-Q/temp/infinity_varq_sageattn_sweep \
#   --gpu_list "1,2" \
#   --python_bin /home/jiaji_lu/conda/envs/varq/bin/python \
#   --enable_max_bs

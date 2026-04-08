#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARQ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ORIG_CONFIG="${SCRIPT_DIR}/Infinity-VAR_Q-8.json"
SAGEATTN_CONFIG="${SCRIPT_DIR}/Infinity-VAR_Q-8-sageattn-only.json"
CHECKPOINTS_DIR="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar"
OUTPUT_DIR="${VARQ_ROOT}/results_maxbs_baselines"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ORIG="0"
GPU_SAGEATTN="1"
WARMUP=1
RUNS=3
SEED=42
PROMPT="A cinematic shot of a cat walking in the rain"
LOG_FILE=""

usage() {
  cat <<EOF
Usage:
  bash temp/launch_maxbs_infinity_vs_sageattn_nohup.sh --output_dir <path> [options]

Required:
  --output_dir <path>              Output root

Optional:
  --orig_config <path>             Original InfinityStar config (default: temp/Infinity-VAR_Q-8.json)
  --sageattn_config <path>         InfinityStar+sageattn config (default: temp/Infinity-VAR_Q-8-sageattn-only.json)
  --checkpoints_dir <path>         Checkpoints root
  --gpu_orig <id>                  GPU id for original task (default: 0)
  --gpu_sageattn <id>              GPU id for sageattn task (default: 1)
  --python_bin <python>            Python executable
  --warmup <int>                   Warmup runs
  --runs <int>                     Timed runs
  --seed <int>                     Base seed
  --prompt <text>                  Prompt
  --log_file <path>                Master log file (default: <output_dir>/nohup_maxbs.log)
  -h, --help                       Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --orig_config) ORIG_CONFIG="$2"; shift 2 ;;
    --sageattn_config) SAGEATTN_CONFIG="$2"; shift 2 ;;
    --checkpoints_dir) CHECKPOINTS_DIR="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpu_orig) GPU_ORIG="$2"; shift 2 ;;
    --gpu_sageattn) GPU_SAGEATTN="$2"; shift 2 ;;
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --log_file) LOG_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "ERROR: --output_dir is required"
  usage
  exit 1
fi

if [[ ! -f "${ORIG_CONFIG}" ]]; then
  echo "ERROR: orig_config not found: ${ORIG_CONFIG}"
  exit 1
fi
if [[ ! -f "${SAGEATTN_CONFIG}" ]]; then
  echo "ERROR: sageattn_config not found: ${SAGEATTN_CONFIG}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
if [[ -z "${LOG_FILE}" ]]; then
  LOG_FILE="${OUTPUT_DIR}/nohup_maxbs.log"
fi
mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "${OUTPUT_DIR}/original/logs" "${OUTPUT_DIR}/sageattn/logs"

BENCH_SCRIPT="${SCRIPT_DIR}/bench_config.py"
ORIG_LOG="${OUTPUT_DIR}/original/logs/bench.log"
SAGE_LOG="${OUTPUT_DIR}/sageattn/logs/bench.log"

INNER_SCRIPT="$(cat <<EOF
set -euo pipefail
echo "[\$(date '+%F %T')] Start original + sageattn max_bs benchmark"
echo "orig_config: ${ORIG_CONFIG}"
echo "sageattn_config: ${SAGEATTN_CONFIG}"
echo "gpu_orig: ${GPU_ORIG}, gpu_sageattn: ${GPU_SAGEATTN}"

CUDA_VISIBLE_DEVICES="${GPU_ORIG}" "${PYTHON_BIN}" "${BENCH_SCRIPT}" \\
  --config_file "${ORIG_CONFIG}" \\
  --checkpoints_dir "${CHECKPOINTS_DIR}" \\
  --output_dir "${OUTPUT_DIR}/original/bench" \\
  --warmup "${WARMUP}" --runs "${RUNS}" \\
  --prompt "${PROMPT}" \\
  --seed "${SEED}" \\
  --enable_max_bs \\
  --save_video \\
  > "${ORIG_LOG}" 2>&1 &
PID_ORIG=\$!

CUDA_VISIBLE_DEVICES="${GPU_SAGEATTN}" "${PYTHON_BIN}" "${BENCH_SCRIPT}" \\
  --config_file "${SAGEATTN_CONFIG}" \\
  --checkpoints_dir "${CHECKPOINTS_DIR}" \\
  --output_dir "${OUTPUT_DIR}/sageattn/bench" \\
  --warmup "${WARMUP}" --runs "${RUNS}" \\
  --prompt "${PROMPT}" \\
  --seed "$((SEED + 1))" \\
  --enable_max_bs \\
  --save_video \\
  > "${SAGE_LOG}" 2>&1 &
PID_SAGE=\$!

echo "PID_ORIG=\${PID_ORIG}"
echo "PID_SAGE=\${PID_SAGE}"
wait "\${PID_ORIG}" || true
wait "\${PID_SAGE}" || true
echo "[\$(date '+%F %T')] Done original + sageattn max_bs benchmark"
EOF
)"

echo "Launching with nohup..."
echo "output_dir: ${OUTPUT_DIR}"
echo "log_file: ${LOG_FILE}"
echo "orig_log: ${ORIG_LOG}"
echo "sage_log: ${SAGE_LOG}"

nohup bash -lc "${INNER_SCRIPT}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "Started. PID=${PID}"
echo "Tail master log: tail -f \"${LOG_FILE}\""
echo "Tail original log: tail -f \"${ORIG_LOG}\""
echo "Tail sageattn log: tail -f \"${SAGE_LOG}\""

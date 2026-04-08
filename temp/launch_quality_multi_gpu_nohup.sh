#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARQ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_DIR="${VARQ_ROOT}/temp/infinity_varq_sageattn_sweep"
CONFIGS=""
CHECKPOINTS_DIR="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar"
OUTPUT_DIR="${VARQ_ROOT}/results_quality_100"
GPU_LIST="0,1,2,3"
JOBS_PER_GPU=1
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_FILE=""
MAX_VIDEOS=100
PROMPTS_JSON="${VARQ_ROOT}/temp/vbench_100_prompts.json"
DELAY_SECONDS=7200

usage() {
  cat <<EOF
Usage:
  bash temp/launch_quality_multi_gpu_nohup.sh --output_dir <path> [options]

Required:
  --output_dir <path>                 Quality output root

Optional:
  --config_dir <path>                 Default config folder
  --configs "A,B,C"                   Run selected configs only (name/stem/path)
  --checkpoints_dir <path>            InfinityStar checkpoints dir
  --gpu_list "0,1,2,3"                GPU pool
  --jobs_per_gpu <int>                Concurrent jobs per GPU (default: 1)
  --python_bin <python>               Python executable
  --max_videos <int>                  Number of prompts/videos (default: 100)
  --prompts_json <path>               Prompt subset json (default: temp/vbench_100_prompts.json)
  --delay_seconds <int>               Delay before start (default: 7200 = 2h)
  --no_delay                          Start immediately (equivalent to --delay_seconds 0)
  --log_file <path>                   nohup log (default: <output_dir>/nohup_quality.log)
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
    --jobs_per_gpu) JOBS_PER_GPU="$2"; shift 2 ;;
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    --max_videos) MAX_VIDEOS="$2"; shift 2 ;;
    --prompts_json) PROMPTS_JSON="$2"; shift 2 ;;
    --delay_seconds) DELAY_SECONDS="$2"; shift 2 ;;
    --no_delay) DELAY_SECONDS=0; shift ;;
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

mkdir -p "${OUTPUT_DIR}"
if [[ -z "${LOG_FILE}" ]]; then
  LOG_FILE="${OUTPUT_DIR}/nohup_quality.log"
fi
mkdir -p "$(dirname "${LOG_FILE}")"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_quality_multi_gpu.py"
  --config_dir "${CONFIG_DIR}"
  --checkpoints_dir "${CHECKPOINTS_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --gpu_list "${GPU_LIST}"
  --jobs_per_gpu "${JOBS_PER_GPU}"
  --python_bin "${PYTHON_BIN}"
  --max_videos "${MAX_VIDEOS}"
  --prompts_json "${PROMPTS_JSON}"
)

if [[ -n "${CONFIGS}" ]]; then
  CMD+=(--configs "${CONFIGS}")
fi

echo "Launching quality-100 multi-GPU with nohup..."
echo "output_dir: ${OUTPUT_DIR}"
echo "gpu_list: ${GPU_LIST}"
echo "jobs_per_gpu: ${JOBS_PER_GPU}"
echo "configs: ${CONFIGS:-<all in config_dir>}"
echo "delay_seconds: ${DELAY_SECONDS}"
echo "log_file: ${LOG_FILE}"
echo "command: ${CMD[*]}"

if [[ "${DELAY_SECONDS}" -lt 0 ]]; then
  echo "ERROR: --delay_seconds must be >= 0"
  exit 1
fi

CMD_STR="$(printf '%q ' "${CMD[@]}")"
if [[ "${DELAY_SECONDS}" -gt 0 ]]; then
  START_AT="$(date -d "+${DELAY_SECONDS} seconds" '+%F %T')"
  echo "scheduled_start: ${START_AT}"
  nohup bash -lc "sleep ${DELAY_SECONDS}; ${CMD_STR}" > "${LOG_FILE}" 2>&1 &
else
  nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
fi
PID=$!
echo "Started. PID=${PID}"
echo "Tail log: tail -f \"${LOG_FILE}\""

# bash /home/jiaji_lu/AR/VAR-Q/temp/launch_quality_multi_gpu_nohup.sh \
#   --output_dir /data/jiaji_lu/VAR-Q/results_quality_100_custom_quality \
#   --config_dir /home/jiaji_lu/AR/VAR-Q/temp/infinity_varq_sageattn_sweep \
#   --gpu_list "4,5,6" \
#   --jobs_per_gpu 1 \
#   --python_bin /home/jiaji_lu/conda/envs/varq/bin/python \
#   --no_delay

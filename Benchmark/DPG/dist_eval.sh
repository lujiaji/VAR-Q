#!/bin/bash

set -euo pipefail

# Guard against running under base by mistake.
if [[ "${CONDA_DEFAULT_ENV:-}" != "varq" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate varq
  else
    echo "[ERROR] conda not found; please activate varq before running."
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONNOUSERSITE=1
IMAGE_ROOT_PATH=$1
RESOLUTION=$2
PIC_NUM=4
PROCESSES=${PROCESSES:-3}
PORT=${PORT:-29500}
PYTHON_BIN="$(command -v python)"


${PYTHON_BIN} -m accelerate.commands.launch --num_machines 1 --num_processes $PROCESSES --multi_gpu --mixed_precision "fp16" --main_process_port $PORT \
  DPG/compute_dpg_bench.py \
  --image-root-path $IMAGE_ROOT_PATH \
  --resolution $RESOLUTION \
  --pic-num $PIC_NUM \
  --vqa-model mplug
#!/bin/bash
set -euo pipefail

# Multi-GPU VBench video generation for InfinityStar + VARQ.
# Override variables below via environment variables if needed.

PROMPTS_JSON="${PROMPTS_JSON:-evaluation/VBench_rewrited_prompt.json}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/vbench_eval_varq}"
CONFIG_FILE="${CONFIG_FILE:-/home/jiaji_lu/AR/VAR-Q/VAR_Q/Infinity-VAR_Q-8.json}"
GPU_IDS="${GPU_IDS:-3,4,5,6}"              # e.g. "0,1,2,3"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SAMPLE_START_IDX="${SAMPLE_START_IDX:-0}"
FILENAME_MODE="${FILENAME_MODE:-id}"   # id | prompt
USE_RANDOM_SEED="${USE_RANDOM_SEED:-1}" # 1 random, 0 deterministic
BASE_SEED="${BASE_SEED:-42}"

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS="${#GPUS[@]}"

TOTAL_PROMPTS=$(python3 - <<PY
import json
with open("$PROMPTS_JSON","r") as f:
    print(len(json.load(f)))
PY
)

PER_GPU=$((TOTAL_PROMPTS / NUM_GPUS))
REM=$((TOTAL_PROMPTS % NUM_GPUS))

echo "=========================================="
echo "InfinityStar+VARQ multi-GPU generation"
echo "prompts_json: $PROMPTS_JSON"
echo "total_prompts: $TOTAL_PROMPTS"
echo "gpu_ids: $GPU_IDS"
echo "num_gpus: $NUM_GPUS"
echo "num_samples: $NUM_SAMPLES"
echo "sample_start_idx: $SAMPLE_START_IDX"
echo "filename_mode: $FILENAME_MODE"
echo "output_dir: $OUTPUT_DIR"
echo "config_file: $CONFIG_FILE"
echo "=========================================="

for idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$idx]}"
    start=$((idx * PER_GPU))
    count=$PER_GPU
    if [ "$idx" -eq $((NUM_GPUS - 1)) ]; then
        count=$((PER_GPU + REM))
    fi

    extra_seed_args=()
    if [ "$USE_RANDOM_SEED" = "1" ]; then
        extra_seed_args+=(--use_random_seed)
    else
        extra_seed_args+=(--base_seed "$BASE_SEED")
    fi

    echo ""
    echo "Launch GPU $gpu: start=$start count=$count"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 evaluation/run_vbench_eval_varq.py \
        --prompts_json "$PROMPTS_JSON" \
        --output_dir "$OUTPUT_DIR" \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --config_file "$CONFIG_FILE" \
        --start_idx "$start" \
        --max_videos "$count" \
        --num_samples "$NUM_SAMPLES" \
        --sample_start_idx "$SAMPLE_START_IDX" \
        --filename_mode "$FILENAME_MODE" \
        "${extra_seed_args[@]}" \
        > "$LOG_DIR/gpu${gpu}.log" 2>&1 &
    echo "PID=$! log=$LOG_DIR/gpu${gpu}.log"
done

echo ""
echo "All jobs launched."
echo "Tail logs: tail -f $LOG_DIR/gpu*.log"
echo "Stop jobs: pkill -f run_vbench_eval_varq.py"

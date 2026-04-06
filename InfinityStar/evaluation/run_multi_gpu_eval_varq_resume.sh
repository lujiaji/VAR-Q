#!/bin/bash
set -euo pipefail

# Resume VBench: skip completed prompts, run only remaining on multi-GPU.
# Run from InfinityStar dir.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PROMPTS_JSON="${PROMPTS_JSON:-evaluation/VBench_rewrited_prompt.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/vbench_eval_varq}"
REMAINING_JSON="${REMAINING_JSON:-evaluation/VBench_remaining_prompts.json}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar}"
CONFIG_FILE="${CONFIG_FILE:-/home/jiaji_lu/AR/VAR-Q/VAR_Q/Infinity-VAR_Q-8.json}"
GPU_IDS="${GPU_IDS:-3,4,5,6}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
SAMPLE_START_IDX="${SAMPLE_START_IDX:-1}"
FILENAME_MODE="${FILENAME_MODE:-id}"
USE_RANDOM_SEED="${USE_RANDOM_SEED:-1}"
BASE_SEED="${BASE_SEED:-42}"

echo "=========================================="
echo "VBench resume: check completed, run remaining"
echo "=========================================="

python3 evaluation/check_completed_prompts.py \
    --output_dir "$OUTPUT_DIR" \
    --prompts_json "$PROMPTS_JSON" \
    --remaining_json "$REMAINING_JSON" \
    --num_samples "$NUM_SAMPLES" \
    --sample_start_idx "$SAMPLE_START_IDX"

REMAINING_COUNT=$(python3 -c "import json; print(len(json.load(open('$REMAINING_JSON'))))")
if [ "$REMAINING_COUNT" -eq 0 ]; then
    echo "All prompts completed. Nothing to do."
    exit 0
fi

echo ""
echo "Launching multi-GPU generation for $REMAINING_COUNT remaining prompts..."
echo ""

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS="${#GPUS[@]}"

PER_GPU=$((REMAINING_COUNT / NUM_GPUS))
REM=$((REMAINING_COUNT % NUM_GPUS))

echo "gpu_ids: $GPU_IDS, num_gpus: $NUM_GPUS"
echo "remaining_prompts: $REMAINING_COUNT"
echo "num_samples: $NUM_SAMPLES, sample_start_idx: $SAMPLE_START_IDX"
echo "output_dir: $OUTPUT_DIR"
echo ""

extra_seed_args=()
if [ "$USE_RANDOM_SEED" = "1" ]; then
    extra_seed_args+=(--use_random_seed)
else
    extra_seed_args+=(--base_seed "$BASE_SEED")
fi

for idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$idx]}"
    start=$((idx * PER_GPU))
    count=$PER_GPU
    if [ "$idx" -eq $((NUM_GPUS - 1)) ]; then
        count=$((PER_GPU + REM))
    fi

    echo "Launch GPU $gpu: start=$start count=$count (remaining slice)"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 evaluation/run_vbench_eval_varq.py \
        --prompts_json "$REMAINING_JSON" \
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

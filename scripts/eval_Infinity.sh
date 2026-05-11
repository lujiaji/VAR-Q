#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFINITY_ROOT="$ROOT/third_party/Infinity"
TASK="${1:-geneval}"
CONFIG="${2:-$ROOT/configs/infinity/varq/base/Infinity-VARQ-8.json}"
OUT_DIR="${3:-$ROOT/Benchmark/output/Infinity/${TASK}}"

if [[ ! -d "$INFINITY_ROOT" ]]; then
  echo "Missing third-party repository: $INFINITY_ROOT" >&2
  echo "Clone Infinity into third_party/Infinity before running this script." >&2
  exit 1
fi

export PYTHONPATH="$ROOT:$INFINITY_ROOT:${PYTHONPATH:-}"

case "$TASK" in
  geneval)
    mkdir -p "$OUT_DIR/results"
    python "$ROOT/Benchmark/GenEval/infer4eval.py" \
      --config_file "$CONFIG" \
      --outdir "$OUT_DIR/images" \
      --metadata_file "$ROOT/Benchmark/GenEval/prompts/evaluation_metadata.jsonl"
    python "$ROOT/Benchmark/GenEval/evaluate_images.py" \
      "$OUT_DIR/images" \
      --outfile "$OUT_DIR/results/det.jsonl" \
      --model-config "$ROOT/Benchmark/GenEval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
      --model-path "$ROOT/Benchmark/weights/mask2former"
    python "$ROOT/Benchmark/GenEval/summary_scores.py" "$OUT_DIR/results/det.jsonl"
    ;;
  dpg)
    python "$ROOT/Benchmark/DPG/infer4eval.py" \
      --config_file "$CONFIG" \
      --outdir "$OUT_DIR"
    (
      cd "$ROOT/Benchmark"
      bash DPG/dist_eval.sh "$OUT_DIR/dpg_images" 1024
    )
    ;;
  imagereward)
    python "$ROOT/Benchmark/ImageReward/infer4eval.py" \
      --config_file "$CONFIG" \
      --outdir "$OUT_DIR"
    python "$ROOT/Benchmark/ImageReward/cal_imagereward.py" \
      --meta_file "$OUT_DIR/metadata.jsonl"
    ;;
  *)
    echo "Unsupported task: $TASK" >&2
    echo "Usage: bash scripts/eval_Infinity.sh [geneval|dpg|imagereward] [config.json] [output_dir]" >&2
    exit 1
    ;;
esac

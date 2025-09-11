# !/bin/bash

python inference_VAR.py \
    --model_depth 30 \
    --seed 0 \
    --cfg 2.0 \
    --top_k 600 \
    --total_iters 10 \
    --batch_size 10 \
    --save_path '../Benchmark/output/VAR/images'
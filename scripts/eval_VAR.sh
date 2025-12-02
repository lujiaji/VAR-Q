# !/bin/bash

python inference_VAR.py \
    --model_depth 30 \
    --seed 0 \
    --cfg 2.0 \
    --top_k 600 \
    --total_iters 1000 \
    --batch_size 50 \
    --save_path '../Benchmark/output/VAR/eval/images'

python evaluator_VAR.py \
    YOUR_PATH/VIRTUAL_imagenet256_labeled.npz \
    YOUR_PATH/YOUR_VAR_INFER_NPZ.npz \

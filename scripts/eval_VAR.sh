# !/bin/bash

python inference_VAR.py \
    --model_depth 30 \
    --seed 0 \
    --cfg 2.0 \
    --top_k 600 \
    --total_iters 1000 \
    --batch_size 50 \
    --save_path '../Benchmark/output/VAR/eval/images'

python ../Benchmark/OpenAI-tool/evaluator.py \
    YOUR_PATH/VIRTUAL_imagenet256_labeled.npz \
    ../Benchmark/output/VAR/eval/imagesYOUR_VAR_INFER_NPZ.npz \

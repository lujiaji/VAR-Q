#!/bin/bash

# VAR-Q Multi-Image Inference Script
# Most parameters are now configured in config.json

python scripts/inference_multi_VAR.py \
    --config VAR-Q/VAR_Q/VAR-VAR_Q-8.json \
    --total_iters 1000 \
    --batch_size 50 \
    --save_path 'Benchmark/outputs/VAR/images'

# Alternative usage examples:
# 
# Use only config file (no overrides):
# python scripts/inference_multi_VAR.py --save_path 'Benchmark/outputs/VAR/images'
#
# Use custom config file:
# python scripts/inference_multi_VAR.py --config custom_config.json --save_path 'Benchmark/outputs/VAR/images'
#
# Override batch settings:
# python scripts/inference_multi_VAR.py --total_iters 100 --batch_size 20
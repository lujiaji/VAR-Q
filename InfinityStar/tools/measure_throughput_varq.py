#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import argparse
import datetime
import os
import os.path as osp
import statistics
import sys
import time
import torch

sys.path.append(osp.dirname(osp.dirname(__file__)))

from infinity.utils.arg_util import Args
from tools.infer_video_720p import InferencePipe, perform_inference, load_varq_quant_config


def build_args(checkpoints_dir: str, quant_config: dict) -> Args:
    args = Args()
    args.pn = "0.90M"
    args.fps = 16
    args.video_frames = 81
    args.model_path = osp.join(checkpoints_dir, "infinitystar_8b_720p_weights")
    args.checkpoint_type = "torch_shard"
    args.vae_path = osp.join(checkpoints_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = osp.join(checkpoints_dir, "text_encoder/flan-t5-xl-official/")
    args.model_type = "infinity_qwen8b"
    args.text_channels = 2048
    args.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2"
    args.bf16 = 1
    args.use_apg = 1
    args.use_cfg = 0
    args.cfg = 34
    args.tau_image = 1
    args.tau_video = 0.4
    args.apg_norm_threshold = 0.05
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]"
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 750
    args.semantic_scales = 12
    args.max_repeat_times = 10000
    args.enable_rewriter = 0

    # VARQ config from JSON
    args.enable_quantization = int(quant_config.get("enable", False))
    args.q_bits = quant_config.get("q_bits", 8)
    args.quant_method = quant_config.get("quant_method", "G_SCALE_HEAD_DIM")
    args.qkv_format = quant_config.get("qkv_format", "BHLc")
    args.rescale_qk = int(quant_config.get("rescale_qk", False))
    args.enable_sageattn = int(quant_config.get("enable_sageattn", False))
    args.sageattn_type = str(quant_config.get("sageattn_type", "sageattn"))
    return args


def run_once(pipe: InferencePipe, args: Args, prompt: str, seed: int) -> float:
    data = {
        "seed": seed,
        "image_path": None,  # Text-to-Video
        "prompt": prompt,
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = perform_inference(pipe, data, args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0


def main():
    parser = argparse.ArgumentParser(description="Measure InfinityStar+VARQ throughput (inference only, no save).")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar",
        help="InfinityStar checkpoints root directory",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="/home/jiaji_lu/AR/VAR-Q/VAR_Q/Infinity-VAR_Q-8.json",
        help="VARQ JSON config path",
    )
    parser.add_argument("--prompt", type=str, default="A cinematic shot of a cat walking in the rain, ultra realistic.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jiaji_lu/AR/VAR-Q/Benchmark/results_txt/throughput",
    )
    parser.add_argument("--tag", type=str, default="infinitystar_varq")
    args_cmd = parser.parse_args()

    os.makedirs(args_cmd.output_dir, exist_ok=True)

    quant_config = load_varq_quant_config(args_cmd.config_file)
    args = build_args(args_cmd.checkpoints_dir, quant_config)

    print("[Info] Loading model pipeline...")
    pipe = InferencePipe(args)
    print("[Info] Model loaded. Start throughput timing...")

    for i in range(args_cmd.warmup):
        dt = run_once(pipe, args, args_cmd.prompt, args_cmd.seed + i)
        print(f"[Warmup {i+1}/{args_cmd.warmup}] {dt:.4f}s")

    times = []
    for i in range(args_cmd.runs):
        dt = run_once(pipe, args, args_cmd.prompt, args_cmd.seed + 1000 + i)
        times.append(dt)
        print(f"[Run {i+1}/{args_cmd.runs}] {dt:.4f}s")

    avg_time = statistics.mean(times)
    p50_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    frames = int(args.video_frames)
    throughput_fps_avg = frames / avg_time
    throughput_fps_p50 = frames / p50_time

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = osp.join(args_cmd.output_dir, f"{args_cmd.tag}_{now}.txt")
    with open(save_path, "w") as f:
        f.write("=== InfinityStar VARQ Throughput Report ===\n")
        f.write(f"timestamp: {now}\n")
        f.write(f"checkpoints_dir: {args_cmd.checkpoints_dir}\n")
        f.write(f"config_file: {args_cmd.config_file}\n")
        f.write(f"prompt: {args_cmd.prompt}\n")
        f.write(f"seed: {args_cmd.seed}\n")
        f.write(f"warmup: {args_cmd.warmup}\n")
        f.write(f"runs: {args_cmd.runs}\n")
        f.write(f"video_frames: {frames}\n")
        f.write("\n[quantization]\n")
        f.write(f"enable: {args.enable_quantization}\n")
        f.write(f"q_bits: {args.q_bits}\n")
        f.write(f"quant_method: {args.quant_method}\n")
        f.write(f"qkv_format: {args.qkv_format}\n")
        f.write(f"rescale_qk: {args.rescale_qk}\n")
        f.write(f"enable_sageattn: {args.enable_sageattn}\n")
        f.write(f"sageattn_type: {args.sageattn_type}\n")
        f.write("\n[timing_seconds]\n")
        f.write(f"all_runs: {times}\n")
        f.write(f"avg: {avg_time:.6f}\n")
        f.write(f"p50: {p50_time:.6f}\n")
        f.write(f"min: {min_time:.6f}\n")
        f.write(f"max: {max_time:.6f}\n")
        f.write("\n[throughput]\n")
        f.write(f"fps_avg: {throughput_fps_avg:.6f}\n")
        f.write(f"fps_p50: {throughput_fps_p50:.6f}\n")

    print("\n=== Done ===")
    print(f"avg_time: {avg_time:.4f}s")
    print(f"throughput_fps(avg): {throughput_fps_avg:.4f}")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()

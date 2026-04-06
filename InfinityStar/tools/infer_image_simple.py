#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch

sys.path.append(osp.dirname(osp.dirname(__file__)))

from infinity.utils.arg_util import Args
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from tools.run_infinity import (
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    gen_one_example,
)
from tools.infer_video_720p import load_varq_quant_config


def pick_schedule_by_frames(dynamic_resolution_h_w, h_div_w_template, pn, num_frames):
    pt2scale_schedule = dynamic_resolution_h_w[h_div_w_template][pn]["pt2scale_schedule"]
    target_key = (num_frames - 1) // 4 + 1
    if target_key in pt2scale_schedule:
        return pt2scale_schedule[target_key]
    keys = sorted(pt2scale_schedule.keys())
    nearest_key = min(keys, key=lambda k: abs(k - target_key))
    return pt2scale_schedule[nearest_key]


def main():
    parser = argparse.ArgumentParser(description="Simple InfinityStar image generation (with VARQ config).")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/gen_images/simple.jpg")
    parser.add_argument("--checkpoints_dir", type=str, default="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar")
    parser.add_argument("--config_file", type=str, default="/home/jiaji_lu/AR/VAR-Q/VAR_Q/Infinity-VAR_Q-8.json")
    parser.add_argument("--gpu", type=int, default=0)
    args_cmd = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cmd.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    quant_cfg = load_varq_quant_config(args_cmd.config_file)

    args = Args()
    args.pn = "0.90M"
    args.fps = 16
    # Use the stable video path (81 frames), then save the first frame as image.
    args.video_frames = 81
    args.model_path = osp.join(args_cmd.checkpoints_dir, "infinitystar_8b_720p_weights")
    args.checkpoint_type = "torch_shard"
    args.vae_path = osp.join(args_cmd.checkpoints_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = osp.join(args_cmd.checkpoints_dir, "text_encoder/flan-t5-xl-official/")
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

    args.enable_quantization = int(quant_cfg.get("enable", False))
    args.q_bits = quant_cfg.get("q_bits", 8)
    args.quant_method = quant_cfg.get("quant_method", "G_SCALE_HEAD_DIM")
    args.qkv_format = quant_cfg.get("qkv_format", "BHLc")
    args.rescale_qk = int(quant_cfg.get("rescale_qk", False))
    args.enable_sageattn = int(quant_cfg.get("enable_sageattn", False))
    args.sageattn_type = str(quant_cfg.get("sageattn_type", "sageattn"))

    print("[Info] Loading models...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args).float().to(device)
    infinity = load_transformer(vae, args)
    _, _, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    # generate a short clip then save the first frame as image
    scale_schedule = pick_schedule_by_frames(dynamic_resolution_h_w, h_div_w_template, args.pn, num_frames=args.video_frames)
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)
    tau = [args.tau_image] * len(scale_schedule)

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        video, _ = gen_one_example(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            args_cmd.prompt,
            negative_prompt="",
            g_seed=args_cmd.seed,
            cfg_list=args.cfg,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[0],
            vae_type=64,
            sampling_per_bits=1,
            enable_positive_prompt=0,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=get_visual_rope_embeds,
            context_info=context_info,
            noise_list=None,
        )

    if video.ndim == 4:
        # [T, H, W, 3] -> first frame
        img = video[0]
    else:
        img = video
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    os.makedirs(osp.dirname(args_cmd.output), exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args_cmd.output, img_bgr)
    print(f"[Done] saved image: {args_cmd.output}")


if __name__ == "__main__":
    main()

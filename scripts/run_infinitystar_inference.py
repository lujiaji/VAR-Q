#!/usr/bin/env python3
"""
InfinityStar 推理脚本（含 VAR-Q），用于验证修复后的生成效果。

运行方式一（本脚本，需在能正常 import torch 的环境下）:
  cd VAR-Q && python scripts/run_infinitystar_inference.py

运行方式二（使用现有 480p 脚本，推荐）:
  cd VAR-Q
  export ENABLE_VARQ=1
  export INFINITY_SCHEDULE=infinity_star_interact
  PYTHONPATH=$PWD python InfinityStar/tools/infer_video_480p.py
  生成结果在 InfinityStar/output/gen_videos/demo.mp4
"""
import os
import sys

# 保证能 import InfinityStar 和 VAR_Q
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
infinity_star_root = os.path.join(repo_root, "InfinityStar")
for p in [repo_root, infinity_star_root]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(infinity_star_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import numpy as np
import torch
import cv2
from PIL import Image

from tools.run_infinity import (
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    gen_one_example,
    save_video,
    transform,
)
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import (
    get_dynamic_resolution_meta,
    get_first_full_spatial_size_scale_index,
)
from infinity.schedules import get_encode_decode_func


def main():
    checkpoints_dir = os.path.join(infinity_star_root, "")
    generation_duration = 5
    num_frames = generation_duration * 16 + 1

    args = __import__("infinity.utils.arg_util", fromlist=["Args"]).Args()
    args.pn = "0.40M"
    args.fps = 16
    args.video_frames = num_frames
    args.model_path = os.path.join(checkpoints_dir, "infinitystar_8b_480p_weights")
    args.checkpoint_type = "torch_shard"
    args.vae_path = os.path.join(checkpoints_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = os.path.join(checkpoints_dir, "text_encoder/flan-t5-xl-official/")
    args.videovae = 10
    args.model_type = "infinity_qwen8b"
    args.text_channels = 2048
    args.dynamic_scale_schedule = "infinity_star_interact"
    args.mask_type = "infinity_star_interact"
    args.bf16 = 1
    args.use_apg = 1
    args.use_cfg = 0
    args.cfg = 34
    args.tau_image = 1
    args.tau_video = 0.4
    args.apg_norm_threshold = 0.05
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 350
    args.semantic_scales = 11
    args.max_repeat_times = 10000
    args.frames_inner_clip = 20
    args.context_from_largest_no = 1
    args.context_interval = 2
    args.apply_spatial_patchify = 0
    args.use_flex_attn = 0
    args.rope2d_each_sa_layer = 1
    args.rope2d_normalized_by_hw = 2
    # VAR-Q：启用量化以验证修复效果
    args.enable_quantization = 1
    args.q_bits = 8
    args.quant_method = "G_SCALE_HEAD_DIM"
    args.qkv_format = "BHLc"
    args.rescale_qk = 0

    # 其他 InfinityStar 所需
    for attr, val in [
        ("noise_input", 0),
        ("use_cfg", 0),
        ("use_apg", 1),
        ("train_h_div_w_list", [0.571, 1.0]),
        ("scale_embeds_num", 128),
        ("context_frames", 1000),
        ("video_fps", 16),
        ("use_learnable_dim_proj", 0),
        ("semantic_scale_dim", 16),
        ("detail_scale_dim", 64),
        ("num_of_label_value", 2),
        ("vae_type", 64),
        ("other_args", None),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, val)
    if args.other_args is None:
        args.other_args = args

    print("[InfinityStar] Loading models (VAR-Q enabled)...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    vae = vae.float().to("cuda")
    infinity = load_transformer(vae, args)
    self_correction = SelfCorrection(vae, args)

    video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(
        args.dynamic_scale_schedule
    )

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(
        args.dynamic_scale_schedule, args.video_frames
    )
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"][
        (num_frames - 1) // 4 + 1
    ]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (
        len(scale_schedule) - args.tower_split_index
    )

    prompt = "A handsome smiling gardener inspecting plants, realistic cinematic lighting, detailed textures, ultra-realistic"
    image_path = os.path.join(infinity_star_root, "assets", "reference_image.webp")
    if not os.path.isfile(image_path):
        image_path = None
    if args.append_duration2caption:
        prompt = f"<<<t={generation_duration}s>>>" + prompt

    gt_leak, gt_ls_Bl = -1, None
    if image_path and os.path.isfile(image_path):
        tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
        ref_image = [cv2.imread(image_path)[:, :, ::-1]]
        ref_img_T3HW = [
            transform(Image.fromarray(f).convert("RGB"), tgt_h, tgt_w) for f in ref_image
        ]
        ref_img_bcthw = (
            torch.stack(ref_img_T3HW, 0).permute(1, 0, 2, 3).unsqueeze(0).to("cuda")
        )
        _, _, gt_ls_Bl, _, _, _ = video_encode(
            vae, ref_img_bcthw, vae_features=None, self_correction=self_correction,
            args=args, infer_mode=True, dynamic_resolution_h_w=dynamic_resolution_h_w,
        )
        gt_leak = 14
        print("[InfinityStar] Image-to-Video with reference image")
    else:
        print("[InfinityStar] Text-to-Video (no reference image)")

    print("[InfinityStar] Generating...")
    st = time.time()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        generated_image, _ = gen_one_example(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
            negative_prompt="",
            g_seed=42,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            cfg_list=args.cfg,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[0],
            vae_type=args.vae_type,
            sampling_per_bits=1,
            enable_positive_prompt=0,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=get_visual_rope_embeds,
            context_info=context_info,
            noise_list=None,
        )
    elapsed = time.time() - st
    if generated_image.dim() == 3:
        generated_image = generated_image.unsqueeze(0)
    out_np = generated_image.cpu().numpy()
    if out_np.ndim == 5:
        out_np = out_np[0]
    print(f"[InfinityStar] Done in {elapsed:.2f}s, shape {out_np.shape}")

    out_dir = os.path.join(repo_root, "scripts", "output")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "infinitystar_varq_demo.mp4")
    save_video(out_np, fps=args.fps, save_filepath=save_path)
    print(f"Video saved: {save_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
"""
Generate VBench videos with InfinityStar + VARQ (supports multi-GPU sharding via start/max args).
"""

import argparse
import json
import os
import os.path as osp
import random
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(__file__)))

from infinity.utils.arg_util import Args
from tools.infer_video_720p import InferencePipe, load_varq_quant_config, perform_inference
from tools.run_infinity import save_video


def load_prompts(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)


def _sanitize_stem(s: str, max_stem_bytes: int) -> str:
    """Sanitize prompt for use as filename stem; truncate by UTF-8 bytes (Linux NAME_MAX ~255 per component)."""
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = "empty_prompt"
    while len(s.encode("utf-8")) > max_stem_bytes:
        s = s[:-1]
    return s.rstrip()


def prompt_video_basename(prompt_en: str, sample_idx: int) -> str:
    """Full basename: <prompt_en sanitized, byte-truncated>-<sample_idx>.mp4 (always original prompt_en, not refined)."""
    suffix = f"-{sample_idx}.mp4"
    max_stem = 255 - len(suffix.encode("utf-8"))
    stem = _sanitize_stem(prompt_en, max_stem_bytes=max_stem)
    return f"{stem}{suffix}"


def build_infer_args(checkpoints_dir: str, quant_config: dict) -> Args:
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

    args.enable_quantization = int(quant_config.get("enable", False))
    args.q_bits = quant_config.get("q_bits", 8)
    args.quant_method = quant_config.get("quant_method", "G_SCALE_HEAD_DIM")
    args.qkv_format = quant_config.get("qkv_format", "BHLc")
    args.rescale_qk = int(quant_config.get("rescale_qk", False))
    args.enable_sageattn = int(quant_config.get("enable_sageattn", False))
    args.sageattn_type = str(quant_config.get("sageattn_type", "sageattn"))
    return args


def main():
    parser = argparse.ArgumentParser(description="VBench generation for InfinityStar+VARQ")
    parser.add_argument("--prompts_json", type=str, default="evaluation/VBench_rewrited_prompt.json")
    parser.add_argument("--output_dir", type=str, default="Benchmark/outputs/vbench_eval_varq")
    parser.add_argument("--checkpoints_dir", type=str, default="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar")
    parser.add_argument("--config_file", type=str, default=osp.join(osp.dirname(osp.dirname(__file__)), "VAR_Q", "Infinity-VAR_Q-8.json"))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt in this run")
    parser.add_argument("--sample_start_idx", type=int, default=0, help="Starting sample index in output filename, e.g. 1 -> xx-01.mp4")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--use_random_seed", action="store_true")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--filename_mode",
        type=str,
        default="id",
        choices=["id", "prompt"],
        help="id: 000-00.mp4, prompt: <original prompt_en>-<idx>.mp4 (sanitized; stem truncated by UTF-8 bytes to fit ext4)",
    )
    parser.add_argument(
        "--inference_prompt_source",
        type=str,
        default="refined",
        choices=["refined", "prompt_en"],
        help="Which field feeds the model: refined_prompt (default) or original prompt_en (matches VBench filename semantics).",
    )
    args_cmd = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        if args_cmd.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cmd.gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.set_device(0)

    prompts = load_prompts(args_cmd.prompts_json)
    if args_cmd.max_videos is not None:
        prompts = prompts[args_cmd.start_idx: args_cmd.start_idx + args_cmd.max_videos]
    else:
        prompts = prompts[args_cmd.start_idx:]

    quant_config = load_varq_quant_config(args_cmd.config_file)
    args = build_infer_args(args_cmd.checkpoints_dir, quant_config)

    output_dir = Path(args_cmd.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] prompts: {len(prompts)}, samples/prompt: {args_cmd.num_samples}, sample_start_idx: {args_cmd.sample_start_idx}")
    print(f"[Info] filename mode: {args_cmd.filename_mode}")
    print(f"[Info] output dir: {output_dir}")
    pipe = InferencePipe(args)

    results = []
    total = len(prompts) * args_cmd.num_samples
    pbar = tqdm(total=total, desc="Generating", unit="video")

    for local_idx, item in enumerate(prompts):
        prompt_idx = item.get("original_idx", args_cmd.start_idx + local_idx)
        refined_prompt = item.get("refined_prompt", item.get("prompt_en", ""))
        prompt_en = item.get("prompt_en", refined_prompt)
        dimensions = item.get("dimension", [])
        infer_prompt = prompt_en if args_cmd.inference_prompt_source == "prompt_en" else refined_prompt

        for sample_offset in range(args_cmd.num_samples):
            sample_idx = args_cmd.sample_start_idx + sample_offset
            if args_cmd.use_random_seed:
                seed = random.randint(0, 2**31 - 1)
            else:
                seed = args_cmd.base_seed + prompt_idx * 1000 + sample_idx

            data = {"seed": seed, "image_path": None, "prompt": infer_prompt}
            try:
                output = perform_inference(pipe, data, args)
                if args_cmd.filename_mode == "prompt":
                    fname = prompt_video_basename(prompt_en, sample_idx)
                else:
                    fname = f"{prompt_idx:03d}-{sample_idx:02d}.mp4"
                vpath = output_dir / fname
                save_video(output["output"], fps=args.fps, save_filepath=str(vpath))
                results.append(
                    {
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "prompt_en": prompt_en,
                        "refined_prompt": refined_prompt,
                        "dimension": dimensions,
                        "video_filename": fname,
                        "video_path": str(vpath),
                        "seed": seed,
                        "elapsed_time": output.get("elapsed_time", None),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "prompt_en": prompt_en,
                        "refined_prompt": refined_prompt,
                        "dimension": dimensions,
                        "video_path": None,
                        "seed": seed,
                        "error": str(e),
                    }
                )
            pbar.update(1)

    pbar.close()
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    ok = len([r for r in results if r.get("video_path")])
    print(f"[Done] generated {ok}/{total} videos")
    print(f"[Done] metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

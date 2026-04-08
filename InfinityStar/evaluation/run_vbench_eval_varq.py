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
import traceback
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


def _resolve_saved_path(requested_path: Path) -> Path:
    """
    Resolve actual file path written by save_video.
    save_video may switch .mp4 to .jpg for single-frame outputs.
    """
    if requested_path.exists():
        return requested_path
    if requested_path.suffix.lower() == ".mp4":
        jpg_path = requested_path.with_suffix(".jpg")
        if jpg_path.exists():
            return jpg_path
    return requested_path


def _save_and_verify_video(output_frames, fps: int, requested_path: Path) -> Path:
    save_video(output_frames, fps=fps, save_filepath=str(requested_path))
    actual_path = _resolve_saved_path(requested_path)
    if not actual_path.exists():
        raise FileNotFoundError(
            f"save_video returned but file not found. requested={requested_path}, resolved={actual_path}"
        )
    file_size = actual_path.stat().st_size
    if file_size <= 0:
        raise RuntimeError(f"saved file is empty: {actual_path}")
    print(
        f"[Save][OK] path={actual_path} size_bytes={file_size}",
        flush=True,
    )
    return actual_path


def _load_existing_metadata(metadata_path: Path) -> dict[tuple[int, int], dict]:
    if not metadata_path.is_file():
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return {}
    if not isinstance(items, list):
        return {}
    out = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        prompt_idx = item.get("prompt_idx")
        sample_idx = item.get("sample_idx")
        if isinstance(prompt_idx, int) and isinstance(sample_idx, int):
            out[(prompt_idx, sample_idx)] = item
    return out


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples whose output file already exists and rebuild metadata incrementally.",
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
    write_test = output_dir / ".write_test.tmp"
    try:
        with open(write_test, "w", encoding="utf-8") as f:
            f.write("ok")
        write_test.unlink()
    except Exception as e:
        raise RuntimeError(f"output_dir is not writable: {output_dir}. err={e}") from e

    print(f"[Info] prompts: {len(prompts)}, samples/prompt: {args_cmd.num_samples}, sample_start_idx: {args_cmd.sample_start_idx}")
    print(f"[Info] filename mode: {args_cmd.filename_mode}")
    print(f"[Info] output dir: {output_dir}")
    print(f"[Info] resume mode: {args_cmd.resume}")
    print(f"[Info] output dir writable check passed: {output_dir}", flush=True)
    pipe = InferencePipe(args)

    metadata_path = output_dir / "metadata.json"
    existing_results = _load_existing_metadata(metadata_path)
    results_by_key = dict(existing_results)
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
            if args_cmd.filename_mode == "prompt":
                fname = prompt_video_basename(prompt_en, sample_idx)
            else:
                fname = f"{prompt_idx:03d}-{sample_idx:02d}.mp4"
            vpath = output_dir / fname
            resolved_existing = _resolve_saved_path(vpath)
            key = (prompt_idx, sample_idx)

            if args_cmd.resume and resolved_existing.exists():
                prev = results_by_key.get(key, {})
                results_by_key[key] = {
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt_en": prompt_en,
                    "refined_prompt": refined_prompt,
                    "dimension": dimensions,
                    "video_filename": resolved_existing.name,
                    "video_path": str(resolved_existing),
                    "seed": prev.get("seed"),
                    "elapsed_time": prev.get("elapsed_time"),
                    "status": "skipped_existing",
                }
                print(
                    f"[Resume][Skip] prompt_idx={prompt_idx} sample_idx={sample_idx} existing={resolved_existing}",
                    flush=True,
                )
                pbar.update(1)
                continue

            if args_cmd.use_random_seed:
                seed = random.randint(0, 2**31 - 1)
            else:
                seed = args_cmd.base_seed + prompt_idx * 1000 + sample_idx

            data = {"seed": seed, "image_path": None, "prompt": infer_prompt}
            try:
                output = perform_inference(pipe, data, args)
                print(
                    f"[Save][Start] prompt_idx={prompt_idx} sample_idx={sample_idx} target={vpath}",
                    flush=True,
                )
                actual_vpath = _save_and_verify_video(output["output"], fps=args.fps, requested_path=vpath)
                results_by_key[key] = {
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt_en": prompt_en,
                    "refined_prompt": refined_prompt,
                    "dimension": dimensions,
                    "video_filename": actual_vpath.name,
                    "video_path": str(actual_vpath),
                    "seed": seed,
                    "elapsed_time": output.get("elapsed_time", None),
                    "status": "ok",
                }
            except Exception as e:
                print(
                    f"[Save][ERROR] prompt_idx={prompt_idx} sample_idx={sample_idx} err={type(e).__name__}: {e}",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)
                results_by_key[key] = {
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt_en": prompt_en,
                    "refined_prompt": refined_prompt,
                    "dimension": dimensions,
                    "video_path": None,
                    "seed": seed,
                    "error": str(e),
                    "status": "failed",
                }
            pbar.update(1)

    pbar.close()
    results = [results_by_key[k] for k in sorted(results_by_key.keys())]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    ok = len([r for r in results if r.get("video_path")])
    print(f"[Done] generated {ok}/{total} videos")
    print(f"[Done] metadata: {metadata_path}")


if __name__ == "__main__":
    main()

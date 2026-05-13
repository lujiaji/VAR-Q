#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import cv2
import torch
from torch.cuda.amp import autocast

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from VAR_Q.hooks import install_varq_hooks
from VAR_Q.infinity_config import apply_infinity_config
from VAR_Q.paths import prepend_sys_path, require_third_party_repo
from VAR_Q.profiling import (
    collect_varq_memory_breakdown,
    format_memory_breakdown,
    reset_cuda_memory_stats,
)


def _load_infinity_runtime():
    infinity_root = str(require_third_party_repo("Infinity", "https://github.com/FoundationVision/Infinity"))
    prepend_sys_path([REPO_ROOT, infinity_root])
    from tools.run_infinity import (
        _import_dynamic_resolution,
        add_common_arguments,
        gen_one_img,
        load_tokenizer,
        load_transformer,
        load_visual_tokenizer,
    )

    return (
        infinity_root,
        add_common_arguments,
        gen_one_img,
        load_tokenizer,
        load_transformer,
        load_visual_tokenizer,
        _import_dynamic_resolution,
    )


def _build_parser(add_common_arguments) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infinity inference with VAR-Q runtime hooks")
    add_common_arguments(parser)
    parser.add_argument(
        "--varq_config",
        "--config",
        dest="varq_config",
        default=osp.join(REPO_ROOT, "configs/infinity/varq/base/Infinity-VARQ-8.json"),
        help="VAR-Q JSON config.",
    )
    parser.add_argument("--prompt", type=str, default="a dog")
    parser.add_argument("--batch_size", type=int, default=1, help="Repeat the prompt this many times for batched smoke inference")
    parser.add_argument("--save_file", type=str, default=osp.join(REPO_ROOT, "scripts/output/infinity.png"))
    parser.add_argument("--profile_memory", action="store_true", help="Print VAR-Q cache and CUDA allocator memory stats")
    return parser


def main() -> None:
    (
        _infinity_root,
        add_common_arguments,
        gen_one_img,
        load_tokenizer,
        load_transformer,
        load_visual_tokenizer,
        import_dynamic_resolution,
    ) = _load_infinity_runtime()

    parser = _build_parser(add_common_arguments)
    args = parser.parse_args()
    quant_config, ablation_config = apply_infinity_config(args)

    args.cfg = list(map(float, str(args.cfg).split(",")))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    install_varq_hooks(infinity, "infinity", quant_config, ablation_config=ablation_config)
    if args.profile_memory:
        reset_cuda_memory_stats()

    dynamic_resolution_h_w, _h_div_w_templates = import_dynamic_resolution()
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    with autocast(dtype=torch.bfloat16), torch.no_grad():
        generated_image = gen_one_img(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            [args.prompt] * int(args.batch_size),
            g_seed=args.seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=args.cfg,
            tau_list=args.tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=args.enable_positive_prompt,
    )
    if args.profile_memory:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stats = collect_varq_memory_breakdown(infinity)
        print("[VAR-Q memory] " + format_memory_breakdown(stats))

    save_file = osp.abspath(args.save_file)
    os.makedirs(osp.dirname(save_file), exist_ok=True)
    if isinstance(generated_image, torch.Tensor) and generated_image.ndim == 4:
        stem, ext = osp.splitext(save_file)
        for idx, image in enumerate(generated_image):
            cv2.imwrite(f"{stem}_{idx:03d}{ext}", image.cpu().numpy())
        print(f"Saved {generated_image.shape[0]} images to {stem}_*.{ext.lstrip('.')}")
    elif isinstance(generated_image, list):
        stem, ext = osp.splitext(save_file)
        for idx, image in enumerate(generated_image):
            cv2.imwrite(f"{stem}_{idx:03d}{ext}", image.cpu().numpy())
        print(f"Saved {len(generated_image)} images to {stem}_*.{ext.lstrip('.')}")
    else:
        cv2.imwrite(save_file, generated_image.cpu().numpy())
        print(f"Saved to {save_file}")


if __name__ == "__main__":
    main()

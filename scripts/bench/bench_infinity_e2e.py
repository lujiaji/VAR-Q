#!/usr/bin/env python3
"""E2E single-image Infinity benchmark for fp16 and VAR-Q KV cache modes."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import torch
from torch.cuda.amp import autocast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VAR_Q.hooks import install_varq_hooks  # noqa: E402
from VAR_Q.hooks.runtime import remove_varq_hooks  # noqa: E402
from VAR_Q.infinity_config import apply_infinity_config  # noqa: E402
from VAR_Q.paths import prepend_sys_path, require_third_party_repo  # noqa: E402


@dataclass(frozen=True)
class BenchCase:
    name: str
    config_path: Path | None
    fused: bool = False
    fused_backend: str = "triton"


def _load_infinity_runtime():
    infinity_root = require_third_party_repo(
        "Infinity", "https://github.com/FoundationVision/Infinity"
    )
    prepend_sys_path([REPO_ROOT, infinity_root])
    from tools import run_infinity  # type: ignore

    import_dynamic_resolution = getattr(run_infinity, "_import_dynamic_resolution", None)
    if import_dynamic_resolution is None:
        def import_dynamic_resolution():
            from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

            return dynamic_resolution_h_w, h_div_w_templates

    return (
        infinity_root,
        run_infinity.add_common_arguments,
        run_infinity.gen_one_img,
        run_infinity.load_tokenizer,
        run_infinity.load_transformer,
        run_infinity.load_visual_tokenizer,
        import_dynamic_resolution,
    )


def _config_for_bits(bits: int) -> Path:
    return REPO_ROOT / "configs" / "infinity" / "varq" / "base" / f"Infinity-VARQ-{bits}.json"


def _build_parser(add_common_arguments) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--prompt", type=str, default="a high-resolution photograph of a corgi wearing sunglasses")
    parser.add_argument("--iters", type=int, default=10, help="timed image generations per case")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup generations per case")
    parser.add_argument(
        "--cases",
        default="fp16,varq8,varq8_fused,varq4,varq2",
        help="comma-separated cases: fp16,varq8,varq8_fused,varq4,varq2",
    )
    parser.add_argument("--config-8", default=str(_config_for_bits(8)))
    parser.add_argument("--config-4", default=str(_config_for_bits(4)))
    parser.add_argument("--config-2", default=str(_config_for_bits(2)))
    parser.add_argument("--fused-backend", default="cuda-direct", choices=("triton", "cuda", "cuda-direct"))
    parser.add_argument("--profile", default="", help="if set to a case name, torch.profiler one generation of that case and print top CUDA ops by self time")
    parser.add_argument("--save-dir", default="", help="optional directory for first image from each case")
    parser.add_argument("--empty-cache-between", action="store_true")
    parser.add_argument("--json-out", default="", help="optional path to write JSON results")
    return parser


def _normalize_cfg(value: Any) -> Any:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        values = [float(part) for part in parts]
        return values[0] if len(values) == 1 else values
    return value


def _prepare_case_args(
    base_args: argparse.Namespace,
    config_path: Path,
    *,
    enable_quant: bool,
) -> tuple[argparse.Namespace, dict[str, Any], dict[str, Any]]:
    args = copy.deepcopy(base_args)
    args.varq_config = str(config_path)
    quant_config, ablation_config = apply_infinity_config(args)
    args.cfg = _normalize_cfg(args.cfg)
    if not enable_quant:
        quant_config, ablation_config = {}, {}
    return args, quant_config, ablation_config


def _scale_schedule(args: argparse.Namespace, import_dynamic_resolution) -> list[tuple[int, int, int]]:
    dynamic_resolution_h_w, _h_div_w_templates = import_dynamic_resolution()
    raw_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    return [(1, h, w) for (_, h, w) in raw_schedule]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _maybe_save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor) and image.ndim == 4:
        image = image[0]
    if isinstance(image, list):
        image = image[0]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    cv2.imwrite(str(path), image)


def _run_generation(
    gen_one_img,
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    args: argparse.Namespace,
    scale_schedule: list[tuple[int, int, int]],
) -> Any:
    return gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        args.prompt,
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


def _case_from_name(name: str, args: argparse.Namespace) -> BenchCase:
    normalized = name.strip().lower()
    if normalized == "fp16":
        return BenchCase("fp16", None)
    if normalized == "varq8":
        return BenchCase("varq8", Path(args.config_8))
    if normalized == "varq8_fused":
        return BenchCase("varq8_fused", Path(args.config_8), fused=True, fused_backend=args.fused_backend)
    if normalized == "varq4":
        return BenchCase("varq4", Path(args.config_4))
    if normalized == "varq2":
        return BenchCase("varq2", Path(args.config_2))
    raise ValueError(f"unknown benchmark case: {name}")


def _iter_cases(args: argparse.Namespace) -> Iterable[BenchCase]:
    for raw in args.cases.split(","):
        if raw.strip():
            yield _case_from_name(raw, args)


def _remove_existing_hooks(infinity) -> None:
    handle = getattr(infinity, "_varq_hook_handle", None)
    if handle is not None:
        remove_varq_hooks(handle)


def _install_case_hooks(infinity, case: BenchCase, quant_config: dict[str, Any], ablation_config: dict[str, Any]) -> dict[str, Any]:
    _remove_existing_hooks(infinity)
    if case.config_path is None:
        return {}
    cfg = dict(quant_config)
    if case.fused:
        cfg["enable_fused_kv_flashattn"] = True
        cfg["fused_kv_backend"] = case.fused_backend
    handle = install_varq_hooks(infinity, "infinity", cfg, ablation_config=ablation_config)
    return {
        "hit_count": int(handle.hit_count),
        "bits": int(cfg.get("q_bits", 0) or 0),
        "fused": bool(cfg.get("enable_fused_kv_flashattn", False)),
        "fused_backend": str(cfg.get("fused_kv_backend", "")),
    }


def _summarize(times_s: list[float]) -> dict[str, Any]:
    return {
        "times_s": times_s,
        "mean_s": statistics.fmean(times_s),
        "median_s": statistics.median(times_s),
        "min_s": min(times_s),
        "max_s": max(times_s),
        "stdev_s": statistics.stdev(times_s) if len(times_s) > 1 else 0.0,
    }


def main() -> int:
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
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    load_config = next((case.config_path for case in _iter_cases(args) if case.config_path is not None), Path(args.config_8))
    load_args, _load_quant_config, _load_ablation_config = _prepare_case_args(
        args,
        load_config,
        enable_quant=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Infinity e2e benchmark requires CUDA")

    print(json.dumps({"event": "load_start", "config": str(load_config)}, ensure_ascii=False), flush=True)
    text_tokenizer, text_encoder = load_tokenizer(t5_path=load_args.text_encoder_ckpt)
    vae = load_visual_tokenizer(load_args)
    infinity = load_transformer(vae, load_args)
    scale_schedule = _scale_schedule(load_args, import_dynamic_resolution)
    print(json.dumps({"event": "load_done", "device": torch.cuda.get_device_name(device)}, ensure_ascii=False), flush=True)

    results: list[dict[str, Any]] = []
    save_dir = Path(args.save_dir) if args.save_dir else None

    with autocast(dtype=torch.bfloat16), torch.no_grad():
        for case in _iter_cases(args):
            case_config = case.config_path if case.config_path is not None else load_config
            case_args, quant_config, ablation_config = _prepare_case_args(
                args,
                case_config,
                enable_quant=case.config_path is not None,
            )
            hook_info = _install_case_hooks(infinity, case, quant_config, ablation_config)
            if case.fused and case.fused_backend in ("cuda", "cuda-direct"):
                from VAR_Q.fused import flash_dequant_cuda

                status = flash_dequant_cuda.availability()
                print(
                    json.dumps(
                        {
                            "event": "fused_backend_status",
                            "case": case.name,
                            "available": status.available,
                            "message": status.message,
                            "module_path": status.module_path,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if not status.available:
                    raise RuntimeError(f"fused backend unavailable for {case.name}: {status.message}")

            print(json.dumps({"event": "case_start", "case": case.name, **hook_info}, ensure_ascii=False), flush=True)
            for _ in range(args.warmup):
                image = _run_generation(
                    gen_one_img, infinity, vae, text_tokenizer, text_encoder, case_args, scale_schedule
                )
                del image
                _sync(device)
                if args.empty_cache_between:
                    torch.cuda.empty_cache()

            if args.profile and args.profile == case.name:
                from torch.profiler import profile, ProfilerActivity

                _sync(device)
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_stack=True,
                ) as prof:
                    image = _run_generation(
                        gen_one_img, infinity, vae, text_tokenizer, text_encoder, case_args, scale_schedule
                    )
                    _sync(device)
                del image
                print(f"==== PROFILE case={case.name} top CUDA ops by self time ====", flush=True)
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30), flush=True)
                print(f"==== PROFILE case={case.name} by call-stack (attribute copies/ops to source) ====", flush=True)
                print(
                    prof.key_averages(group_by_stack_n=6).table(
                        sort_by="self_cuda_time_total", row_limit=30
                    ),
                    flush=True,
                )

            times_s: list[float] = []
            first_image = None
            for iter_idx in range(args.iters):
                _sync(device)
                start = time.perf_counter()
                image = _run_generation(
                    gen_one_img, infinity, vae, text_tokenizer, text_encoder, case_args, scale_schedule
                )
                _sync(device)
                elapsed = time.perf_counter() - start
                times_s.append(elapsed)
                if iter_idx == 0 and save_dir is not None:
                    first_image = image
                else:
                    del image
                if args.empty_cache_between:
                    torch.cuda.empty_cache()
                print(
                    json.dumps(
                        {"event": "iter", "case": case.name, "iter": iter_idx, "seconds": elapsed},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            if first_image is not None and save_dir is not None:
                _maybe_save_image(first_image, save_dir / f"{case.name}.png")
                del first_image

            summary = {"case": case.name, **hook_info, **_summarize(times_s)}
            results.append(summary)
            print(json.dumps({"event": "case_done", **summary}, ensure_ascii=False), flush=True)

    _remove_existing_hooks(infinity)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"event": "summary", "results": results}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

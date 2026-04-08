#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import os.path as osp
import statistics
import sys
import traceback
from typing import Any, Dict, List, Optional

import torch


THIS_DIR = osp.dirname(osp.abspath(__file__))
VARQ_ROOT = osp.dirname(THIS_DIR)
WORKSPACE_ROOT = osp.dirname(VARQ_ROOT)
INFINITYSTAR_ROOT = osp.join(VARQ_ROOT, "InfinityStar")

for p in (VARQ_ROOT, WORKSPACE_ROOT, INFINITYSTAR_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.measure_throughput_varq import (  # noqa: E402
    InferencePipe,
    build_args,
    run_once,
)
from tools.infer_video_720p import (  # noqa: E402
    load_varq_quant_config,
    perform_inference,
)
from tools.run_infinity import save_video  # noqa: E402
from RuntimeProfiler.utils.infinitystar_profile import (  # noqa: E402
    cuda_memory_stats_mib,
    get_activation_peak_mb,
    reset_peak_and_get_baseline_alloc_mb,
)


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _run_single_bs_throughput(
    pipe: InferencePipe,
    args: Any,
    prompt: str,
    seed: int,
    warmup: int,
    runs: int,
) -> Dict[str, Any]:
    warmup_times: List[float] = []
    run_times: List[float] = []

    for i in range(max(warmup, 0)):
        t = run_once(pipe, args, prompt, seed + i)
        warmup_times.append(t)

    for i in range(max(runs, 1)):
        t = run_once(pipe, args, prompt, seed + 10_000 + i)
        run_times.append(t)

    avg_s = statistics.mean(run_times)
    p50_s = statistics.median(run_times)
    fps_avg = float(args.video_frames) / avg_s
    fps_p50 = float(args.video_frames) / p50_s

    return {
        "warmup_times_s": warmup_times,
        "run_times_s": run_times,
        "avg_time_s": avg_s,
        "p50_time_s": p50_s,
        "min_time_s": min(run_times),
        "max_time_s": max(run_times),
        "video_frames": int(args.video_frames),
        "throughput_fps_avg": fps_avg,
        "throughput_fps_p50": fps_p50,
    }


def _run_single_bs_memory(
    pipe: InferencePipe,
    args: Any,
    prompt: str,
    seed: int,
    save_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    data = {"seed": int(seed), "image_path": None, "prompt": prompt}

    baseline_alloc_mb = reset_peak_and_get_baseline_alloc_mb()
    out = perform_inference(pipe, data, args)
    if save_video_path:
        os.makedirs(osp.dirname(save_video_path), exist_ok=True)
        save_video(out["output"], fps=args.fps, save_filepath=save_video_path)
    activation_peak_mb = get_activation_peak_mb(baseline_alloc_mb)
    cuda_stats = cuda_memory_stats_mib()

    mem_trace = list(getattr(pipe.infinity, "runtime_mem_trace", []) or [])
    kv_live_mb = _safe_float(getattr(pipe.infinity, "last_kv_cache_mb", 0.0), 0.0) or 0.0
    kv_peak_from_trace_mb = max(
        (_safe_float(p.get("kv_cache_mb"), 0.0) or 0.0 for p in mem_trace),
        default=0.0,
    )
    alloc_peak_from_trace_mb = max(
        (_safe_float(p.get("alloc_mem_mb"), 0.0) or 0.0 for p in mem_trace),
        default=0.0,
    )

    kv_mem_single_bs_mb = max(kv_live_mb, kv_peak_from_trace_mb)
    total_mem_single_bs_mb = max(
        alloc_peak_from_trace_mb,
        _safe_float(cuda_stats.get("max_alloc_mem_mb"), 0.0) or 0.0,
    )

    return {
        "elapsed_time_s": _safe_float(out.get("elapsed_time")),
        "weights_baseline_mb": _safe_float(out.get("weights_baseline_mb")),
        "activation_peak_mb": _safe_float(activation_peak_mb),
        "kv_mem_single_bs_mb": kv_mem_single_bs_mb,
        "total_mem_single_bs_mb": total_mem_single_bs_mb,
        "runtime_mem_trace_len": len(mem_trace),
        "cuda_memory_stats_mib": cuda_stats,
        "saved_video_path": osp.abspath(save_video_path) if save_video_path else None,
    }


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _cleanup_cuda_oom(pipe: InferencePipe) -> None:
    try:
        pipe.infinity.last_kv_cache_bytes = 0
        pipe.infinity.last_kv_cache_mb = 0.0
        if hasattr(pipe.infinity, "_reset_runtime_mem_trace"):
            pipe.infinity._reset_runtime_mem_trace()
        if hasattr(pipe.infinity, "_reset_latency_trace"):
            pipe.infinity._reset_latency_trace()
        for block in getattr(pipe.infinity, "unregistered_blocks", []) or []:
            try:
                block.attn.kv_caching(False)
            except Exception:
                pass
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def _run_batch_inference(
    pipe: InferencePipe,
    args: Any,
    prompt: str,
    seed: int,
    batch_size: int,
    save_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    data = {
        "seed": int(seed),
        "image_path": None,
        "prompt": prompt,
        "batch_size": int(batch_size),
    }

    baseline_alloc_mb = reset_peak_and_get_baseline_alloc_mb()
    out = perform_inference(pipe, data, args)
    if save_video_path:
        os.makedirs(osp.dirname(save_video_path), exist_ok=True)
        video = out["output"]
        if len(video.shape) == 5:
            video = video[0]
        save_video(video, fps=args.fps, save_filepath=save_video_path)
    activation_peak_mb = get_activation_peak_mb(baseline_alloc_mb)
    cuda_stats = cuda_memory_stats_mib()

    mem_trace = list(getattr(pipe.infinity, "runtime_mem_trace", []) or [])
    kv_live_mb = _safe_float(getattr(pipe.infinity, "last_kv_cache_mb", 0.0), 0.0) or 0.0
    kv_peak_from_trace_mb = max(
        (_safe_float(p.get("kv_cache_mb"), 0.0) or 0.0 for p in mem_trace),
        default=0.0,
    )
    alloc_peak_from_trace_mb = max(
        (_safe_float(p.get("alloc_mem_mb"), 0.0) or 0.0 for p in mem_trace),
        default=0.0,
    )

    kv_mem_mb = max(kv_live_mb, kv_peak_from_trace_mb)
    total_mem_mb = max(
        alloc_peak_from_trace_mb,
        _safe_float(cuda_stats.get("max_alloc_mem_mb"), 0.0) or 0.0,
    )
    elapsed_s = _safe_float(out.get("elapsed_time"))
    throughput_fps = None
    if elapsed_s and elapsed_s > 0:
        throughput_fps = float(args.video_frames) * float(batch_size) / float(elapsed_s)

    return {
        "batch_size": int(batch_size),
        "elapsed_time_s": elapsed_s,
        "throughput_fps": throughput_fps,
        "weights_baseline_mb": _safe_float(out.get("weights_baseline_mb")),
        "activation_peak_mb": _safe_float(activation_peak_mb),
        "kv_mem_mb": kv_mem_mb,
        "total_mem_mb": total_mem_mb,
        "runtime_mem_trace_len": len(mem_trace),
        "cuda_memory_stats_mib": cuda_stats,
        "saved_video_path": osp.abspath(save_video_path) if save_video_path else None,
    }


def _probe_max_bs(
    pipe: InferencePipe,
    args: Any,
    prompt: str,
    seed: int,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    low = 1
    high = 1
    best: Optional[Dict[str, Any]] = None

    while True:
        try:
            stats = _run_batch_inference(pipe, args, prompt, seed + high * 100, high)
            attempts.append({"batch_size": high, "status": "ok", **stats})
            best = stats
            low = high
            high *= 2
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            attempts.append(
                {
                    "batch_size": high,
                    "status": "oom",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            _cleanup_cuda_oom(pipe)
            break

    if best is None:
        raise RuntimeError("Batch size 1 already OOM during max_bs probing.")

    left = low + 1
    right = high - 1
    while left <= right:
        mid = (left + right) // 2
        try:
            stats = _run_batch_inference(pipe, args, prompt, seed + mid * 100, mid)
            attempts.append({"batch_size": mid, "status": "ok", **stats})
            best = stats
            left = mid + 1
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            attempts.append(
                {
                    "batch_size": mid,
                    "status": "oom",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            _cleanup_cuda_oom(pipe)
            right = mid - 1

    return {
        "max_bs": int(best["batch_size"]),
        "throughput_at_max_bs_fps": best["throughput_fps"],
        "kv_mem_max_bs_mb": best["kv_mem_mb"],
        "total_mem_max_bs_mb": best["total_mem_mb"],
        "elapsed_time_at_max_bs_s": best["elapsed_time_s"],
        "attempts": attempts,
    }


def _summarize_latency_trace(latency_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not latency_trace:
        return {
            "has_trace": False,
            "num_steps": 0,
            "total_time_s": None,
            "attn_time_s": None,
            "ffn_time_s": None,
            "others_time_s": None,
            "scales": [],
        }

    total_time_s = 0.0
    attn_time_s = 0.0
    ffn_time_s = 0.0
    others_time_s = 0.0
    scales = set()

    for step in latency_trace:
        total_time_s += _safe_float(step.get("total_time_s"), 0.0) or 0.0
        attn_time_s += _safe_float(step.get("attn_time_s"), 0.0) or 0.0
        ffn_time_s += _safe_float(step.get("ffn_time_s"), 0.0) or 0.0
        others_time_s += _safe_float(step.get("others_time_s"), 0.0) or 0.0
        si = step.get("scale_ind")
        if isinstance(si, int):
            scales.add(si)

    return {
        "has_trace": True,
        "num_steps": len(latency_trace),
        "total_time_s": total_time_s,
        "attn_time_s": attn_time_s,
        "ffn_time_s": ffn_time_s,
        "others_time_s": others_time_s,
        "scales": sorted(scales),
    }


def run_benchmark(args_cmd: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args_cmd.output_dir, exist_ok=True)

    quant_config = load_varq_quant_config(args_cmd.config_file)
    args = build_args(args_cmd.checkpoints_dir, quant_config)
    pipe = InferencePipe(args)

    save_video_path = None
    if args_cmd.save_video and not args_cmd.only_max_bs:
        if args_cmd.video_output_dir:
            video_output_dir = args_cmd.video_output_dir
        else:
            video_output_dir = osp.join(args_cmd.output_dir, "videos")
        video_filename = args_cmd.video_filename or f"single_bs_seed{args_cmd.seed}.mp4"
        save_video_path = osp.join(video_output_dir, video_filename)

    throughput: Dict[str, Any] = {
        "warmup_times_s": [],
        "run_times_s": [],
        "avg_time_s": None,
        "p50_time_s": None,
        "min_time_s": None,
        "max_time_s": None,
        "video_frames": int(args.video_frames),
        "throughput_fps_avg": None,
        "throughput_fps_p50": None,
        "status": "skipped" if args_cmd.only_max_bs else "ok",
        "message": "Skipped because --only_max_bs was enabled." if args_cmd.only_max_bs else "",
    }
    single_mem: Dict[str, Any] = {
        "elapsed_time_s": None,
        "weights_baseline_mb": None,
        "activation_peak_mb": None,
        "kv_mem_single_bs_mb": None,
        "total_mem_single_bs_mb": None,
        "runtime_mem_trace_len": 0,
        "cuda_memory_stats_mib": None,
        "saved_video_path": None,
        "status": "skipped" if args_cmd.only_max_bs else "ok",
        "message": "Skipped because --only_max_bs was enabled." if args_cmd.only_max_bs else "",
    }
    latency_summary: Dict[str, Any] = {
        "has_trace": False,
        "num_steps": 0,
        "total_time_s": None,
        "attn_time_s": None,
        "ffn_time_s": None,
        "others_time_s": None,
        "scales": [],
        "status": "skipped" if args_cmd.only_max_bs else "ok",
        "message": "Skipped because --only_max_bs was enabled." if args_cmd.only_max_bs else "",
    }
    latency_trace_file = None

    if not args_cmd.only_max_bs:
        throughput = _run_single_bs_throughput(
            pipe=pipe,
            args=args,
            prompt=args_cmd.prompt,
            seed=args_cmd.seed,
            warmup=args_cmd.warmup,
            runs=args_cmd.runs,
        )
        single_mem = _run_single_bs_memory(
            pipe=pipe,
            args=args,
            prompt=args_cmd.prompt,
            seed=args_cmd.seed + 1_000_000,
            save_video_path=save_video_path,
        )
        latency_trace = list(getattr(pipe.infinity, "runtime_latency_trace", []) or [])
        latency_summary = _summarize_latency_trace(latency_trace)
        latency_trace_file = osp.join(args_cmd.output_dir, "latency_trace_single_bs.json")
        with open(latency_trace_file, "w", encoding="utf-8") as f:
            json.dump(latency_trace, f, indent=2, ensure_ascii=False)

    max_bs_result: Dict[str, Any] = {
        "enabled": bool(args_cmd.enable_max_bs),
        "status": "skipped",
        "max_bs": None,
        "throughput_at_max_bs_fps": None,
        "kv_mem_max_bs_mb": None,
        "total_mem_max_bs_mb": None,
        "message": "Skipped. Use --enable_max_bs to try max_bs probing.",
    }
    if args_cmd.enable_max_bs:
        try:
            max_bs_probe = _probe_max_bs(
                pipe=pipe,
                args=args,
                prompt=args_cmd.prompt,
                seed=args_cmd.seed + 2_000_000,
            )
            max_bs_result.update(
                {
                    "status": "ok",
                    "max_bs": int(max_bs_probe["max_bs"]),
                    "throughput_at_max_bs_fps": max_bs_probe["throughput_at_max_bs_fps"],
                    "kv_mem_max_bs_mb": max_bs_probe["kv_mem_max_bs_mb"],
                    "total_mem_max_bs_mb": max_bs_probe["total_mem_max_bs_mb"],
                    "elapsed_time_at_max_bs_s": max_bs_probe["elapsed_time_at_max_bs_s"],
                    "attempts": max_bs_probe["attempts"],
                    "message": "max_bs probing succeeded.",
                }
            )
        except Exception as exc:
            max_bs_result.update(
                {
                    "status": "unsupported",
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

    return {
        "script": "temp/bench_config.py",
        "created_at": _now(),
        "config_file": osp.abspath(args_cmd.config_file),
        "checkpoints_dir": osp.abspath(args_cmd.checkpoints_dir),
        "prompt": args_cmd.prompt,
        "seed": int(args_cmd.seed),
        "warmup": int(args_cmd.warmup),
        "runs": int(args_cmd.runs),
        "only_max_bs": bool(args_cmd.only_max_bs),
        "quant_config": quant_config,
        "throughput": throughput,
        "single_bs_memory": single_mem,
        "single_bs_latency": {
            **latency_summary,
            "trace_file": osp.abspath(latency_trace_file) if latency_trace_file else None,
        },
        "max_bs": max_bs_result,
        "metrics": {
            "throughput_fps": throughput["throughput_fps_avg"],
            "max_bs": max_bs_result["max_bs"],
            "throughput_at_max_bs_fps": max_bs_result["throughput_at_max_bs_fps"],
            "kv_mem_single_bs_mb": single_mem["kv_mem_single_bs_mb"],
            "total_mem_single_bs_mb": single_mem["total_mem_single_bs_mb"],
            "saved_video_path": single_mem["saved_video_path"],
            "latency_total_single_bs_s": latency_summary["total_time_s"],
            "latency_attn_single_bs_s": latency_summary["attn_time_s"],
            "latency_ffn_single_bs_s": latency_summary["ffn_time_s"],
            "latency_others_single_bs_s": latency_summary["others_time_s"],
            "kv_mem_max_bs_mb": max_bs_result["kv_mem_max_bs_mb"],
            "total_mem_max_bs_mb": max_bs_result["total_mem_max_bs_mb"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one VAR-Q config by wrapping existing throughput/inference/profiler logic."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to VAR-Q config json.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic shot of a cat walking in the rain, ultra realistic.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--enable_max_bs",
        action="store_true",
        help="Probe the largest runnable batch size with repeated prompts and record throughput/memory.",
    )
    parser.add_argument(
        "--only_max_bs",
        action="store_true",
        help="Skip single-bs throughput/memory/latency and only run max_bs probing.",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save one generated single-bs video during memory/latency run.",
    )
    parser.add_argument(
        "--video_output_dir",
        type=str,
        default=None,
        help="Where to save bench-generated video. Default: <output_dir>/videos",
    )
    parser.add_argument(
        "--video_filename",
        type=str,
        default=None,
        help="Bench-generated video filename. Default: single_bs_seed<seed>.mp4",
    )
    return parser.parse_args()


def main() -> int:
    args_cmd = parse_args()
    os.makedirs(args_cmd.output_dir, exist_ok=True)
    out_file = osp.join(args_cmd.output_dir, "bench_result.json")

    try:
        result = run_benchmark(args_cmd)
        result["status"] = "ok"
    except Exception as exc:
        result = {
            "script": "temp/bench_config.py",
            "created_at": _now(),
            "status": "failed",
            "config_file": osp.abspath(args_cmd.config_file),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[bench_config] saved: {out_file}")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

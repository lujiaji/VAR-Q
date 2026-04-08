#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import glob
import json
import os
import os.path as osp
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Optional


THIS_DIR = osp.dirname(osp.abspath(__file__))
VARQ_ROOT = osp.dirname(THIS_DIR)
BENCH_SCRIPT = osp.join(THIS_DIR, "bench_config.py")
QUALITY_SCRIPT = osp.join(THIS_DIR, "run_quality_100.py")


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not osp.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _run_cmd(
    cmd: List[str],
    log_path: str,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    os.makedirs(osp.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "cmd": cmd,
        "cwd": cwd,
        "log_path": log_path,
        "returncode": int(proc.returncode),
    }


def _extract_bench_metrics(bench_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(bench_json, dict):
        return {}
    m = bench_json.get("metrics", {}) if isinstance(bench_json.get("metrics"), dict) else {}
    return {
        "throughput_fps": m.get("throughput_fps"),
        "max_bs": m.get("max_bs"),
        "throughput_at_max_bs_fps": m.get("throughput_at_max_bs_fps"),
        "kv_mem_single_bs_mb": m.get("kv_mem_single_bs_mb"),
        "total_mem_single_bs_mb": m.get("total_mem_single_bs_mb"),
        "latency_total_single_bs_s": m.get("latency_total_single_bs_s"),
        "latency_attn_single_bs_s": m.get("latency_attn_single_bs_s"),
        "latency_ffn_single_bs_s": m.get("latency_ffn_single_bs_s"),
        "latency_others_single_bs_s": m.get("latency_others_single_bs_s"),
        "kv_mem_max_bs_mb": m.get("kv_mem_max_bs_mb"),
        "total_mem_max_bs_mb": m.get("total_mem_max_bs_mb"),
    }


def _find_vbench_result_json(results_dir: str, config_name: str) -> Optional[str]:
    candidates = [
        osp.join(results_dir, f"{config_name}_combined_eval_results.json"),
        osp.join(results_dir, f"{config_name}_combined_final_score.json"),
        osp.join(results_dir, "final_score.json"),
    ]
    for p in candidates:
        if osp.isfile(p):
            return p
    wildcard = sorted(glob.glob(osp.join(results_dir, "*combined*.json")))
    return wildcard[0] if wildcard else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run bench_config.py + run_quality_100.py over config_dir and summarize results."
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=osp.join(VARQ_ROOT, "temp", "infinity_varq_sageattn_sweep"),
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_list", type=str, default="0,1,2,3")
    parser.add_argument("--skip_bench", action="store_true")
    parser.add_argument("--skip_quality", action="store_true")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", type=str, default="A cinematic shot of a cat walking in the rain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_max_bs", action="store_true")
    parser.add_argument(
        "--bench_cuda_device",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value only for bench_config.py subprocess.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config_files = sorted(glob.glob(osp.join(args.config_dir, "*.json")))
    if not config_files:
        raise SystemExit(f"No config json found in: {args.config_dir}")

    all_rows: List[Dict[str, Any]] = []
    for cfg in config_files:
        config_name = osp.splitext(osp.basename(cfg))[0]
        config_root = osp.join(args.output_dir, config_name)
        bench_out = osp.join(config_root, "bench")
        quality_out = osp.join(config_root, "quality")
        logs_dir = osp.join(config_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        row: Dict[str, Any] = {
            "config_name": config_name,
            "config_file": osp.abspath(cfg),
            "created_at": _now(),
            "bench_status": "skipped" if args.skip_bench else "pending",
            "quality_status": "skipped" if args.skip_quality else "pending",
        }

        try:
            if not args.skip_bench:
                bench_cmd = [
                    args.python_bin,
                    BENCH_SCRIPT,
                    "--config_file",
                    cfg,
                    "--checkpoints_dir",
                    args.checkpoints_dir,
                    "--output_dir",
                    bench_out,
                    "--warmup",
                    str(args.warmup),
                    "--runs",
                    str(args.runs),
                    "--prompt",
                    args.prompt,
                    "--seed",
                    str(args.seed),
                ]
                if args.enable_max_bs:
                    bench_cmd.append("--enable_max_bs")
                bench_env = os.environ.copy()
                if args.bench_cuda_device is not None:
                    bench_env["CUDA_VISIBLE_DEVICES"] = str(args.bench_cuda_device)

                bench_step = _run_cmd(
                    cmd=bench_cmd,
                    log_path=osp.join(logs_dir, "bench_config.log"),
                    cwd=VARQ_ROOT,
                    env=bench_env,
                )
                row["bench_returncode"] = bench_step["returncode"]
                row["bench_log_path"] = bench_step["log_path"]
                row["bench_result_path"] = osp.join(bench_out, "bench_result.json")
                row["bench_status"] = "ok" if bench_step["returncode"] == 0 else "failed"

                bench_json = _read_json_if_exists(row["bench_result_path"])
                row.update(_extract_bench_metrics(bench_json))

            if not args.skip_quality:
                quality_cmd = [
                    args.python_bin,
                    QUALITY_SCRIPT,
                    "--config_file",
                    cfg,
                    "--config_name",
                    config_name,
                    "--checkpoints_dir",
                    args.checkpoints_dir,
                    "--output_dir",
                    quality_out,
                    "--gpu_list",
                    args.gpu_list,
                ]
                quality_step = _run_cmd(
                    cmd=quality_cmd,
                    log_path=osp.join(logs_dir, "run_quality_100.log"),
                    cwd=VARQ_ROOT,
                    env=os.environ.copy(),
                )
                row["quality_returncode"] = quality_step["returncode"]
                row["quality_log_path"] = quality_step["log_path"]
                row["quality_result_path"] = osp.join(quality_out, "quality_result.json")
                row["quality_status"] = "ok" if quality_step["returncode"] == 0 else "failed"

                results_dir = osp.join(quality_out, "vbench_results")
                row["vbench_results_dir"] = results_dir
                row["vbench_score_json"] = _find_vbench_result_json(results_dir, config_name)

        except Exception as exc:
            row["status"] = "failed"
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            row["traceback"] = traceback.format_exc()

        all_rows.append(row)

    summary_json = osp.join(args.output_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "script": "temp/run_batch_experiments.py",
                "created_at": _now(),
                "config_dir": osp.abspath(args.config_dir),
                "output_dir": osp.abspath(args.output_dir),
                "num_configs": len(config_files),
                "results": all_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_csv = osp.join(args.output_dir, "summary.csv")
    csv_fields = [
        "config_name",
        "config_file",
        "bench_status",
        "bench_returncode",
        "quality_status",
        "quality_returncode",
        "throughput_fps",
        "max_bs",
        "throughput_at_max_bs_fps",
        "kv_mem_single_bs_mb",
        "total_mem_single_bs_mb",
        "latency_total_single_bs_s",
        "latency_attn_single_bs_s",
        "latency_ffn_single_bs_s",
        "latency_others_single_bs_s",
        "kv_mem_max_bs_mb",
        "total_mem_max_bs_mb",
        "vbench_score_json",
        "bench_result_path",
        "quality_result_path",
        "vbench_results_dir",
        "bench_log_path",
        "quality_log_path",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in csv_fields})

    print(f"[run_batch_experiments] saved: {summary_json}")
    print(f"[run_batch_experiments] saved: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

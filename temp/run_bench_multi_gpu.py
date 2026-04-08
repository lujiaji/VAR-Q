#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import glob
import json
import os
import os.path as osp
import queue
import subprocess
import sys
import threading
import traceback
from typing import Any, Dict, List


THIS_DIR = osp.dirname(osp.abspath(__file__))
VARQ_ROOT = osp.dirname(THIS_DIR)
BENCH_SCRIPT = osp.join(THIS_DIR, "bench_config.py")


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _resolve_configs(config_dir: str, configs_arg: str) -> List[str]:
    all_cfgs = sorted(glob.glob(osp.join(config_dir, "*.json")))
    if not configs_arg:
        return all_cfgs

    by_stem = {osp.splitext(osp.basename(p))[0]: p for p in all_cfgs}
    by_name = {osp.basename(p): p for p in all_cfgs}
    chosen: List[str] = []
    missing: List[str] = []
    for token in [x.strip() for x in configs_arg.split(",") if x.strip()]:
        if osp.isabs(token) and osp.isfile(token):
            chosen.append(token)
            continue
        candidate = osp.join(config_dir, token)
        if osp.isfile(candidate):
            chosen.append(candidate)
            continue
        if token in by_name:
            chosen.append(by_name[token])
            continue
        if token in by_stem:
            chosen.append(by_stem[token])
            continue
        if f"{token}.json" in by_name:
            chosen.append(by_name[f"{token}.json"])
            continue
        missing.append(token)

    if missing:
        raise FileNotFoundError(f"Configs not found: {missing}")
    dedup = []
    seen = set()
    for p in chosen:
        ap = osp.abspath(p)
        if ap not in seen:
            seen.add(ap)
            dedup.append(ap)
    return dedup


def _run_one(
    *,
    python_bin: str,
    cfg_path: str,
    checkpoints_dir: str,
    output_dir: str,
    warmup: int,
    runs: int,
    prompt: str,
    seed: int,
    enable_max_bs: bool,
    only_max_bs: bool,
    gpu_id: str,
) -> Dict[str, Any]:
    config_name = osp.splitext(osp.basename(cfg_path))[0]
    bench_out = osp.join(output_dir, config_name, "bench")
    logs_dir = osp.join(output_dir, config_name, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(bench_out, exist_ok=True)

    cmd = [
        python_bin,
        BENCH_SCRIPT,
        "--config_file",
        cfg_path,
        "--checkpoints_dir",
        checkpoints_dir,
        "--output_dir",
        bench_out,
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--prompt",
        prompt,
        "--seed",
        str(seed),
        "--save_video",
    ]
    if enable_max_bs:
        cmd.append("--enable_max_bs")
    if only_max_bs:
        cmd.append("--only_max_bs")

    log_path = osp.join(logs_dir, f"bench_gpu{gpu_id}.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=VARQ_ROOT,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    bench_json = osp.join(bench_out, "bench_result.json")
    bench_data = None
    if osp.isfile(bench_json):
        try:
            with open(bench_json, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
        except Exception:
            bench_data = None
    metrics = bench_data.get("metrics", {}) if isinstance(bench_data, dict) else {}

    return {
        "config_name": config_name,
        "config_file": osp.abspath(cfg_path),
        "gpu_id": str(gpu_id),
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
        "log_path": log_path,
        "bench_result_path": bench_json,
        "saved_video_path": metrics.get("saved_video_path"),
        "throughput_fps": metrics.get("throughput_fps"),
        "max_bs": metrics.get("max_bs"),
        "throughput_at_max_bs_fps": metrics.get("throughput_at_max_bs_fps"),
        "kv_mem_single_bs_mb": metrics.get("kv_mem_single_bs_mb"),
        "total_mem_single_bs_mb": metrics.get("total_mem_single_bs_mb"),
        "latency_total_single_bs_s": metrics.get("latency_total_single_bs_s"),
        "latency_attn_single_bs_s": metrics.get("latency_attn_single_bs_s"),
        "latency_ffn_single_bs_s": metrics.get("latency_ffn_single_bs_s"),
        "latency_others_single_bs_s": metrics.get("latency_others_single_bs_s"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run bench_config on multiple GPUs, one config per GPU at a time.")
    p.add_argument("--config_dir", type=str, default=osp.join(VARQ_ROOT, "temp", "infinity_varq_sageattn_sweep"))
    p.add_argument("--configs", type=str, default="", help="Comma separated config names/stems/paths.")
    p.add_argument("--checkpoints_dir", type=str, default="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--gpu_list", type=str, default="0,1,2,3", help="Comma separated gpu ids.")
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--prompt", type=str, default="A cinematic shot of a cat walking in the rain")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable_max_bs", action="store_true")
    p.add_argument("--only_max_bs", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfgs = _resolve_configs(args.config_dir, args.configs)
    if not cfgs:
        raise SystemExit(f"No configs selected from: {args.config_dir}")

    gpus = [g.strip() for g in args.gpu_list.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("gpu_list is empty.")

    q: queue.Queue = queue.Queue()
    for c in cfgs:
        q.put(c)

    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while True:
            try:
                cfg = q.get_nowait()
            except queue.Empty:
                break
            try:
                row = _run_one(
                    python_bin=args.python_bin,
                    cfg_path=cfg,
                    checkpoints_dir=args.checkpoints_dir,
                    output_dir=args.output_dir,
                    warmup=args.warmup,
                    runs=args.runs,
                    prompt=args.prompt,
                    seed=args.seed,
                    enable_max_bs=args.enable_max_bs,
                    only_max_bs=args.only_max_bs,
                    gpu_id=gpu_id,
                )
            except Exception as exc:
                row = {
                    "config_name": osp.splitext(osp.basename(cfg))[0],
                    "config_file": osp.abspath(cfg),
                    "gpu_id": str(gpu_id),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            with lock:
                results.append(row)
            q.task_done()

    threads: List[threading.Thread] = []
    for gpu in gpus:
        t = threading.Thread(target=worker, args=(gpu,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    results = sorted(results, key=lambda x: x.get("config_name", ""))
    summary_json = osp.join(args.output_dir, "bench_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "script": "temp/run_bench_multi_gpu.py",
                "created_at": _now(),
                "config_dir": osp.abspath(args.config_dir),
                "selected_configs": [osp.abspath(c) for c in cfgs],
                "gpu_list": gpus,
                "output_dir": osp.abspath(args.output_dir),
                "num_configs": len(cfgs),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_csv = osp.join(args.output_dir, "bench_summary.csv")
    fields = [
        "config_name",
        "config_file",
        "gpu_id",
        "status",
        "returncode",
        "throughput_fps",
        "max_bs",
        "throughput_at_max_bs_fps",
        "kv_mem_single_bs_mb",
        "total_mem_single_bs_mb",
        "latency_total_single_bs_s",
        "latency_attn_single_bs_s",
        "latency_ffn_single_bs_s",
        "latency_others_single_bs_s",
        "saved_video_path",
        "bench_result_path",
        "log_path",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fields})

    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"[run_bench_multi_gpu] done: {ok}/{len(results)} ok")
    print(f"[run_bench_multi_gpu] saved: {summary_json}")
    print(f"[run_bench_multi_gpu] saved: {summary_csv}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

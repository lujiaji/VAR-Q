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
QUALITY_SCRIPT = osp.join(THIS_DIR, "run_quality_100.py")


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
    gpu_id: str,
    max_videos: int,
    prompts_json: str,
) -> Dict[str, Any]:
    config_name = osp.splitext(osp.basename(cfg_path))[0]
    quality_out = osp.join(output_dir, config_name, "quality")
    logs_dir = osp.join(output_dir, config_name, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(quality_out, exist_ok=True)

    cmd = [
        python_bin,
        QUALITY_SCRIPT,
        "--config_file",
        cfg_path,
        "--config_name",
        config_name,
        "--checkpoints_dir",
        checkpoints_dir,
        "--output_dir",
        quality_out,
        "--gpu_list",
        str(gpu_id),
        "--infer_gpu_id",
        str(gpu_id),
        "--max_videos",
        str(max_videos),
        "--prompts_json",
        prompts_json,
    ]

    log_path = osp.join(logs_dir, f"quality_gpu{gpu_id}.log")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=VARQ_ROOT,
            env=os.environ.copy(),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    result_json = osp.join(quality_out, "quality_result.json")
    result_data = None
    if osp.isfile(result_json):
        try:
            with open(result_json, "r", encoding="utf-8") as f:
                result_data = json.load(f)
        except Exception:
            result_data = None

    step_inf = {}
    step_eval = {}
    if isinstance(result_data, dict):
        steps = result_data.get("steps", {})
        if isinstance(steps, dict):
            step_inf = steps.get("inference_100", {}) if isinstance(steps.get("inference_100"), dict) else {}
            step_eval = steps.get("vbench_quality", {}) if isinstance(steps.get("vbench_quality"), dict) else {}

    return {
        "config_name": config_name,
        "config_file": osp.abspath(cfg_path),
        "gpu_id": str(gpu_id),
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
        "log_path": log_path,
        "quality_result_path": result_json,
        "inference_returncode": step_inf.get("returncode"),
        "vbench_eval_returncode": step_eval.get("returncode"),
        "videos_dir": osp.join(quality_out, "videos"),
        "vbench_results_dir": osp.join(quality_out, "vbench_results"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run quality-100 across configs with multi-GPU queue, supporting multiple concurrent jobs per GPU."
    )
    p.add_argument("--config_dir", type=str, default=osp.join(VARQ_ROOT, "temp", "infinity_varq_sageattn_sweep"))
    p.add_argument("--configs", type=str, default="", help="Comma separated config names/stems/paths.")
    p.add_argument("--checkpoints_dir", type=str, default="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--gpu_list", type=str, default="0,1,2,3")
    p.add_argument("--jobs_per_gpu", type=int, default=1, help="Concurrent config jobs per GPU.")
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--max_videos", type=int, default=100)
    p.add_argument("--prompts_json", type=str, default=osp.join(VARQ_ROOT, "temp", "vbench_100_prompts.json"))
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
    if args.jobs_per_gpu <= 0:
        raise SystemExit("jobs_per_gpu must be > 0.")

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
                    gpu_id=gpu_id,
                    max_videos=args.max_videos,
                    prompts_json=args.prompts_json,
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
        for _ in range(args.jobs_per_gpu):
            t = threading.Thread(target=worker, args=(gpu,), daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()

    results = sorted(results, key=lambda x: x.get("config_name", ""))
    summary_json = osp.join(args.output_dir, "quality_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "script": "temp/run_quality_multi_gpu.py",
                "created_at": _now(),
                "config_dir": osp.abspath(args.config_dir),
                "selected_configs": [osp.abspath(c) for c in cfgs],
                "gpu_list": gpus,
                "jobs_per_gpu": int(args.jobs_per_gpu),
                "output_dir": osp.abspath(args.output_dir),
                "num_configs": len(cfgs),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_csv = osp.join(args.output_dir, "quality_summary.csv")
    fields = [
        "config_name",
        "config_file",
        "gpu_id",
        "status",
        "returncode",
        "inference_returncode",
        "vbench_eval_returncode",
        "quality_result_path",
        "videos_dir",
        "vbench_results_dir",
        "log_path",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fields})

    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"[run_quality_multi_gpu] done: {ok}/{len(results)} ok")
    print(f"[run_quality_multi_gpu] saved: {summary_json}")
    print(f"[run_quality_multi_gpu] saved: {summary_csv}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

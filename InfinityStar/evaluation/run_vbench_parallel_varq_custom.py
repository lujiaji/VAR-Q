#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import os.path as osp
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


QUALITY_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
]


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _snapshot_files(results_dir: Path) -> Tuple[set[str], set[str]]:
    eval_files = {p.name for p in results_dir.glob("results_*_eval_results.json")}
    info_files = {p.name for p in results_dir.glob("results_*_full_info.json")}
    return eval_files, info_files


def _find_newest_new_file(results_dir: Path, before: set[str], pattern: str) -> Optional[Path]:
    candidates = [p for p in results_dir.glob(pattern) if p.name not in before]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _copy_json(src: Path, dst: Path) -> None:
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _run_one_dimension(
    *,
    gpu_id: str,
    dim: str,
    videos_path: Path,
    results_dir: Path,
    model_name: str,
    prompt_file: Path,
    python_env: str,
    vbench_eval_script: str,
    imaging_quality_preprocessing_mode: str,
    log_path: Path,
) -> Dict[str, object]:
    os.makedirs(log_path.parent, exist_ok=True)
    before_eval, before_info = _snapshot_files(results_dir)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("MASTER_ADDR", "localhost")
    env.setdefault("RANK", "0")
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("WORLD_SIZE", "1")
    env["MASTER_PORT"] = str(52000 + (abs(hash((model_name, dim, gpu_id))) % 10000))

    cmd = [
        python_env,
        vbench_eval_script,
        "--videos_path",
        str(videos_path),
        "--output_path",
        str(results_dir),
        "--dimension",
        dim,
        "--mode",
        "custom_input",
        "--prompt_file",
        str(prompt_file),
        "--imaging_quality_preprocessing_mode",
        imaging_quality_preprocessing_mode,
    ]

    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )

    latest_eval = _find_newest_new_file(results_dir, before_eval, "results_*_eval_results.json")
    latest_info = _find_newest_new_file(results_dir, before_info, "results_*_full_info.json")

    named_eval = results_dir / f"{model_name}_{dim}_eval_results.json"
    named_info = results_dir / f"{model_name}_{dim}_full_info.json"
    if latest_eval is not None:
        _copy_json(latest_eval, named_eval)
    if latest_info is not None:
        _copy_json(latest_info, named_info)

    status = "ok" if proc.returncode == 0 and named_eval.exists() else "failed"
    return {
        "dimension": dim,
        "gpu_id": str(gpu_id),
        "cmd": cmd,
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "status": status,
        "named_eval_result": str(named_eval) if named_eval.exists() else None,
        "named_full_info": str(named_info) if named_info.exists() else None,
        "raw_eval_result": str(latest_eval) if latest_eval is not None else None,
        "raw_full_info": str(latest_info) if latest_info is not None else None,
    }


def _combine_named_results(results_dir: Path, model_name: str, dimensions: List[str]) -> Tuple[int, int]:
    combined_results: Dict[str, object] = {}
    combined_full_info: List[object] = []

    for dim in dimensions:
        eval_path = results_dir / f"{model_name}_{dim}_eval_results.json"
        info_path = results_dir / f"{model_name}_{dim}_full_info.json"
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key, value in data.items():
                    combined_results[key] = value
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                combined_full_info.extend(data)
            elif data is not None:
                combined_full_info.append(data)

    if combined_results:
        with open(results_dir / f"{model_name}_combined_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
    if combined_full_info:
        with open(results_dir / f"{model_name}_combined_full_info.json", "w", encoding="utf-8") as f:
            json.dump(combined_full_info, f, indent=2, ensure_ascii=False)
    return len(combined_results), len(combined_full_info)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VBench custom_input quality-only evaluation for VAR-Q outputs.")
    parser.add_argument("--videos_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--gpu_list", type=str, default="0")
    parser.add_argument("--python_env", type=str, default="/home/jiaji_lu/conda/envs/vbench/bin/python")
    parser.add_argument("--vbench_eval_script", type=str, default="/home/jiaji_lu/WM/VBench/evaluate.py")
    parser.add_argument("--vbench_score_script", type=str, default="/home/jiaji_lu/WM/VBench/scripts/cal_final_score.py")
    parser.add_argument("--imaging_quality_preprocessing_mode", type=str, default="longer")
    parser.add_argument("--dimensions", nargs="*", default=QUALITY_DIMENSIONS)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    results_dir = Path(args.results_dir)
    videos_path = Path(args.videos_path)
    prompt_file = Path(args.prompt_file)
    logs_dir = results_dir / "logs"
    status_dir = results_dir / "status"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    if not videos_path.is_dir():
        raise SystemExit(f"videos_path does not exist: {videos_path}")
    if not prompt_file.is_file():
        raise SystemExit(f"prompt_file does not exist: {prompt_file}")

    gpus = [x.strip() for x in str(args.gpu_list).split(",") if x.strip()]
    if not gpus:
        raise SystemExit("gpu_list is empty")
    dimensions = [x.strip() for x in args.dimensions if x.strip()]
    if not dimensions:
        raise SystemExit("dimensions is empty")

    results: List[Dict[str, object]] = []
    for idx, dim in enumerate(dimensions):
        gpu_id = gpus[idx % len(gpus)]
        row = _run_one_dimension(
            gpu_id=gpu_id,
            dim=dim,
            videos_path=videos_path,
            results_dir=results_dir,
            model_name=args.model_name,
            prompt_file=prompt_file,
            python_env=args.python_env,
            vbench_eval_script=args.vbench_eval_script,
            imaging_quality_preprocessing_mode=args.imaging_quality_preprocessing_mode,
            log_path=logs_dir / f"{dim}.log",
        )
        results.append(row)
        if row["status"] == "ok":
            (status_dir / f"{dim}.done").write_text("ok\n", encoding="utf-8")

    combined_count, full_info_count = _combine_named_results(results_dir, args.model_name, dimensions)

    combined_eval = results_dir / f"{args.model_name}_combined_eval_results.json"
    score_status: Dict[str, object] = {"status": "skipped"}
    if combined_eval.exists():
        zip_path = Path("/tmp") / f"{args.model_name}_combined_results.zip"
        zip_proc = subprocess.run(
            ["zip", "-j", str(zip_path), str(combined_eval)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        if zip_proc.returncode == 0 and zip_path.exists():
            score_proc = subprocess.run(
                [
                    args.python_env,
                    args.vbench_score_script,
                    "--zip_file",
                    str(zip_path),
                    "--model_name",
                    f"{args.model_name}_combined",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            score_status = {
                "status": "ok" if score_proc.returncode == 0 else "failed",
                "returncode": int(score_proc.returncode),
                "output": score_proc.stdout,
            }
            zip_path.unlink(missing_ok=True)

    summary = {
        "script": "InfinityStar/evaluation/run_vbench_parallel_varq_custom.py",
        "created_at": _now(),
        "videos_path": str(videos_path),
        "results_dir": str(results_dir),
        "model_name": args.model_name,
        "prompt_file": str(prompt_file),
        "gpu_list": gpus,
        "dimensions": dimensions,
        "results": results,
        "combined_result_items": combined_count,
        "combined_full_info_items": full_info_count,
        "score": score_status,
    }
    with open(results_dir / "custom_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    ok = sum(1 for row in results if row["status"] == "ok")
    print(f"[run_vbench_parallel_varq_custom] done: {ok}/{len(results)} dims ok")
    print(f"[run_vbench_parallel_varq_custom] combined_result_items: {combined_count}")
    print(f"[run_vbench_parallel_varq_custom] combined_full_info_items: {full_info_count}")
    return 0 if ok == len(results) and combined_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

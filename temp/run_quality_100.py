#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import os.path as osp
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Optional


THIS_DIR = osp.dirname(osp.abspath(__file__))
VARQ_ROOT = osp.dirname(THIS_DIR)

FULL_PROMPTS_JSON = osp.join(VARQ_ROOT, "InfinityStar", "evaluation", "VBench_rewrited_prompt.json")
DEFAULT_PROMPTS_100_JSON = osp.join(VARQ_ROOT, "temp", "vbench_100_prompts.json")
VBENCH_INFER_SCRIPT = osp.join(VARQ_ROOT, "InfinityStar", "evaluation", "run_vbench_eval_varq.py")
VBENCH_QUALITY_SCRIPT = osp.join(VARQ_ROOT, "InfinityStar", "evaluation", "run_vbench_parallel_varq_custom.py")

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


def _ensure_prompt_subset_100(
    full_prompts_json: str,
    prompts_100_json: str,
    force_regen: bool,
) -> Dict[str, Any]:
    if osp.exists(prompts_100_json) and (not force_regen):
        with open(prompts_100_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "path": prompts_100_json,
            "count": len(data),
            "created": False,
            "message": "Reused existing prompt subset.",
        }

    with open(full_prompts_json, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)
    subset = list(all_prompts[:100])
    if len(subset) < 100:
        raise ValueError(f"Expected >=100 prompts, got {len(subset)} from {full_prompts_json}")

    os.makedirs(osp.dirname(prompts_100_json), exist_ok=True)
    with open(prompts_100_json, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    return {
        "path": prompts_100_json,
        "count": len(subset),
        "created": True,
        "message": "Created fixed 100-prompt subset from full VBench prompts.",
    }


def _build_prompt_mapping_from_metadata(metadata_json: str, output_json: str) -> Dict[str, Any]:
    with open(metadata_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    mapping: Dict[str, str] = {}
    ok_count = 0
    for item in items:
        video_path = item.get("video_path")
        prompt_en = item.get("prompt_en")
        if not video_path or not prompt_en:
            continue
        video_name = osp.basename(str(video_path))
        mapping[video_name] = str(prompt_en)
        ok_count += 1

    if not mapping:
        raise RuntimeError(f"No successful generated videos found in metadata: {metadata_json}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    return {
        "path": output_json,
        "num_entries": len(mapping),
        "source_metadata": metadata_json,
        "num_success_videos": ok_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 100-prompt quality pipeline by wrapping existing VBench infer/eval scripts."
    )
    parser.add_argument("--config_file", type=str, default=None, help="Quant config json path for VAR-Q/setting.")
    parser.add_argument("--config_name", type=str, default=None, help="Name used in output/model tags.")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/data/jiaji_lu/WM/infinitystar/checkpoints/InfinityStar",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Quality output root.")
    parser.add_argument("--gpu_list", type=str, default="0,1,2,3")
    parser.add_argument(
        "--infer_gpu_id",
        type=int,
        default=None,
        help="GPU id for inference stage. Default: first id in --gpu_list.",
    )
    parser.add_argument("--prompts_json", type=str, default=DEFAULT_PROMPTS_100_JSON)
    parser.add_argument("--full_prompts_json", type=str, default=FULL_PROMPTS_JSON)
    parser.add_argument("--regen_prompts", action="store_true", help="Force regenerate fixed 100 prompt subset.")
    parser.add_argument("--max_videos", type=int, default=100)
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument(
        "--vbench_eval_python",
        type=str,
        default=os.environ.get("PYTHON_ENV", "/home/jiaji_lu/conda/envs/vbench/bin/python"),
        help="Forwarded to run_vbench_parallel_varq.sh via PYTHON_ENV env.",
    )
    parser.add_argument(
        "--quality_dimensions",
        nargs="*",
        default=QUALITY_DIMENSIONS,
        help="Custom-input quality dimensions to evaluate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = osp.join(args.output_dir, "logs")
    videos_dir = osp.join(args.output_dir, "videos")
    results_dir = osp.join(args.output_dir, "vbench_results")
    prompt_mapping_json = osp.join(args.output_dir, "vbench_prompt_mapping.json")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    config_name = args.config_name
    if not config_name:
        if args.config_file:
            config_name = osp.splitext(osp.basename(args.config_file))[0]
        else:
            config_name = "baseline1_sageattn"

    summary_path = osp.join(args.output_dir, "quality_result.json")
    out: Dict[str, Any] = {
        "script": "temp/run_quality_100.py",
        "created_at": _now(),
        "status": "ok",
        "config_name": config_name,
        "config_file": osp.abspath(args.config_file) if args.config_file else None,
        "checkpoints_dir": osp.abspath(args.checkpoints_dir),
        "output_dir": osp.abspath(args.output_dir),
        "videos_dir": osp.abspath(videos_dir),
        "vbench_results_dir": osp.abspath(results_dir),
        "steps": {},
    }

    try:
        prompt_info = _ensure_prompt_subset_100(
            full_prompts_json=args.full_prompts_json,
            prompts_100_json=args.prompts_json,
            force_regen=args.regen_prompts,
        )
        out["steps"]["prepare_prompts"] = {"status": "ok", **prompt_info}

        infer_gpu_id = args.infer_gpu_id
        if infer_gpu_id is None:
            gpu_tokens = [x.strip() for x in str(args.gpu_list).split(",") if x.strip()]
            infer_gpu_id = int(gpu_tokens[0]) if gpu_tokens else 0

        infer_cmd = [
            args.python_bin,
            VBENCH_INFER_SCRIPT,
            "--prompts_json",
            args.prompts_json,
            "--output_dir",
            videos_dir,
            "--checkpoints_dir",
            args.checkpoints_dir,
            "--max_videos",
            str(args.max_videos),
            "--start_idx",
            "0",
            "--gpu_id",
            str(infer_gpu_id),
            "--filename_mode",
            "prompt",
            "--inference_prompt_source",
            "prompt_en",
            "--resume",
        ]
        if args.config_file:
            infer_cmd.extend(["--config_file", args.config_file])

        infer_step = _run_cmd(
            cmd=infer_cmd,
            log_path=osp.join(logs_dir, "run_vbench_eval_varq.log"),
            cwd=VARQ_ROOT,
            env=os.environ.copy(),
        )
        infer_step["status"] = "ok" if infer_step["returncode"] == 0 else "failed"
        out["steps"]["inference_100"] = infer_step

        if infer_step["returncode"] != 0:
            raise RuntimeError("VBench inference step failed.")

        mapping_info = _build_prompt_mapping_from_metadata(
            metadata_json=osp.join(videos_dir, "metadata.json"),
            output_json=prompt_mapping_json,
        )
        out["steps"]["prompt_mapping"] = {"status": "ok", **mapping_info}

        env = os.environ.copy()
        quality_cmd = [
            args.python_bin,
            VBENCH_QUALITY_SCRIPT,
            "--videos_path",
            videos_dir,
            "--results_dir",
            results_dir,
            "--model_name",
            config_name,
            "--prompt_file",
            prompt_mapping_json,
            "--gpu_list",
            args.gpu_list,
            "--python_env",
            args.vbench_eval_python,
            "--dimensions",
            *args.quality_dimensions,
        ]
        quality_step = _run_cmd(
            cmd=quality_cmd,
            log_path=osp.join(logs_dir, "run_vbench_parallel_varq.log"),
            cwd=VARQ_ROOT,
            env=env,
        )
        quality_step["status"] = "ok" if quality_step["returncode"] == 0 else "failed"
        out["steps"]["vbench_quality"] = quality_step

        if quality_step["returncode"] != 0:
            raise RuntimeError("VBench quality evaluation step failed.")

    except Exception as exc:
        out["status"] = "failed"
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)
        out["traceback"] = traceback.format_exc()

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[run_quality_100] saved: {summary_path}")
    return 0 if out.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

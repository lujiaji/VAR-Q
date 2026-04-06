#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


MISSING_PATTERN = re.compile(r"The missing video is: (.+)")


def extract_missing_prompts(log_dir: Path) -> set[str]:
    prompts = set()
    for log_path in sorted(log_dir.glob("*.log")):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = MISSING_PATTERN.search(line)
                if not m:
                    continue
                fname = m.group(1).strip()
                if not fname.endswith(".mp4"):
                    continue
                stem = fname[:-4]
                sep = stem.rfind("-")
                if sep <= 0:
                    continue
                prompt = stem[:sep]
                prompts.add(prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Build prompt subset from VBench missing-video logs.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/vbench_results_varq/logs",
        help="Directory containing dimension logs (*.log).",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default="evaluation/VBench_rewrited_prompt.json",
        help="Prompt source JSON used for generation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/vbench_results_varq/missing_prompts_subset.json",
        help="Output JSON containing only missing prompts.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"log_dir not found: {log_dir}")

    missing_prompts = extract_missing_prompts(log_dir)
    if not missing_prompts:
        print("[Info] No missing videos found in logs.")
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        return

    with open(args.prompts_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    subset = []
    seen = set()
    for item in prompts:
        prompt_en = item.get("prompt_en", "").strip()
        if prompt_en in missing_prompts and prompt_en not in seen:
            subset.append(item)
            seen.add(prompt_en)

    unresolved = sorted(missing_prompts - seen)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    print(f"[Done] missing prompt names in logs: {len(missing_prompts)}")
    print(f"[Done] matched in prompts_json: {len(subset)}")
    print(f"[Done] output: {out_path}")
    if unresolved:
        print(f"[Warn] unresolved prompts: {len(unresolved)}")
        for s in unresolved[:20]:
            print(f"  - {s}")


if __name__ == "__main__":
    main()

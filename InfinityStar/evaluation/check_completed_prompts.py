#!/usr/bin/env python3
"""
Scan output dir for existing mp4 files, determine completed prompts,
output remaining_prompts.json for resume.
"""
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/vbench_eval_varq")
    parser.add_argument("--prompts_json", type=str, default="evaluation/VBench_rewrited_prompt.json")
    parser.add_argument("--remaining_json", type=str, default="evaluation/VBench_remaining_prompts.json")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--sample_start_idx", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output dir not exist: {output_dir}")
        return

    pattern = re.compile(r"^(\d{3})-(\d{2})\.mp4$")
    completed = set()
    for f in output_dir.glob("*.mp4"):
        m = pattern.match(f.name)
        if m:
            pid, sid = int(m.group(1)), int(m.group(2))
            completed.add((pid, sid))

    required_samples = [(args.sample_start_idx + i) for i in range(args.num_samples)]
    total_prompts = len(json.load(open(args.prompts_json)))
    remaining_indices = []
    for pid in range(total_prompts):
        if all((pid, sid) in completed for sid in required_samples):
            pass
        else:
            remaining_indices.append(pid)

    prompts = json.load(open(args.prompts_json))
    remaining = []
    for idx in remaining_indices:
        item = dict(prompts[idx])
        item["original_idx"] = idx
        remaining.append(item)

    out_path = Path(args.remaining_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(remaining, f, indent=2, ensure_ascii=False)

    print(f"Total prompts: {total_prompts}")
    print(f"Completed: {total_prompts - len(remaining_indices)}")
    print(f"Remaining: {len(remaining_indices)}")
    print(f"Remaining prompts saved to: {out_path}")


if __name__ == "__main__":
    main()

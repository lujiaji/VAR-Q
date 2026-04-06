#!/usr/bin/env python3
"""
Rename videos from ID format (000-00.mp4) to VBench format ({prompt_en}-{sample_idx}.mp4).
Videos are assumed to follow VBench_rewrited_prompt order: 000 = prompt[0], 001 = prompt[1], etc.
"""
import argparse
import json
import re
from pathlib import Path


def sanitize_filename(text: str) -> str:
    """Sanitize for filesystem, no truncation."""
    text = re.sub(r"[\\/:*?\"<>|]", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = "empty_prompt"
    return text


def main():
    parser = argparse.ArgumentParser(description="Rename videos from ID format to VBench prompt format")
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/vbench_eval_varq",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default="evaluation/VBench_rewrited_prompt.json",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print renames without applying",
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    prompts_path = Path(args.prompts_json)
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos_dir not found: {videos_dir}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts_json not found: {prompts_path}")

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    id_pattern = re.compile(r"^(\d{3})-(\d{2})\.mp4$")
    rename_count = 0
    skip_count = 0

    for mp4 in sorted(videos_dir.glob("*.mp4")):
        m = id_pattern.match(mp4.name)
        if not m:
            skip_count += 1
            continue

        prompt_idx = int(m.group(1))
        sample_idx_old = int(m.group(2))

        if prompt_idx >= len(prompts):
            skip_count += 1
            continue

        prompt_en = prompts[prompt_idx].get("prompt_en", "").strip()
        if not prompt_en:
            skip_count += 1
            continue

        clean_prompt = sanitize_filename(prompt_en)
        new_name = f"{clean_prompt}-{sample_idx_old}.mp4"
        new_path = videos_dir / new_name

        if mp4.name == new_name:
            continue

        if new_path.exists() and new_path != mp4:
            new_name = f"{clean_prompt}-{sample_idx_old}_id{prompt_idx}.mp4"
            new_path = videos_dir / new_name
            if new_path.exists():
                skip_count += 1
                continue

        if args.dry_run:
            print(f"{mp4.name} -> {new_name}")
        else:
            mp4.rename(new_path)

        rename_count += 1

    print(f"Renamed: {rename_count}, skipped: {skip_count}")
    if args.dry_run and rename_count > 0:
        print("Run without --dry_run to apply renames")


if __name__ == "__main__":
    main()

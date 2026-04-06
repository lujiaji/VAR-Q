#!/usr/bin/env python3
"""
Fix video filenames that were truncated (e.g. '...animated st-0.mp4') 
to full VBench prompt format (e.g. '...animated style-0.mp4').
Uses VBench_full_info.json as the source of truth for exact prompt strings.
"""
import argparse
import json
import re
from pathlib import Path


def sanitize(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|]", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "empty_prompt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/vbench_eval_varq",
    )
    parser.add_argument(
        "--full_info_json",
        type=str,
        default="/home/jiaji_lu/WM/VBench/vbench/VBench_full_info.json",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    with open(args.full_info_json, "r", encoding="utf-8") as f:
        full_info = json.load(f)

    prompt_to_full = {}
    for item in full_info:
        p = item.get("prompt_en", "").strip()
        if p:
            prompt_to_full[sanitize(p)] = p

    pattern = re.compile(r"^(.+)-(\d+)\.mp4$")
    rename_count = 0
    ambiguous = 0
    not_found = 0

    for mp4 in sorted(videos_dir.glob("*.mp4")):
        if re.match(r"^\d{3}-\d{2}\.mp4$", mp4.name):
            continue
        m = pattern.match(mp4.name)
        if not m:
            continue

        stem, sample = m.group(1), m.group(2)
        if stem in prompt_to_full:
            full_prompt = prompt_to_full[stem]
            new_name = f"{full_prompt}-{sample}.mp4"
        else:
            candidates = [p for p in prompt_to_full if p.startswith(stem) or stem.startswith(p)]
            if len(candidates) == 1:
                full_prompt = prompt_to_full[candidates[0]]
                clean_full = sanitize(full_prompt)
                new_name = f"{clean_full}-{sample}.mp4"
            elif len(candidates) > 1:
                best = min(candidates, key=len)
                if stem.startswith(best) or best.startswith(stem):
                    full_prompt = prompt_to_full[best]
                    clean_full = sanitize(full_prompt)
                    new_name = f"{clean_full}-{sample}.mp4"
                else:
                    ambiguous += 1
                    continue
            else:
                not_found += 1
                continue

        new_path = videos_dir / new_name
        if mp4.name == new_name:
            continue
        if new_path.exists() and new_path != mp4:
            ambiguous += 1
            continue

        if args.dry_run:
            print(f"{mp4.name} -> {new_name}")
        else:
            mp4.rename(new_path)
        rename_count += 1

    print(f"Renamed: {rename_count}, ambiguous: {ambiguous}, not_found: {not_found}")


if __name__ == "__main__":
    main()

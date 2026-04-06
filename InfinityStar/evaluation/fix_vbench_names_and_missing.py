#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def sanitize_filename(text: str, max_len: int) -> str:
    text = re.sub(r"[\\/:*?\"<>|]", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = "empty_prompt"
    return text[:max_len].rstrip()


def load_prompts(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_expected_names(prompts, num_samples: int, max_len: int):
    expected = {}
    for item in prompts:
        prompt_en = item.get("prompt_en", "").strip()
        if not prompt_en:
            continue
        clean = sanitize_filename(prompt_en, max_len=max_len)
        for s in range(num_samples):
            fname = f"{clean}-{s}.mp4"
            expected[fname] = prompt_en
    return expected


def main():
    parser = argparse.ArgumentParser(description="Fix truncated VBench video filenames and build missing prompt subset.")
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
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument(
        "--old_max_len",
        type=int,
        default=120,
        help="Old truncation length used by previous generation script.",
    )
    parser.add_argument(
        "--new_max_len",
        type=int,
        default=240,
        help="Current target max length for prompt filenames.",
    )
    parser.add_argument(
        "--missing_subset_out",
        type=str,
        default="/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/vbench_results_varq/missing_prompts_subset_after_rename.json",
    )
    parser.add_argument(
        "--report_out",
        type=str,
        default="/home/jiaji_lu/AR/VAR-Q/Benchmark/outputs/vbench_results_varq/fix_vbench_names_report.json",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply renames. If omitted, run in dry-run mode.",
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    prompts_path = Path(args.prompts_json)
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos_dir not found: {videos_dir}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts_json not found: {prompts_path}")

    prompts = load_prompts(prompts_path)
    expected = build_expected_names(prompts, num_samples=args.num_samples, max_len=args.new_max_len)
    existing = {p.name: p for p in videos_dir.glob("*.mp4")}

    missing_before = [name for name in expected if name not in existing]

    src_to_dsts = defaultdict(list)
    for dst in missing_before:
        prompt_en = expected[dst]
        sample_idx = int(dst[:-4].rsplit("-", 1)[1])
        src = f"{sanitize_filename(prompt_en, max_len=args.old_max_len)}-{sample_idx}.mp4"
        if src in existing:
            src_to_dsts[src].append(dst)

    rename_pairs = []
    ambiguous = {}
    for src, dsts in src_to_dsts.items():
        if len(dsts) == 1:
            dst = dsts[0]
            if src != dst and dst not in existing:
                rename_pairs.append((src, dst))
        else:
            ambiguous[src] = sorted(dsts)

    renamed = 0
    if args.apply:
        for src, dst in rename_pairs:
            src_path = videos_dir / src
            dst_path = videos_dir / dst
            if not src_path.exists() or dst_path.exists():
                continue
            src_path.rename(dst_path)
            renamed += 1

    existing_after = {p.name: p for p in videos_dir.glob("*.mp4")}
    missing_after = [name for name in expected if name not in existing_after]

    missing_prompts = set()
    for name in missing_after:
        missing_prompts.add(expected[name])

    subset = []
    seen = set()
    for item in prompts:
        prompt_en = item.get("prompt_en", "").strip()
        if prompt_en in missing_prompts and prompt_en not in seen:
            subset.append(item)
            seen.add(prompt_en)

    missing_subset_out = Path(args.missing_subset_out)
    missing_subset_out.parent.mkdir(parents=True, exist_ok=True)
    with open(missing_subset_out, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    report = {
        "mode": "apply" if args.apply else "dry_run",
        "videos_dir": str(videos_dir),
        "prompts_json": str(prompts_path),
        "num_samples": args.num_samples,
        "old_max_len": args.old_max_len,
        "new_max_len": args.new_max_len,
        "expected_videos": len(expected),
        "existing_videos_before": len(existing),
        "missing_videos_before": len(missing_before),
        "rename_candidates_unique": len(rename_pairs),
        "renamed_count": renamed,
        "ambiguous_source_count": len(ambiguous),
        "existing_videos_after": len(existing_after),
        "missing_videos_after": len(missing_after),
        "missing_prompt_count_after": len(missing_prompts),
        "missing_subset_out": str(missing_subset_out),
        "rename_examples": rename_pairs[:20],
        "ambiguous_examples": {k: v[:10] for k, v in list(ambiguous.items())[:10]},
    }

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[Done] mode: {report['mode']}")
    print(f"[Done] expected_videos: {report['expected_videos']}")
    print(f"[Done] existing_before: {report['existing_videos_before']}")
    print(f"[Done] missing_before: {report['missing_videos_before']}")
    print(f"[Done] unique_rename_candidates: {report['rename_candidates_unique']}")
    print(f"[Done] renamed_count: {report['renamed_count']}")
    print(f"[Done] ambiguous_sources: {report['ambiguous_source_count']}")
    print(f"[Done] missing_after: {report['missing_videos_after']}")
    print(f"[Done] missing_prompt_count_after: {report['missing_prompt_count_after']}")
    print(f"[Done] missing_subset_out: {missing_subset_out}")
    print(f"[Done] report_out: {report_out}")


if __name__ == "__main__":
    main()

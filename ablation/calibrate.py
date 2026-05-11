#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ablation.quant import (
    save_ablation_stats_from_qkv_dump,
    save_kvquant_codebook_from_qkv_dump,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ablation calibration artifacts from saved QKV dumps.")
    parser.add_argument("--mode", type=str, default="flexgen_stats", choices=["flexgen_stats", "kvquant_codebook"])
    parser.add_argument("--qkv-dump-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--chunk-lengths",
        type=str,
        default="",
        help="Comma-separated chunk lengths, e.g. 1,4,9,16",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--max-samples-per-role", type=int, default=200000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "flexgen_stats":
        chunk_lengths = [int(v.strip()) for v in args.chunk_lengths.split(",") if v.strip()]
        if not chunk_lengths:
            raise ValueError("chunk-lengths must not be empty for flexgen_stats")
        save_ablation_stats_from_qkv_dump(
            dump_dir=args.qkv_dump_dir,
            out_dir=args.out_dir,
            chunk_lengths=chunk_lengths,
        )
    else:
        save_kvquant_codebook_from_qkv_dump(
            dump_dir=args.qkv_dump_dir,
            out_path=args.out_dir,
            bits=args.bits,
            max_samples_per_role=args.max_samples_per_role,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

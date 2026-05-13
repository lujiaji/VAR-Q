#!/usr/bin/env python3
import argparse

import torch

from VAR_Q.quant import VAR_Q
from VAR_Q.profiling import format_memory_breakdown, reset_cuda_memory_stats


def bf16_baseline_cache(steps, batch, step_len, heads, head_dim, device):
    cache = []
    for _ in range(steps):
        cache.append(torch.randn(batch, step_len, heads, head_dim, device=device, dtype=torch.bfloat16))
        _ = torch.cat(cache, dim=1)
    return torch.cat(cache, dim=1)


def varq_cache(steps, batch, step_len, heads, head_dim, bits, method, device):
    quantizer = VAR_Q(
        quant_bits=bits,
        qkv_format="BLHc",
        quant_method=method,
        kv_role="k",
        pack_to_int32=True,
        dequant_dtype="bf16",
    )
    for _ in range(steps):
        item = torch.randn(batch, step_len, heads, head_dim, device=device, dtype=torch.bfloat16)
        _ = quantizer.use_var_q(item)
    return quantizer


def main():
    parser = argparse.ArgumentParser(description="Profile VAR-Q KV cache memory without checkpoints.")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--step_len", type=int, default=64)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--method", type=str, default="VARQ")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        reset_cuda_memory_stats()
    baseline = bf16_baseline_cache(args.steps, args.batch, args.step_len, args.heads, args.head_dim, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        baseline_stats = {
            "packed_kv_bytes": int(baseline.numel() * baseline.element_size()),
            "scale_bytes": 0,
            "dequant_workspace_bytes": 0,
            "cuda_memory_allocated": int(torch.cuda.memory_allocated()),
            "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()),
            "cuda_memory_reserved": int(torch.cuda.memory_reserved()),
            "cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved()),
        }
    else:
        baseline_stats = {"packed_kv_bytes": int(baseline.numel() * baseline.element_size())}
    del baseline

    if device.type == "cuda":
        reset_cuda_memory_stats()
    quantizer = varq_cache(
        args.steps,
        args.batch,
        args.step_len,
        args.heads,
        args.head_dim,
        args.bits,
        args.method,
        device,
    )
    varq_stats = quantizer.memory_breakdown()

    print(f"device={device}")
    print("[BF16 baseline] " + format_memory_breakdown(baseline_stats))
    print("[VAR-Q] " + format_memory_breakdown(varq_stats))


if __name__ == "__main__":
    main()

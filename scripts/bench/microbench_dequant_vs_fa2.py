#!/usr/bin/env python3
"""Synthetic last-step VAR-Q dequant-vs-attention microbenchmark."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VAR_Q.quant import VAR_Q  # noqa: E402


PATCH = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
BATCH = 1
HEAD_DIM = 128
DTYPE = torch.float16
MB = 1_000_000.0
SUPPORTED_BITS = (2, 3, 4, 6, 8)


class BenchShape(NamedTuple):
    patch: Sequence[int]
    cache_patch: Sequence[int]
    seq: int
    cache_tok: int
    last_tok: int
    kv_tok: int
    batch: int
    heads: int
    head_dim: int


def parse_bits(bits_csv: str) -> list[int]:
    bits = [int(part.strip()) for part in bits_csv.split(",") if part.strip()]
    if not bits:
        raise ValueError("--bits must contain at least one bit width")
    unsupported = [bit for bit in bits if bit not in SUPPORTED_BITS]
    if unsupported:
        raise ValueError(f"unsupported packed VAR-Q bit widths: {unsupported}")
    return bits


def make_shape(patch: Sequence[int], heads: int) -> BenchShape:
    seq = sum(p * p for p in patch)
    last_tok = patch[-1] * patch[-1]
    cache_patch = patch[:-1]
    cache_tok = sum(p * p for p in cache_patch)
    kv_tok = cache_tok + last_tok
    assert seq == kv_tok
    return BenchShape(
        patch=patch,
        cache_patch=cache_patch,
        seq=seq,
        cache_tok=cache_tok,
        last_tok=last_tok,
        kv_tok=kv_tok,
        batch=BATCH,
        heads=heads,
        head_dim=HEAD_DIM,
    )


def median_ms_cpu(fn: Callable[[], object], iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()

    times_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(times_ms))


def median_ms_cuda_event(
    fn: Callable[[], object],
    iters: int,
    warmup: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        times_ms.append(float(start.elapsed_time(end)))
    return float(statistics.median(times_ms))


def median_ms_cuda_sync_wall(
    fn: Callable[[], object],
    iters: int,
    warmup: int,
    device: torch.device,
) -> float:
    """Time host allocator calls with CUDA syncs; events do not include CPU time."""
    for _ in range(warmup):
        torch.cuda.synchronize(device)
        fn()
        torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(times_ms))


def make_quantizer(bits: int, kv_role: str) -> VAR_Q:
    return VAR_Q(
        quant_bits=bits,
        qkv_format="BHLc",
        quant_method="VARQ",
        kv_role=kv_role,
        pack_to_int32=True,
        dequant_dtype="fp16",
    )


def randn_bhlc(shape: BenchShape, tokens: int, device: torch.device) -> torch.Tensor:
    return torch.randn(
        (shape.batch, shape.heads, tokens, shape.head_dim),
        device=device,
        dtype=DTYPE,
    )


def randn_blhc(shape: BenchShape, tokens: int, device: torch.device) -> torch.Tensor:
    return torch.randn(
        (shape.batch, tokens, shape.heads, shape.head_dim),
        device=device,
        dtype=DTYPE,
    )


def build_cached_kv(
    bits: int,
    shape: BenchShape,
    device: torch.device,
) -> tuple[VAR_Q, VAR_Q]:
    k_q = make_quantizer(bits, kv_role="k")
    v_q = make_quantizer(bits, kv_role="v")

    for p in shape.cache_patch:
        tokens = p * p
        k_q.use_var_q(randn_bhlc(shape, tokens, device), cache_current=True)
        v_q.use_var_q(randn_bhlc(shape, tokens, device), cache_current=True)

    if k_q.cached_len != shape.cache_tok or v_q.cached_len != shape.cache_tok:
        raise RuntimeError(
            f"cache length mismatch: k={k_q.cached_len}, v={v_q.cached_len}, "
            f"expected={shape.cache_tok}"
        )
    return k_q, v_q


def active_cache_bytes(q: VAR_Q) -> int:
    return int(q.memory_breakdown()["active_cache_bytes"])


def detect_flash_attn(smoke: bool) -> tuple[bool, Callable[..., torch.Tensor] | None]:
    if smoke:
        return False, None
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        return False, None
    return True, flash_attn_func


def make_attention_call(
    q_blhc: torch.Tensor,
    k_blhc: torch.Tensor,
    v_blhc: torch.Tensor,
    flash_attn_func: Callable[..., torch.Tensor] | None,
) -> Callable[[], torch.Tensor]:
    if flash_attn_func is not None:
        return lambda: flash_attn_func(q_blhc, k_blhc, v_blhc, dropout_p=0.0, causal=False)

    q_bhld = q_blhc.transpose(1, 2).contiguous()
    k_bhld = k_blhc.transpose(1, 2).contiguous()
    v_bhld = v_blhc.transpose(1, 2).contiguous()
    return lambda: F.scaled_dot_product_attention(
        q_bhld,
        k_bhld,
        v_bhld,
        dropout_p=0.0,
        is_causal=False,
    )


def dequant_cached_kv(k_q: VAR_Q, v_q: VAR_Q) -> tuple[torch.Tensor, torch.Tensor]:
    return k_q.dequant_all(), v_q.dequant_all()


def current_roundtrip(q: VAR_Q, x_last: torch.Tensor) -> torch.Tensor:
    q.quant(x_last)
    return q.dequant_current()


def alloc_free_workspace(shape: BenchShape, device: torch.device) -> None:
    tmp = torch.empty(
        (shape.batch, shape.heads, shape.cache_tok, shape.head_dim),
        device=device,
        dtype=DTYPE,
    )
    del tmp


def print_env(
    device: torch.device,
    shape: BenchShape,
    flash_available: bool,
) -> None:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = "cpu"
    print(f"torch: {torch.__version__}")
    print(f"device: {device_name}")
    print(f"flash_attn available: {'y' if flash_available else 'n'}")
    print(
        f"SEQ={shape.seq} CACHE_TOK={shape.cache_tok} LAST_TOK={shape.last_tok} "
        f"B={shape.batch} H={shape.heads} D={shape.head_dim}"
    )


def print_table(rows: Iterable[dict[str, float | int | str]]) -> None:
    print()
    print(
        "bits | packedKV_MB | A:t_roundtrip(ms) | B:t_dequant_cached(ms) | "
        "C:t_attn(ms) | fuse_ratio=(B+t_workspace)/C | "
        "total_tax=(A+B+t_workspace+t_empty_cache)/C | verdict"
    )
    print("-" * 168)
    for row in rows:
        print(
            f"{int(row['bits']):>4d} | "
            f"{float(row['packedKV_MB']):>11.3f} | "
            f"{float(row['A_ms']):>17.3f} | "
            f"{float(row['B_ms']):>23.3f} | "
            f"{float(row['C_ms']):>12.3f} | "
            f"{float(row['fuse_ratio']):>28.3f} | "
            f"{float(row['total_tax']):>43.3f} | "
            f"{row['verdict']}"
        )


def ratio(numer: float, denom: float) -> float:
    if denom == 0.0:
        return float("inf")
    return numer / denom


def print_takeaways(rows: Iterable[dict[str, float | int | str]]) -> None:
    for row in rows:
        bits = int(row["bits"])
        a_ms = float(row["A_ms"])
        b_ms = float(row["B_ms"])
        c_ms = float(row["C_ms"])
        print(
            f"takeaway q{bits}: A={a_ms:.3f} ms, B={b_ms:.3f} ms, C={c_ms:.3f} ms; "
            f"the skipped last-scale roundtrip is {ratio(a_ms, b_ms):.3f}x B, "
            f"and cached dequant is {ratio(b_ms, c_ms):.3f}x attention."
        )


def run(args: argparse.Namespace) -> int:
    torch.manual_seed(0)

    if args.smoke:
        shape = make_shape([1, 2, 4], heads=2)
        bits = [8, 4]
        device = torch.device("cpu")
    else:
        shape = make_shape(PATCH, heads=int(args.heads))
        assert shape.seq == 10521, f"expected Infinity-8B SEQ=10521, got {shape.seq}"
        assert shape.cache_tok == 6425, f"expected CACHE_TOK=6425, got {shape.cache_tok}"
        assert shape.last_tok == 4096, f"expected LAST_TOK=4096, got {shape.last_tok}"
        assert shape.kv_tok == 10521, f"expected KV_TOK=10521, got {shape.kv_tok}"
        bits = parse_bits(args.bits)
        device = torch.device(args.device)
        if device.type != "cuda":
            print("error: non-smoke benchmark requires --device cuda", file=sys.stderr)
            return 1
        if not torch.cuda.is_available():
            print("error: --device cuda requested but torch.cuda.is_available() is false", file=sys.stderr)
            return 1
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

    if device.type == "cuda":
        torch.cuda.set_device(device)

    flash_available, flash_attn_func = detect_flash_attn(args.smoke)
    backend = "flash_attn_func" if flash_attn_func is not None else "sdpa"
    print_env(device, shape, flash_available)

    gpu_timer = device.type == "cuda"
    event_timer = (
        (lambda fn: median_ms_cuda_event(fn, args.iters, args.warmup, device))
        if gpu_timer
        else (lambda fn: median_ms_cpu(fn, args.iters, args.warmup))
    )
    host_timer = (
        (lambda fn: median_ms_cuda_sync_wall(fn, args.iters, args.warmup, device))
        if gpu_timer
        else (lambda fn: median_ms_cpu(fn, args.iters, args.warmup))
    )

    with torch.inference_mode():
        q_attn = randn_blhc(shape, shape.last_tok, device)
        k_attn = randn_blhc(shape, shape.kv_tok, device)
        v_attn = randn_blhc(shape, shape.kv_tok, device)
        attn_call = make_attention_call(q_attn, k_attn, v_attn, flash_attn_func)
        t_attn = event_timer(attn_call)

        t_workspace = host_timer(lambda: alloc_free_workspace(shape, device))
        if gpu_timer:
            t_empty_cache = host_timer(torch.cuda.empty_cache)
        else:
            t_empty_cache = 0.0

        rows: list[dict[str, float | int | str]] = []
        for bit in bits:
            k_q, v_q = build_cached_kv(bit, shape, device)
            deq = k_q.dequant_all()
            expected = [shape.batch, shape.heads, shape.cache_tok, shape.head_dim]
            assert list(deq.shape) == expected, f"dequant cache shape {list(deq.shape)} != {expected}"

            t_dequant_cached = event_timer(lambda: dequant_cached_kv(k_q, v_q))

            x_last = randn_bhlc(shape, shape.last_tok, device)
            kq2 = make_quantizer(bit, kv_role="k")
            t_current_roundtrip = event_timer(lambda: current_roundtrip(kq2, x_last))

            packed_kv_mb = (active_cache_bytes(k_q) + active_cache_bytes(v_q)) / MB
            fuse_ratio = ratio(t_dequant_cached + t_workspace, t_attn)
            total_tax = ratio(
                t_current_roundtrip + t_dequant_cached + t_workspace + t_empty_cache,
                t_attn,
            )
            verdict = (
                "FUSE-WORTH"
                if fuse_ratio >= 0.30
                else "fusion small -> trick+overhead dominate"
            )
            rows.append(
                {
                    "bits": bit,
                    "packedKV_MB": packed_kv_mb,
                    "A_ms": t_current_roundtrip,
                    "B_ms": t_dequant_cached,
                    "C_ms": t_attn,
                    "fuse_ratio": fuse_ratio,
                    "total_tax": total_tax,
                    "verdict": verdict,
                }
            )

    print_table(rows)
    print(
        f"footer: t_attn={t_attn:.3f} ms, t_workspace={t_workspace:.3f} ms, "
        f"t_empty_cache={t_empty_cache:.3f} ms, backend={backend}"
    )
    print_takeaways(rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic Infinity-8B last-step VAR-Q dequant vs FA2 benchmark."
    )
    parser.add_argument("--bits", default="8,4,3,2", help="comma-separated VAR-Q bit widths")
    parser.add_argument("--iters", type=int, default=20, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    parser.add_argument("--heads", type=int, default=32, help="attention heads")
    parser.add_argument("--device", default="cuda", help="torch device, default cuda")
    parser.add_argument("--smoke", action="store_true", help="CPU-only tiny sanity run")
    args = parser.parse_args()

    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.heads <= 0:
        parser.error("--heads must be positive")

    try:
        return run(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

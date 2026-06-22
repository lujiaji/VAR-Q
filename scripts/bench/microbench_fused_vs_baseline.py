#!/usr/bin/env python3
"""Synthetic A100 benchmark for fused dequant attention vs baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bench.microbench_dequant_vs_fa2 import (  # noqa: E402
    DTYPE,
    PATCH,
    median_ms_cuda_event,
)
from VAR_Q.fused import fused_dequant_attention  # noqa: E402
from VAR_Q.fused.flash_dequant import _plain_attention  # noqa: E402
from VAR_Q.quant import VAR_Q  # noqa: E402


BATCH = 1
HEADS = 32
HEAD_DIM = 128
QKV_FORMAT = "BHLc"
CACHE_PATCH = list(PATCH[:-1])
LAST_PATCH = PATCH[-1]
FRESH_TOKENS = LAST_PATCH * LAST_PATCH
CACHE_TOKENS = sum(p * p for p in CACHE_PATCH)
FULL_KV_TOKENS = CACHE_TOKENS + FRESH_TOKENS
CORRECTNESS_ATOL = 3e-2
CORRECTNESS_RTOL = 3e-2


class BenchState(NamedTuple):
    q: torch.Tensor
    kq: VAR_Q
    vq: VAR_Q
    fk: torch.Tensor
    fv: torch.Tensor
    flash_attn_func: Callable[..., torch.Tensor]


def make_quantizer(kv_role: str) -> VAR_Q:
    return VAR_Q(
        quant_bits=8,
        qkv_format=QKV_FORMAT,
        quant_method="VARQ",
        kv_role=kv_role,
        pack_to_int32=True,
        dequant_dtype="fp16",
    )


def randn_bhlc(tokens: int, device: torch.device) -> torch.Tensor:
    return torch.randn(
        (BATCH, HEADS, tokens, HEAD_DIM),
        device=device,
        dtype=DTYPE,
    )


def detect_flash_attn() -> tuple[bool, Callable[..., torch.Tensor] | None]:
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        return False, None
    return True, flash_attn_func


def bhlc_to_blhc(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2)


def blhc_to_bhlc(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def build_q8_cache(device: torch.device) -> tuple[VAR_Q, VAR_Q, torch.Tensor, torch.Tensor]:
    kq = make_quantizer("k")
    vq = make_quantizer("v")

    for p in CACHE_PATCH:
        tokens = p * p
        kq.use_var_q(randn_bhlc(tokens, device), cache_current=True)
        vq.use_var_q(randn_bhlc(tokens, device), cache_current=True)

    if kq.cached_len != CACHE_TOKENS or vq.cached_len != CACHE_TOKENS:
        raise RuntimeError(
            f"cache length mismatch: k={kq.cached_len}, v={vq.cached_len}, "
            f"expected={CACHE_TOKENS}"
        )

    ref_k = kq.dequant_all().clone()
    ref_v = vq.dequant_all().clone()
    return kq, vq, ref_k, ref_v


def production_pipeline(state: BenchState) -> torch.Tensor:
    k_full = state.kq.dequant_all()
    v_full = state.vq.dequant_all()
    k = torch.cat([k_full, state.fk], dim=2)
    v = torch.cat([v_full, state.fv], dim=2)
    out = state.flash_attn_func(
        bhlc_to_blhc(state.q),
        bhlc_to_blhc(k),
        bhlc_to_blhc(v),
        dropout_p=0.0,
        causal=False,
    )
    return blhc_to_bhlc(out)


def fused_pipeline(state: BenchState) -> torch.Tensor:
    return fused_dequant_attention(
        state.q,
        state.kq,
        state.vq,
        state.fk,
        state.fv,
        qkv_format=QKV_FORMAT,
    )


def plain_isolated_pipeline(state: BenchState) -> torch.Tensor:
    k_full = state.kq.dequant_all()
    v_full = state.vq.dequant_all()
    k = torch.cat([k_full, state.fk], dim=2)
    v = torch.cat([v_full, state.fv], dim=2)
    return _plain_attention(state.q, k, v)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def correctness_gate(state: BenchState) -> bool:
    production = production_pipeline(state)
    fused = fused_pipeline(state)
    plain = plain_isolated_pipeline(state)
    torch.cuda.synchronize(state.q.device)

    comparisons = (
        ("production", "fused", production, fused),
        ("production", "plain-isolated", production, plain),
        ("fused", "plain-isolated", fused, plain),
    )
    ok = True
    for lhs_name, rhs_name, lhs, rhs in comparisons:
        if not torch.allclose(lhs, rhs, atol=CORRECTNESS_ATOL, rtol=CORRECTNESS_RTOL):
            print(
                f"correctness mismatch: {lhs_name} vs {rhs_name}: "
                f"max_abs_diff={max_abs_diff(lhs, rhs):.6f}",
                file=sys.stderr,
            )
            ok = False
    return ok


def print_env(device: torch.device, flash_available: bool) -> None:
    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"flash_attn available: {'y' if flash_available else 'n'}")
    print(f"qkv_format: {QKV_FORMAT}")
    print(
        f"B={BATCH} H={HEADS} D={HEAD_DIM} dtype=fp16 "
        f"q={FRESH_TOKENS} cached={CACHE_TOKENS} fresh={FRESH_TOKENS} "
        f"full_kv={FULL_KV_TOKENS}"
    )


def print_table(rows: Iterable[tuple[str, float, float]]) -> None:
    print()
    print("pipeline | total_ms | speedup_vs_production")
    print("-" * 48)
    for name, total_ms, speedup in rows:
        print(f"{name:<15} | {total_ms:>8.3f} | {speedup:>22.3f}x")


def validate_static_shape() -> None:
    if CACHE_PATCH != [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48]:
        raise RuntimeError(f"unexpected CACHE_PATCH from microbench helper: {CACHE_PATCH}")
    if LAST_PATCH != 64:
        raise RuntimeError(f"unexpected last patch from microbench helper: {LAST_PATCH}")
    if CACHE_TOKENS != 6425:
        raise RuntimeError(f"expected CACHE_TOKENS=6425, got {CACHE_TOKENS}")
    if FRESH_TOKENS != 4096:
        raise RuntimeError(f"expected FRESH_TOKENS=4096, got {FRESH_TOKENS}")
    if FULL_KV_TOKENS != 10521:
        raise RuntimeError(f"expected FULL_KV_TOKENS=10521, got {FULL_KV_TOKENS}")


def run(args: argparse.Namespace) -> int:
    validate_static_shape()
    torch.manual_seed(0)

    device = torch.device(args.device)
    if device.type != "cuda":
        print("error: this benchmark requires --device cuda", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("error: --device cuda requested but CUDA is not available", file=sys.stderr)
        return 1
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)

    flash_available, flash_attn_func = detect_flash_attn()
    print_env(device, flash_available)
    print(f"iters: {args.iters} warmup: {args.warmup}")
    if flash_attn_func is None:
        print("error: flash_attn_func is required for the production baseline", file=sys.stderr)
        return 1

    with torch.inference_mode():
        kq, vq, ref_k, ref_v = build_q8_cache(device)
        if ref_k.shape != (BATCH, HEADS, CACHE_TOKENS, HEAD_DIM):
            raise RuntimeError(f"unexpected K cache shape: {tuple(ref_k.shape)}")
        if ref_v.shape != (BATCH, HEADS, CACHE_TOKENS, HEAD_DIM):
            raise RuntimeError(f"unexpected V cache shape: {tuple(ref_v.shape)}")

        state = BenchState(
            q=randn_bhlc(FRESH_TOKENS, device),
            kq=kq,
            vq=vq,
            fk=randn_bhlc(FRESH_TOKENS, device),
            fv=randn_bhlc(FRESH_TOKENS, device),
            flash_attn_func=flash_attn_func,
        )

        if not correctness_gate(state):
            return 1
        print("correctness: PASS")

        timer = lambda fn: median_ms_cuda_event(fn, args.iters, args.warmup, device)
        production_ms = timer(lambda: production_pipeline(state))
        fused_ms = timer(lambda: fused_pipeline(state))
        plain_ms = timer(lambda: plain_isolated_pipeline(state))

    rows = [
        ("production", production_ms, 1.0),
        ("fused", fused_ms, production_ms / fused_ms),
        ("plain-isolated", plain_ms, production_ms / plain_ms),
    ]
    print_table(rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark fused VAR-Q dequant attention vs production and plain baselines."
    )
    parser.add_argument("--iters", type=int, default=50, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    parser.add_argument("--device", default="cuda", help="torch CUDA device")
    args = parser.parse_args()

    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

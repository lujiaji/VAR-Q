"""CUDA-only fused packed-KV attention entrypoint."""

from __future__ import annotations

from typing import Iterable

import torch

from .flash_dequant_cuda import fused_flash_dequant_attention


def _normalize_backend(backend: str) -> str:
    value = str(backend).lower().replace("-", "_")
    if value in ("cuda", "cuda_direct", "fused_cuda"):
        return "cuda" if value == "fused_cuda" else value
    if value == "triton":
        # Older configs used this value before the public backend became
        # CUDA-only.  Preserve config compatibility without importing Triton.
        return "cuda_direct"
    raise ValueError(f"Unsupported fused attention backend: {backend}")


def _to_bhlc(tensor: torch.Tensor, layout: str, name: str) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be rank-4, got {tuple(tensor.shape)}")
    if layout == "BHLc":
        return tensor.contiguous()
    if layout == "BLHc":
        return tensor.permute(0, 2, 1, 3).contiguous()
    raise ValueError(f"Unsupported QKV layout {layout!r}; expected BLHc or BHLc")


def _from_bhlc(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "BHLc":
        return tensor
    if layout == "BLHc":
        return tensor.permute(0, 2, 1, 3).contiguous()
    raise ValueError(f"Unsupported QKV layout {layout!r}; expected BLHc or BHLc")


def _build_step_ids(
    group_lengths: Iterable[int],
    cached_len: int,
    device: torch.device,
) -> torch.Tensor:
    lengths = torch.tensor(
        tuple(int(length) for length in group_lengths),
        dtype=torch.int64,
        device=device,
    )
    if lengths.numel() == 0:
        if cached_len == 0:
            return torch.empty(0, dtype=torch.int32, device=device)
        lengths = torch.tensor((cached_len,), dtype=torch.int64, device=device)
    if int(lengths.sum().item()) != int(cached_len):
        raise ValueError(
            "compact scale group lengths must cover the packed cache exactly: "
            f"sum={int(lengths.sum().item())}, cached_len={cached_len}"
        )
    groups = torch.arange(lengths.numel(), dtype=torch.int32, device=device)
    return torch.repeat_interleave(groups, lengths)


def fused_dequant_attention(
    q,
    k_quant,
    v_quant,
    k_fresh,
    v_fresh,
    qkv_format="BHLc",
    backend="cuda_direct",
    softmax_scale=None,
):
    """Run fused CUDA attention over packed history and fresh K/V.

    The quantizers retain compact per-scale metadata.  Only the small
    token-to-scale map is materialized; the full dequantized historical cache
    is never constructed by the direct backend.
    """

    backend = _normalize_backend(backend)
    bits = int(k_quant.quant_bits)
    if bits != int(v_quant.quant_bits):
        raise ValueError("K/V quantizers must use the same bit width")

    cache_layout = str(getattr(k_quant, "qkv_format", qkv_format))
    if cache_layout != str(getattr(v_quant, "qkv_format", cache_layout)):
        raise ValueError("K/V quantizer cache layouts must match")

    k_packed = _to_bhlc(k_quant._valid_cached_item(), cache_layout, "k_packed")
    v_packed = _to_bhlc(v_quant._valid_cached_item(), cache_layout, "v_packed")
    k_scale = _to_bhlc(k_quant._valid_cached_scale(), cache_layout, "k_scale")
    v_scale = _to_bhlc(v_quant._valid_cached_scale(), cache_layout, "v_scale")
    if k_packed.shape != v_packed.shape:
        raise ValueError("K/V packed cache shapes must match")

    q_bhlc = _to_bhlc(q, qkv_format, "q")
    k_fresh_bhlc = _to_bhlc(k_fresh, qkv_format, "k_fresh")
    v_fresh_bhlc = _to_bhlc(v_fresh, qkv_format, "v_fresh")
    step_ids = _build_step_ids(
        getattr(k_quant, "_scale_L_counts", ()),
        int(k_packed.shape[2]),
        q_bhlc.device,
    )

    output = fused_flash_dequant_attention(
        q_bhlc,
        k_packed,
        v_packed,
        k_scale,
        v_scale,
        step_ids,
        k_fresh_bhlc,
        v_fresh_bhlc,
        direct=backend == "cuda_direct",
        softmax_scale=softmax_scale,
    )
    return _from_bhlc(output, qkv_format)

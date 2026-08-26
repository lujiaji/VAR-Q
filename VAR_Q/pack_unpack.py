"""VAR-Q packed-cache operators.

CUDA tensors are handled exclusively by the compiled VAR-Q CUDA extension.
The small PyTorch implementation below is intentionally CPU-only so package
imports and public CPU smoke tests do not require a CUDA toolchain.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


CUDA_PACK_BITS = (2, 3, 4, 6, 8)

# Compatibility symbols for callers written before the CUDA-only backend.
# They do not indicate that Triton is imported or used.
_HAS_TRITON = False
TRITON_PACK_BITS = CUDA_PACK_BITS


def _vals_per_word(bits: int) -> int:
    bits = int(bits)
    if bits not in CUDA_PACK_BITS:
        raise ValueError(f"bits must be one of {CUDA_PACK_BITS}, got {bits}")
    # Q3 deliberately uses only 30 payload bits per int32 word.
    return 10 if bits == 3 else 32 // bits


def _pack_meta(orig_c: int, bits: int) -> Dict[str, int]:
    vals = _vals_per_word(bits)
    packed_words = (int(orig_c) + vals - 1) // vals
    return {
        "orig_c": int(orig_c),
        "vals_per_word": int(vals),
        "pad_len": int(packed_words * vals - int(orig_c)),
        "bits": int(bits),
    }


def _cuda_ops():
    # Lazy import keeps `import VAR_Q` valid on CPU-only GitHub runners.
    from .fused import flash_dequant_cuda

    return flash_dequant_cuda


def pack_last_dim_to_int32_cuda(
    q_int8: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    if not q_int8.is_cuda:
        raise ValueError("CUDA pack requires a CUDA tensor")
    if q_int8.dtype != torch.int8:
        raise TypeError("q_int8 must have dtype torch.int8")
    meta = _pack_meta(int(q_int8.shape[-1]), bits)
    packed = _cuda_ops().pack_int8(q_int8.contiguous(), int(bits))
    return packed, meta


def quantize_pack_last_dim_to_int32_cuda(
    x: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    qkv_format: str,
    *,
    scale_group_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Quantize and pack a rank-4 tensor in one compiled CUDA path."""

    if not x.is_cuda or not scale.is_cuda:
        raise ValueError("CUDA quantize+pack requires CUDA tensors")
    if x.ndim != 4 or scale.ndim != 4:
        raise ValueError("CUDA quantize+pack expects rank-4 tensors")
    packed, _ = _cuda_ops().quantize_pack(
        x.contiguous(),
        int(bits),
        scale=scale.contiguous(),
        group_ids=scale_group_ids,
        scale_dtype=scale.dtype,
        layout=qkv_format,
    )
    return packed, _pack_meta(int(x.shape[-1]), bits)


def unpack_last_dim_from_int32_cuda(
    packed: torch.Tensor,
    meta: Dict[str, int],
) -> torch.Tensor:
    if not packed.is_cuda:
        raise ValueError("CUDA unpack requires a CUDA tensor")
    if packed.dtype != torch.int32:
        raise TypeError("packed must have dtype torch.int32")
    return _cuda_ops().unpack_int8(
        packed.contiguous(),
        int(meta["bits"]),
        int(meta["orig_c"]),
    )


def unpack_dequant_last_dim_from_int32_cuda(
    packed: torch.Tensor,
    scale: torch.Tensor,
    meta: Dict[str, int],
    out: Optional[torch.Tensor],
    qkv_format: str,
    *,
    scale_group_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unpack and dequantize with the compiled CUDA implementation."""

    if not packed.is_cuda or not scale.is_cuda:
        raise ValueError("CUDA unpack+dequant requires CUDA tensors")
    output_dtype = out.dtype if out is not None else scale.dtype
    result = _cuda_ops().unpack_dequant(
        packed.contiguous(),
        scale.contiguous(),
        int(meta["bits"]),
        int(meta["orig_c"]),
        output_dtype=output_dtype,
        group_ids=scale_group_ids,
        layout=qkv_format,
        out=out,
    )
    return result


# Source-compatible names for older callers.  These aliases route to CUDA and
# never import Triton; new code should use the *_cuda names above.
pack_last_dim_to_int32_triton = pack_last_dim_to_int32_cuda
unpack_last_dim_from_int32_triton = unpack_last_dim_from_int32_cuda
unpack_dequant_last_dim_from_int32_triton = unpack_dequant_last_dim_from_int32_cuda


def _signed_to_unsigned_width(x_int8: torch.Tensor, bits: int) -> torch.Tensor:
    return x_int8.to(torch.int32) & ((1 << int(bits)) - 1)


def _unsigned_to_signed_width(unsigned: torch.Tensor, bits: int) -> torch.Tensor:
    bits = int(bits)
    sign = 1 << (bits - 1)
    unsigned = unsigned.to(torch.int32) & ((1 << bits) - 1)
    signed = torch.where((unsigned & sign) != 0, unsigned - (1 << bits), unsigned)
    return signed.to(torch.int8)


def pack_last_dim_to_int32_python(
    q_int8: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """CPU reference path used by smoke tests and CPU-only tooling."""

    if q_int8.is_cuda:
        return pack_last_dim_to_int32_cuda(q_int8, bits)
    meta = _pack_meta(int(q_int8.shape[-1]), bits)
    vals = int(meta["vals_per_word"])
    pad_len = int(meta["pad_len"])
    if pad_len:
        pad = torch.zeros(
            (*q_int8.shape[:-1], pad_len),
            dtype=q_int8.dtype,
            device=q_int8.device,
        )
        q_int8 = torch.cat((q_int8, pad), dim=-1)
    packed_words = q_int8.shape[-1] // vals
    unsigned = _signed_to_unsigned_width(q_int8, bits)
    unsigned = unsigned.view(*q_int8.shape[:-1], packed_words, vals)
    shifts = (torch.arange(vals, dtype=torch.int32) * int(bits)).view(
        *([1] * (unsigned.ndim - 1)), vals
    )
    packed = (unsigned << shifts).sum(dim=-1).to(torch.int32)
    return packed, meta


def unpack_last_dim_from_int32_python(
    packed: torch.Tensor,
    meta: Dict[str, int],
) -> torch.Tensor:
    """CPU reference path used by smoke tests and CPU-only tooling."""

    if packed.is_cuda:
        return unpack_last_dim_from_int32_cuda(packed, meta)
    bits = int(meta["bits"])
    vals = int(meta["vals_per_word"])
    orig_c = int(meta["orig_c"])
    shifts = torch.arange(vals, dtype=torch.int32) * bits
    pieces = [
        (packed >> int(shift)) & ((1 << bits) - 1)
        for shift in shifts.tolist()
    ]
    unpacked = torch.stack(pieces, dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * vals
    )
    unpacked = unpacked[..., :orig_c]
    return _unsigned_to_signed_width(unpacked, bits)

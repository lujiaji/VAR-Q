from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .pack_unpack import (
    CUDA_PACK_BITS,
    unpack_last_dim_from_int32_cuda,
    unpack_last_dim_from_int32_python,
)


DEFAULT_KIVI_GROUP_SIZE = 128
DEFAULT_KIVI_CALI_K_GROUP_SIZE = 128
DEFAULT_KIVI_CALI_V_GROUP_SIZE = 128
SUPPORTED_QKV_FORMATS = ("BLHc", "BHLc")


def resolve_dequant_dtype(dequant_dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dequant_dtype, torch.dtype):
        return dequant_dtype
    if dequant_dtype == "fp32":
        return torch.float32
    if dequant_dtype == "fp16":
        return torch.float16
    if dequant_dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dequant_dtype: {dequant_dtype}")


def _dequant_dtype_name(dequant_dtype: str | torch.dtype) -> str:
    resolved = resolve_dequant_dtype(dequant_dtype)
    if resolved == torch.float32:
        return "fp32"
    if resolved == torch.float16:
        return "fp16"
    return "bf16"


def _normalize_kv_role(kv_role: str) -> str:
    role = str(kv_role).lower()
    if role not in ("k", "v"):
        raise ValueError(f"Unsupported kv_role={kv_role}; expected 'k' or 'v'")
    return role


def _group_lengths_for_axis(total: int, group_size: int) -> List[int]:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if total < 0:
        raise ValueError(f"axis length must be non-negative, got {total}")
    return [min(group_size, total - start) for start in range(0, total, group_size)]


def _to_bhld_layout(tensor: torch.Tensor, qkv_format: str) -> torch.Tensor:
    if qkv_format == "BLHc":
        return tensor.permute(0, 2, 1, 3).contiguous()
    if qkv_format == "BHLc":
        return tensor
    raise ValueError(f"Unsupported qkv_format={qkv_format}")


def _from_bhld_layout(tensor: torch.Tensor, qkv_format: str) -> torch.Tensor:
    if qkv_format == "BLHc":
        return tensor.permute(0, 2, 1, 3).contiguous()
    if qkv_format == "BHLc":
        return tensor
    raise ValueError(f"Unsupported qkv_format={qkv_format}")


def _clone_quant_meta(quant_meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if quant_meta is None:
        return None
    cloned: Dict[str, Any] = {}
    for key, value in quant_meta.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        elif isinstance(value, tuple):
            cloned[key] = tuple(value)
        else:
            cloned[key] = value
    for key in ("group_lengths", "token_group_lengths", "dim_group_lengths"):
        if key in cloned and cloned[key] is not None:
            cloned[key] = [int(v) for v in cloned[key]]
    if "orig_shape" in cloned and cloned["orig_shape"] is not None:
        cloned["orig_shape"] = tuple(int(v) for v in cloned["orig_shape"])
    return cloned


def _normalize_kivi_scheme(scheme: str) -> str:
    normalized = str(scheme).replace("_", "-").upper()
    if normalized in ("KIVI", "ABL-KIVI"):
        return "KIVI"
    if normalized in ("KIVI-CALI", "ABL-KIVI-CALI"):
        return "KIVI-CALI"
    return normalized


def _kivi_group_axis(quant_meta: Dict[str, Any]) -> str:
    scheme = _normalize_kivi_scheme(str(quant_meta.get("scheme", "")))
    if scheme == "KIVI":
        return "TOKEN" if _normalize_kv_role(quant_meta.get("kv_role", "k")) == "k" else "D"
    return str(
        quant_meta.get(
            "group_axis",
            "TOKEN" if _normalize_kv_role(quant_meta.get("kv_role", "k")) == "k" else "HD_FLAT",
        )
    )


def _is_kivi_meta(quant_meta: Optional[Dict[str, Any]]) -> bool:
    if not quant_meta:
        return False
    return _normalize_kivi_scheme(str(quant_meta.get("scheme", ""))) in ("KIVI", "KIVI-CALI")


def _merge_kivi_quant_meta(
    base_meta: Optional[Dict[str, Any]],
    next_meta: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if next_meta is None:
        return _clone_quant_meta(base_meta)
    if base_meta is None:
        return _clone_quant_meta(next_meta)
    if not _is_kivi_meta(base_meta) or not _is_kivi_meta(next_meta):
        raise ValueError("KIVI quant meta merge requires KIVI-compatible metadata.")

    base = _clone_quant_meta(base_meta)
    nxt = _clone_quant_meta(next_meta)
    if _normalize_kivi_scheme(str(base.get("scheme", ""))) != _normalize_kivi_scheme(
        str(nxt.get("scheme", ""))
    ):
        raise ValueError("Cannot merge KIVI quant meta with mismatched schemes.")
    if _normalize_kv_role(base.get("kv_role", "k")) != _normalize_kv_role(nxt.get("kv_role", "k")):
        raise ValueError("Cannot merge KIVI quant meta with mismatched kv_role.")
    if str(base.get("qkv_format", "BLHc")) != str(nxt.get("qkv_format", "BLHc")):
        raise ValueError("Cannot merge KIVI quant meta with mismatched qkv_format.")

    group_axis = _kivi_group_axis(base)
    if group_axis != _kivi_group_axis(nxt):
        raise ValueError("Cannot merge KIVI quant meta with mismatched group_axis.")

    if group_axis == "TOKEN":
        base["group_lengths"] = list(base.get("group_lengths", ())) + list(nxt.get("group_lengths", ()))
    else:
        if list(base.get("group_lengths", ())) != list(nxt.get("group_lengths", ())):
            raise ValueError("Cannot merge KIVI quant meta with mismatched group lengths.")

    base_shape = list(base.get("orig_shape", ()))
    next_shape = list(nxt.get("orig_shape", ()))
    if base_shape and next_shape and len(base_shape) == len(next_shape):
        seq_dim = 1 if str(base.get("qkv_format", "BLHc")) == "BLHc" else 2
        base_shape[seq_dim] = int(base_shape[seq_dim]) + int(next_shape[seq_dim])
        base["orig_shape"] = tuple(base_shape)

    if group_axis == "HD_FLAT":
        for key in ("num_heads", "head_dim"):
            if int(base.get(key, 0)) != int(nxt.get(key, 0)):
                raise ValueError(f"Cannot merge KIVI-CALI metadata with mismatched {key}.")
    return base


def _expand_kivi_scale_for_shape(
    scale: torch.Tensor,
    quant_meta: Optional[Dict[str, Any]],
    target_shape: Sequence[int],
) -> torch.Tensor:
    if not _is_kivi_meta(quant_meta):
        return scale

    qkv_format = str(quant_meta.get("qkv_format", "BLHc"))
    group_axis = _kivi_group_axis(quant_meta)
    group_lengths = [int(v) for v in quant_meta.get("group_lengths", ())]
    if not group_lengths:
        return scale

    counts = torch.tensor(group_lengths, device=scale.device, dtype=torch.long)
    target_len = int(target_shape[1] if qkv_format == "BLHc" else target_shape[2])
    target_dim = int(target_shape[-1])

    if group_axis == "TOKEN":
        scale_bhgd = scale.permute(0, 2, 1, 3).contiguous() if qkv_format == "BLHc" else scale
        expanded = scale_bhgd.repeat_interleave(counts, dim=2)
        expanded = expanded[:, :, :target_len, :]
        if qkv_format == "BLHc":
            return expanded.permute(0, 2, 1, 3).contiguous()
        return expanded

    if group_axis == "D":
        scale_bhlg = scale.permute(0, 2, 1, 3).contiguous() if qkv_format == "BLHc" else scale
        expanded = scale_bhlg.repeat_interleave(counts, dim=3)
        expanded = expanded[:, :, :, :target_dim]
        if qkv_format == "BLHc":
            return expanded.permute(0, 2, 1, 3).contiguous()
        return expanded

    if group_axis == "HD_FLAT":
        num_heads = int(quant_meta.get("num_heads", 0))
        head_dim = int(quant_meta.get("head_dim", 0))
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("KIVI-CALI V metadata must include positive num_heads/head_dim.")
        if scale.dim() == 4:
            scale_blg = scale.squeeze(2) if qkv_format == "BLHc" else scale.squeeze(1)
        elif scale.dim() == 3:
            scale_blg = scale
        else:
            raise ValueError(f"Unsupported KIVI-CALI scale rank: {scale.dim()}")
        expanded = scale_blg.repeat_interleave(counts, dim=2)
        expanded = expanded[:, :target_len, : num_heads * head_dim]
        expanded = expanded.view(expanded.size(0), target_len, num_heads, head_dim)
        if qkv_format == "BLHc":
            return expanded.contiguous()
        return expanded.permute(0, 2, 1, 3).contiguous()

    raise ValueError(f"Unsupported KIVI group_axis={group_axis}")


def dequantize_tensor(
    packed: torch.Tensor,
    scale: torch.Tensor,
    pack_meta: Optional[Dict[str, int]],
    dequant_dtype: str | torch.dtype,
    quant_meta: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    target_dtype = resolve_dequant_dtype(dequant_dtype)
    if pack_meta is not None:
        if packed.dtype != torch.int32:
            raise ValueError(f"Expected packed int32 tensor when pack_meta is present, got {packed.dtype}")
        bits = int(pack_meta["bits"])
        if packed.is_cuda and bits in CUDA_PACK_BITS:
            q_int8 = unpack_last_dim_from_int32_cuda(packed, pack_meta)
        else:
            q_int8 = unpack_last_dim_from_int32_python(packed, pack_meta)
    else:
        q_int8 = packed
    if _is_kivi_meta(quant_meta):
        scale = _expand_kivi_scale_for_shape(scale, quant_meta, q_int8.shape)
    return (q_int8.to(torch.float32) * scale).to(target_dtype)

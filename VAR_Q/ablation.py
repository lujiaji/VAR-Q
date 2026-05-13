"""Compatibility router for older backend adapters.

Some local backend integrations historically imported ``VAR_Q.ablation`` for
both VAR-Q grouping methods and ablation methods.  The public package now keeps
the implementations split: VAR-Q lives in :mod:`VAR_Q.quant`, while ablations
live in the top-level :mod:`ablation` package.  This module preserves the old
import path without merging the runtime implementations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from VAR_Q.quant import (
    CANONICAL_QUANT_METHODS,
    build_kv_cache_quantizer as _build_varq_kv_cache_quantizer,
    dequantize_tensor as _varq_dequantize_tensor,
    quantize_tensor as _varq_quantize_tensor,
)


_FLEXGEN_ALIASES = {
    "ABL_KV_GPTQ": "ABL_KV_FLexGen",
    "GPTQ": "FLexGen",
}


def _normalize_method_name(quant_method: str) -> str:
    return _FLEXGEN_ALIASES.get(str(quant_method), str(quant_method))


def _is_varq_method(quant_method: str) -> bool:
    return _normalize_method_name(quant_method) in CANONICAL_QUANT_METHODS


def quantize_tensor(
    item: torch.Tensor,
    quant_bits: int,
    qkv_format: str,
    quant_method: str,
    pack_to_int32: bool,
    dequant_dtype: str | torch.dtype,
    kv_role: str = "k",
    kivi_group_size: int = 128,
    kivi_cali_k_group_size: int = 128,
    kivi_cali_v_group_size: int = 128,
    ablation_config: Optional[Dict[str, Any]] = None,
    compression_ratio: float = 1.0,
    max_scale_seq_len: Optional[int] = None,
    **_: Any,
) -> Dict[str, Any]:
    method = _normalize_method_name(quant_method)
    if _is_varq_method(method):
        return _varq_quantize_tensor(
            item=item,
            quant_bits=quant_bits,
            qkv_format=qkv_format,
            quant_method=method,
            pack_to_int32=pack_to_int32,
            dequant_dtype=dequant_dtype,
            kv_role=kv_role,
            kivi_group_size=kivi_group_size,
            kivi_cali_k_group_size=kivi_cali_k_group_size,
            kivi_cali_v_group_size=kivi_cali_v_group_size,
            compression_ratio=compression_ratio,
            max_scale_seq_len=max_scale_seq_len,
        )

    from ablation import quantize_tensor as _ablation_quantize_tensor

    return _ablation_quantize_tensor(
        item=item,
        quant_bits=quant_bits,
        qkv_format=qkv_format,
        quant_method=method,
        pack_to_int32=pack_to_int32,
        dequant_dtype=dequant_dtype,
        kv_role=kv_role,
        kivi_group_size=kivi_group_size,
        kivi_cali_k_group_size=kivi_cali_k_group_size,
        kivi_cali_v_group_size=kivi_cali_v_group_size,
        ablation_config=ablation_config,
    )


def dequantize_tensor(
    packed: torch.Tensor,
    scale: torch.Tensor,
    pack_meta: Optional[Dict[str, int]],
    dequant_dtype: str | torch.dtype,
    quant_meta: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> torch.Tensor:
    scheme = str((quant_meta or {}).get("scheme", ""))
    if scheme.startswith("ABL_") or scheme in {"KIVI", "KIVI-cali"}:
        from ablation import dequantize_tensor as _ablation_dequantize_tensor

        return _ablation_dequantize_tensor(
            packed=packed,
            scale=scale,
            pack_meta=pack_meta,
            dequant_dtype=dequant_dtype,
            quant_meta=quant_meta,
        )
    return _varq_dequantize_tensor(
        packed=packed,
        scale=scale,
        pack_meta=pack_meta,
        dequant_dtype=dequant_dtype,
        quant_meta=quant_meta,
    )


def build_kv_cache_quantizer(
    quant_bits: int,
    qkv_format: str,
    quant_method: str,
    **kwargs: Any,
):
    method = _normalize_method_name(quant_method)
    if _is_varq_method(method):
        return _build_varq_kv_cache_quantizer(
            quant_bits=quant_bits,
            qkv_format=qkv_format,
            quant_method=method,
            **kwargs,
        )

    from ablation import build_kv_cache_quantizer as _build_ablation_kv_cache_quantizer

    return _build_ablation_kv_cache_quantizer(
        quant_bits=quant_bits,
        qkv_format=qkv_format,
        quant_method=method,
        **kwargs,
    )


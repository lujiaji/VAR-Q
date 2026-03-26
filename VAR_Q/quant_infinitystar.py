import torch
from typing import Dict, Iterable, List, Optional, Union

from VAR_Q.quant import VAR_Q
from VAR_Q.pack_unpack import (
    pack_last_dim_to_int32_python,
    unpack_last_dim_from_int32_python,
)

try:
    from VAR_Q.pack_unpack import (
        pack_last_dim_to_int32_triton,
        unpack_last_dim_from_int32_triton,
    )
    HAS_TRITON_PACK = True
except Exception:
    HAS_TRITON_PACK = False


ScaleId = Union[int, str]


class InfinityStarVARQ:
    """
    VAR-Q adapter for InfinityStar-style KV cache lifecycle:
    - cache per scale_id
    - fetch selected ref scales by ref_sids
    - optional concat with current kv tensor
    """

    def __init__(
        self,
        quant_bits: int = 8,
        qkv_format: str = "BHLc",
        quant_method: str = "G_SCALE_HEAD_DIM",
        pack_to_int32: bool = True,
        eps: float = 1e-12,
        debug: bool = False,
        rescale_qk: bool = False,
        dequant_dtype: str = "bf16",
    ):
        self.quant_bits = quant_bits
        self.qkv_format = qkv_format
        self.quant_method = quant_method
        self.pack_to_int32 = pack_to_int32
        self.eps = eps
        self.debug = debug
        self.rescale_qk = rescale_qk
        self.dequant_dtype = dequant_dtype

        self._scale_quantizers: Dict[ScaleId, VAR_Q] = {}

    def _new_quantizer(self) -> VAR_Q:
        return VAR_Q(
            quant_bits=self.quant_bits,
            qkv_format=self.qkv_format,
            quant_method=self.quant_method,
            pack_to_int32=self.pack_to_int32,
            eps=self.eps,
            debug=self.debug,
            rescale_qk=self.rescale_qk,
            dequant_dtype=self.dequant_dtype,
        )

    def _ensure_quantizer(self, scale_id: ScaleId) -> VAR_Q:
        if scale_id not in self._scale_quantizers:
            self._scale_quantizers[scale_id] = self._new_quantizer()
        return self._scale_quantizers[scale_id]

    @staticmethod
    def _reset_quantizer_cache(q: VAR_Q) -> None:
        q.cached_item = None
        q.cached_scale = None
        q.quantized_item = None
        q.scale = None
        q._pack_meta = None

    def cache_scale(
        self,
        scale_id: ScaleId,
        kv_tensor: torch.Tensor,
        overwrite: bool = True,
        return_dequant: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Quantize and cache KV for one scale.
        overwrite=True is usually what InfinityStar needs (one final KV per scale).
        """
        q = self._ensure_quantizer(scale_id)
        if overwrite:
            self._reset_quantizer_cache(q)
        if return_dequant:
            return q.use_var_q(kv_tensor)
        q.quant_and_cache(kv_tensor)
        return None

    def has_scale(self, scale_id: ScaleId) -> bool:
        q = self._scale_quantizers.get(scale_id)
        return q is not None and q.cached_item is not None

    def get_scale(self, scale_id: ScaleId) -> torch.Tensor:
        q = self._scale_quantizers.get(scale_id)
        if q is None or q.cached_item is None:
            raise KeyError(f"scale_id={scale_id} is not cached")
        return q.dequant_all()

    def get_scale_quantized(self, scale_id: ScaleId):
        """Return raw cached quantized tensor, scale tensor and pack meta for fused kernels."""
        q = self._scale_quantizers.get(scale_id)
        if q is None or q.cached_item is None or q.cached_scale is None:
            raise KeyError(f"scale_id={scale_id} is not cached")
        return q.cached_item, q.cached_scale, q._pack_meta

    def get_selected(
        self,
        ref_scale_ids: Iterable[ScaleId],
        current_kv: Optional[torch.Tensor] = None,
        cat_dim: int = 2,
    ) -> torch.Tensor:
        """
        Dequantize selected cached scales and concatenate them (plus optional current tensor).
        """
        tensors: List[torch.Tensor] = [self.get_scale(sid) for sid in ref_scale_ids]
        if current_kv is not None:
            tensors.append(current_kv)
        if not tensors:
            raise ValueError("No tensor to concatenate in get_selected()")
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=cat_dim)

    def clear_scales(self, scale_ids: Iterable[ScaleId]) -> None:
        for sid in scale_ids:
            if sid in self._scale_quantizers:
                del self._scale_quantizers[sid]

    def clear_all(self) -> None:
        self._scale_quantizers.clear()

    def cache_bytes(self) -> Dict[str, int]:
        packed_bytes = 0
        scale_bytes = 0
        for q in self._scale_quantizers.values():
            if q.cached_item is not None:
                packed_bytes += q.cached_item.numel() * q.cached_item.element_size()
            if q.cached_scale is not None:
                scale_bytes += q.cached_scale.numel() * q.cached_scale.element_size()
        return {
            "packed_bytes": packed_bytes,
            "scale_bytes": scale_bytes,
            "total_bytes": packed_bytes + scale_bytes,
        }

    def live_scale_ids(self) -> List[ScaleId]:
        return sorted(list(self._scale_quantizers.keys()), key=lambda x: str(x))


__all__ = [
    "InfinityStarVARQ",
    "pack_last_dim_to_int32_python",
    "unpack_last_dim_from_int32_python",
    "HAS_TRITON_PACK",
]

if HAS_TRITON_PACK:
    __all__.extend(
        [
            "pack_last_dim_to_int32_triton",
            "unpack_last_dim_from_int32_triton",
        ]
    )

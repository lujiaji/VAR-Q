import torch
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from VAR_Q.pack_unpack import (
    _HAS_TRITON as _PACK_HAS_TRITON,
    pack_last_dim_to_int32_python,
    unpack_last_dim_from_int32_python,
    TRITON_PACK_BITS,
)

try:
    from VAR_Q.pack_unpack import (
        pack_last_dim_to_int32_triton,
        unpack_last_dim_from_int32_triton,
    )
    _HAS_TRITON = bool(_PACK_HAS_TRITON)
except Exception:
    pack_last_dim_to_int32_triton = None
    unpack_last_dim_from_int32_triton = None
    _HAS_TRITON = False

"""
For autoregressive (AR) models, image generation is performed across multiple scales.  
For example, in VAR there are 10 scales, with patch sizes increasing as: (1, 2, 3, 4, 5, 6, 8, 10, 13, 16).  

At each scale, the number of tokens is given by `patch_num ** 2`.  
Therefore, the total token length **L** is the accumulation across all scales:  

        L = Σ (patch_num ** 2)   over all scales

For VAR, this results in L = 680.  

At each scale, tokens are processed through self-attention layers, which involve Query (Q), Key (K), and Value (V) projections.  
The Q, K, V tensors are typically shaped as `BLHc` or `BHLc`, where:  
- **B**: batch size (number of instances during generation)  
- **L**: total token length (sum of tokens across all scales, e.g., 680 in VAR)  
- **H**: number of attention heads  
- **c**: dimension of each head  

In VAR-Q, we define different grouping strategies for quantization:  

- **G_TENSOR**: treat the entire tensor as a single unit.  
  This results in only one global scaling factor for all Q/K/V values.  

- **G_SCALE_HEAD_DIM**: quantize each incoming K/V tensor per scale before concatenating with the cached K/V.  
  Each scale maintains its own scaling factors, leading to `num_scales × H × c` groups (e.g., `10 × 20 × 64 = 12,800` for VAR).  

- **VARQ**: quantize each incoming K/V tensor per scale, per batch sample, and per feature.  
  This keeps one scale for every `B × H × c` group inside the current AR step, which is useful when
  different samples in the batch have noticeably different activation ranges.

- **G_HEAD_DIM**: first dequantize cached tensors, concatenate with the new tensor, and then quantize the entire result along the head and dimension axes.  
  This produces `H × c` groups (e.g., `20 × 64` for VAR).  

- **G_SCALE**: group tensors by scale.  
  Since VAR has 10 scales, this corresponds to exactly 10 groups.  

- **G_TOKEN**: group tensors along the token axis **L**.  
  For VAR, this results in 680 or 2240 groups (depending on configuration).  

- **G_TOKEN_HEAD**: group tensors by both token and head dimensions.  
  This yields `L × H` groups (e.g., `2240 × 20` for VAR).  

These grouping strategies allow us to explore different quantization granularities for Q/K/V tensors in AR-based image generation models.

"""

CANONICAL_QUANT_METHODS = (
    "VARQ",
    "G_TENSOR",
    "G_SCALE_HEAD_DIM",
    "G_HEAD_DIM",
    "G_SCALE",
    "G_TOKEN",
    "G_TOKEN_HEAD",
)

SUPPORTED_QUANT_METHODS = CANONICAL_QUANT_METHODS

DEFAULT_QUANT_METHOD = "VARQ"
DEFAULT_COMPRESSION_RATIO = 1.0


def normalize_quant_method(quant_method: str) -> str:
    return str(quant_method)


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
    cloned = dict(quant_meta)
    if "group_lengths" in cloned and cloned["group_lengths"] is not None:
        cloned["group_lengths"] = [int(v) for v in cloned["group_lengths"]]
    if "orig_shape" in cloned and cloned["orig_shape"] is not None:
        cloned["orig_shape"] = tuple(int(v) for v in cloned["orig_shape"])
    return cloned


def _is_compact_scale_meta(quant_meta: Optional[Dict[str, Any]]) -> bool:
    return bool(quant_meta) and str(quant_meta.get("scheme", "")) == "COMPACT_SCALE"


def _expand_compact_scale_for_shape(
    scale: torch.Tensor,
    quant_meta: Dict[str, Any],
    target_shape: Sequence[int],
) -> torch.Tensor:
    group_lengths = [int(v) for v in quant_meta.get("group_lengths", ())]
    if not group_lengths:
        return scale
    qkv_format = str(quant_meta.get("qkv_format", "BLHc"))
    seq_dim = 1 if qkv_format == "BLHc" else 2
    expanded = scale.repeat_interleave(
        torch.tensor(group_lengths, device=scale.device, dtype=torch.long),
        dim=seq_dim,
    )
    target_len = int(target_shape[seq_dim])
    if expanded.size(seq_dim) > target_len:
        slices = [slice(None)] * expanded.ndim
        slices[seq_dim] = slice(0, target_len)
        expanded = expanded[tuple(slices)]
    return expanded

class VAR_Q:
    def __init__(
        self,
        quant_bits: int = 8,
        qkv_format: str = 'BLHc',  # (B,L,H,c) or (B,H,L,c)
        quant_method: str = DEFAULT_QUANT_METHOD,
        kv_role: str = "k",
        compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
        max_scale_seq_len: Optional[int] = None,
        blk_idx: int = 0,
        pack_to_int32: bool = True,
        eps: float = 1e-12,
        debug: bool = False,
        rescale_qk = False,
        dequant_dtype: str = "bf16",
    ):
        assert qkv_format in ('BLHc','BHLc'), f"Invalid qkv_format: {qkv_format}"
        quant_method = normalize_quant_method(quant_method)
        assert quant_method in CANONICAL_QUANT_METHODS, \
            f"Invalid quant_method: {quant_method}"
        kv_role = _normalize_kv_role(kv_role)
        assert 1 < quant_bits <= 8, "Only support (2..8] bits; validated entry points allow 2/3/4/6/8 bits"
        if pack_to_int32 and quant_bits not in (2, 3, 4, 6, 8):
            raise ValueError(
                f"pack_to_int32=True only supports q_bits in (2, 3, 4, 6, 8), got q_bits={quant_bits}"
            )
        if not (0.0 < float(compression_ratio)):
            raise ValueError(
                f"compression_ratio must be positive, got {compression_ratio}"
            )
        if max_scale_seq_len is not None and int(max_scale_seq_len) <= 0:
            raise ValueError(
                f"max_scale_seq_len must be positive when provided, got {max_scale_seq_len}"
            )
        self.quant_bits = quant_bits
        self.quant_method = quant_method
        self.kv_role = kv_role
        self.compression_ratio = float(compression_ratio)
        self.max_scale_seq_len = int(max_scale_seq_len) if max_scale_seq_len is not None else None
        self.cur_blk_idx = blk_idx
        self.pack_to_int32 = pack_to_int32
        self.eps = eps
        self.debug = debug
        self.rescale_qk_enabled = rescale_qk
        if dequant_dtype == "fp32":
            self.dequant_dtype = torch.float32
        elif dequant_dtype == "fp16":
            self.dequant_dtype = torch.float16
        else:
            self.dequant_dtype = torch.bfloat16
        self.scale_dtype = torch.float16 if self.dequant_dtype == torch.float16 else torch.bfloat16

        self._qkv_format = None
        self.set_qkv_format(qkv_format)

        self.bound_min = -(2 ** (quant_bits - 1))
        self.bound_max =  (2 ** (quant_bits - 1)) - 1

        # Cache: maybe int8 or packed int32. Buffers grow geometrically and use
        # valid lengths to avoid torch.cat on every AR step.
        self.cached_item: Optional[torch.Tensor] = None
        self.cached_scale: Optional[torch.Tensor] = None
        self.cached_quant_meta: Optional[Dict[str, Any]] = None
        self.cached_len: int = 0
        self.cached_scale_len: int = 0
        self._cache_capacity: int = 0
        self._scale_cache_capacity: int = 0
        self._dequant_workspace: Optional[torch.Tensor] = None
        self._dequant_workspace_capacity: int = 0
        self._dequant_workspace_peak_bytes: int = 0

        # Current step
        self.quantized_item: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        self.quant_meta: Optional[Dict[str, Any]] = None
        self._replace_cache_on_next_cache: bool = False

        # pack/unpack meta (assuming c is constant across steps, so meta is constant)
        self._pack_meta: Optional[Dict[str,int]] = None  # {'orig_c','vals_per_word','pad_len','bits'}

        # Per-step token counts for compact-scale methods
        # (G_SCALE_HEAD_DIM, VARQ, G_SCALE, G_TENSOR).
        # These methods produce a single scale vector (L-dim=1) per step; we store it compactly
        # and use repeat_interleave to reconstruct full-L scale only during dequantization.
        self._scale_L_counts: list = []
        self._cur_scale_L_counts: list = []

    def set_qkv_format(self, fmt: str):
        """Update qkv_format and synchronize dim_cat / dim_map accordingly."""
        if fmt == self._qkv_format:
            return
        self._qkv_format = fmt
        if fmt == 'BLHc':
            self.dim_cat = 1
            self.dim_map = {
                'G_TENSOR':        (0,1,2,3),
                'VARQ':            (1,),
                'G_SCALE_HEAD_DIM':(0,1),
                'G_HEAD_DIM':      (0,1),
                'G_TOKEN':         (0,2,3),
                'G_TOKEN_HEAD':    (0,3),
                'G_SCALE':         (0,1,2,3),
            }
        else:  # 'BHLc'
            self.dim_cat = 2
            self.dim_map = {
                'G_TENSOR':        (0,1,2,3),
                'VARQ':            (2,),
                'G_SCALE_HEAD_DIM':(0,2),
                'G_HEAD_DIM':      (0,2),
                'G_TOKEN':         (0,1,3),
                'G_TOKEN_HEAD':    (0,3),
                'G_SCALE':         (0,1,2,3),
            }

    @property
    def qkv_format(self):
        return self._qkv_format

    @qkv_format.setter
    def qkv_format(self, fmt: str):
        self.set_qkv_format(fmt)

    # ---------- Basic numerical operations ----------
    def _compute_scale(self, x: torch.Tensor, reduce_dims: Tuple[int,...], keepdim=True) -> torch.Tensor:
        x32 = x.to(torch.float32)
        max_abs = x32.abs().amax(dim=reduce_dims, keepdim=keepdim)
        scale = (max_abs / float(self.bound_max)).clamp_min(self.eps)
        return scale.to(self.scale_dtype).detach()

    def _quantize_to_int8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)
        q = torch.round(x32 / scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
        return q

    def _dequantize_from_int8(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (q.to(self.dequant_dtype) * scale.to(self.dequant_dtype)).to(self.dequant_dtype)

    def _seq_dim_size(self, tensor: torch.Tensor) -> int:
        return int(tensor.size(self.dim_cat))

    def _compact_chunk_lengths(self, tensor: torch.Tensor) -> List[int]:
        seq_len = self._seq_dim_size(tensor)
        if seq_len <= 0:
            return []
        if self.max_scale_seq_len is None and self.compression_ratio >= 1.0:
            return [seq_len]
        base_len = int(self.max_scale_seq_len or seq_len)
        chunk_seq_len = max(1, int(torch.ceil(torch.tensor(base_len * self.compression_ratio)).item()))
        if chunk_seq_len >= seq_len:
            return [seq_len]
        return [min(chunk_seq_len, seq_len - start) for start in range(0, seq_len, chunk_seq_len)]

    def _slice_seq(self, tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
        slices = [slice(None)] * tensor.ndim
        slices[self.dim_cat] = slice(start, end)
        return tensor[tuple(slices)]

    def _quantize_compact_chunks(
        self,
        item: torch.Tensor,
        reduce_dims: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        chunk_lengths = self._compact_chunk_lengths(item)
        if not chunk_lengths:
            raise ValueError("Cannot quantize empty tensor with compact chunking.")
        q_chunks: List[torch.Tensor] = []
        scale_chunks: List[torch.Tensor] = []
        start = 0
        for chunk_len in chunk_lengths:
            end = start + chunk_len
            chunk = self._slice_seq(item, start, end)
            scale_chunk = self._compute_scale(chunk, reduce_dims, keepdim=True)
            q_chunks.append(self._quantize_to_int8(chunk, scale_chunk))
            scale_chunks.append(scale_chunk)
            start = end
        q = torch.cat(q_chunks, dim=self.dim_cat)
        scale = torch.cat(scale_chunks, dim=self.dim_cat)
        return q, scale, chunk_lengths

    def _to_bhld(self, tensor: torch.Tensor) -> torch.Tensor:
        return _to_bhld_layout(tensor, self.qkv_format)

    def _from_bhld(self, tensor: torch.Tensor) -> torch.Tensor:
        return _from_bhld_layout(tensor, self.qkv_format)

    def _expand_scale_for_tensor(
        self,
        scale: torch.Tensor,
        quant_meta: Optional[Dict[str, Any]],
        target_shape: Sequence[int],
    ) -> torch.Tensor:
        if _is_compact_scale_meta(quant_meta):
            return _expand_compact_scale_for_shape(scale, quant_meta, target_shape)
        return scale

    def _slice_along_cat(self, tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
        slices = [slice(None)] * tensor.ndim
        slices[self.dim_cat] = slice(start, end)
        return tensor[tuple(slices)]

    def _valid_cached_item(self) -> torch.Tensor:
        if self.cached_item is None:
            raise RuntimeError("cached_item is not initialized")
        return self._slice_along_cat(self.cached_item, 0, self.cached_len)

    def _valid_cached_scale(self) -> torch.Tensor:
        if self.cached_scale is None:
            raise RuntimeError("cached_scale is not initialized")
        return self._slice_along_cat(self.cached_scale, 0, self.cached_scale_len)

    def _next_capacity(self, needed: int, current: int) -> int:
        if current >= needed:
            return current
        new_capacity = max(needed, 1 if current == 0 else current)
        while new_capacity < needed:
            new_capacity *= 2
        return new_capacity

    def _ensure_sequence_buffer(
        self,
        attr: str,
        capacity_attr: str,
        valid_len: int,
        append_tensor: torch.Tensor,
    ) -> torch.Tensor:
        needed = valid_len + int(append_tensor.size(self.dim_cat))
        current_buffer = getattr(self, attr)
        current_capacity = int(getattr(self, capacity_attr))
        if current_buffer is not None and current_capacity >= needed:
            return current_buffer

        new_capacity = self._next_capacity(needed, current_capacity)
        shape = list(append_tensor.shape)
        shape[self.dim_cat] = new_capacity
        new_buffer = torch.empty(shape, dtype=append_tensor.dtype, device=append_tensor.device)
        if current_buffer is not None and valid_len > 0:
            with torch.no_grad():
                self._slice_along_cat(new_buffer, 0, valid_len).copy_(
                    self._slice_along_cat(current_buffer, 0, valid_len)
                )
        setattr(self, attr, new_buffer)
        setattr(self, capacity_attr, new_capacity)
        return new_buffer

    def _append_cached_item(self, q_cur: torch.Tensor) -> None:
        buffer = self._ensure_sequence_buffer("cached_item", "_cache_capacity", self.cached_len, q_cur)
        start = self.cached_len
        end = start + int(q_cur.size(self.dim_cat))
        with torch.no_grad():
            self._slice_along_cat(buffer, start, end).copy_(q_cur.detach())
        self.cached_len = end

    def _append_cached_scale(self, scale_cur: torch.Tensor) -> None:
        scale_cur = scale_cur.to(self.scale_dtype).detach()
        buffer = self._ensure_sequence_buffer(
            "cached_scale",
            "_scale_cache_capacity",
            self.cached_scale_len,
            scale_cur,
        )
        start = self.cached_scale_len
        end = start + int(scale_cur.size(self.dim_cat))
        with torch.no_grad():
            self._slice_along_cat(buffer, start, end).copy_(scale_cur)
        self.cached_scale_len = end

    def _reset_cache_buffers(self) -> None:
        self.cached_len = 0
        self.cached_scale_len = 0
        self.cached_quant_meta = None
        self._scale_L_counts = []

    def _snapshot_cache_state(self) -> Dict[str, Any]:
        return {
            "cached_len": self.cached_len,
            "cached_scale_len": self.cached_scale_len,
            "cached_quant_meta": _clone_quant_meta(self.cached_quant_meta),
            "scale_L_counts": list(self._scale_L_counts),
        }

    def _restore_cache_state(self, state: Dict[str, Any]) -> None:
        self.cached_len = int(state["cached_len"])
        self.cached_scale_len = int(state["cached_scale_len"])
        self.cached_quant_meta = _clone_quant_meta(state["cached_quant_meta"])
        self._scale_L_counts = list(state["scale_L_counts"])

    def _ensure_dequant_workspace(self, q_int8: torch.Tensor) -> torch.Tensor:
        needed = int(q_int8.size(self.dim_cat))
        workspace = self._dequant_workspace
        compatible = (
            workspace is not None
            and workspace.dtype == self.dequant_dtype
            and workspace.device == q_int8.device
            and workspace.ndim == q_int8.ndim
            and all(
                workspace.size(i) == q_int8.size(i)
                for i in range(q_int8.ndim)
                if i != self.dim_cat
            )
        )
        if compatible and self._dequant_workspace_capacity >= needed:
            return workspace
        new_capacity = self._next_capacity(
            needed,
            self._dequant_workspace_capacity if compatible else 0,
        )
        shape = list(q_int8.shape)
        shape[self.dim_cat] = new_capacity
        self._dequant_workspace = torch.empty(shape, dtype=self.dequant_dtype, device=q_int8.device)
        self._dequant_workspace_capacity = new_capacity
        self._dequant_workspace_peak_bytes = max(
            self._dequant_workspace_peak_bytes,
            self._tensor_bytes(self._dequant_workspace),
        )
        return self._dequant_workspace

    def release_dequant_workspace(self) -> None:
        """Drop temporary dequant storage after attention consumes it."""
        self._dequant_workspace = None
        self._dequant_workspace_capacity = 0

    def _write_dequant_into_workspace(
        self,
        q_int8: torch.Tensor,
        scale: torch.Tensor,
        quant_meta: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        q_int8 = q_int8.detach()
        scale = scale.detach()
        workspace = self._ensure_dequant_workspace(q_int8)
        out = self._slice_along_cat(workspace, 0, int(q_int8.size(self.dim_cat)))
        if _is_compact_scale_meta(quant_meta):
            group_lengths = [int(v) for v in quant_meta.get("group_lengths", ())]
            if not group_lengths:
                group_lengths = [int(q_int8.size(self.dim_cat))]
            token_start = 0
            for scale_idx, group_len in enumerate(group_lengths):
                token_end = min(token_start + group_len, int(q_int8.size(self.dim_cat)))
                if token_start >= token_end:
                    break
                q_chunk = self._slice_along_cat(q_int8, token_start, token_end)
                scale_chunk = self._slice_along_cat(scale, scale_idx, scale_idx + 1)
                out_chunk = self._slice_along_cat(out, token_start, token_end)
                torch.mul(
                    q_chunk.to(self.dequant_dtype),
                    scale_chunk.to(self.dequant_dtype),
                    out=out_chunk,
                )
                token_start = token_end
        else:
            scale = self._expand_scale_for_tensor(scale, quant_meta, q_int8.shape)
            torch.mul(q_int8.to(self.dequant_dtype), scale.to(self.dequant_dtype), out=out)
        return out
    
    # ---------- Pack/unpack last dimension to/from int32 ----------
    def _pack_last_dim_to_int32(self, q_int8: torch.Tensor, bits: int) -> Tuple[torch.Tensor, Dict[str,int]]:
        if _HAS_TRITON and q_int8.is_cuda and bits in TRITON_PACK_BITS:
            return pack_last_dim_to_int32_triton(q_int8, bits)
        else:
            return pack_last_dim_to_int32_python(q_int8, bits)

    def _unpack_last_dim_from_int32(self, packed: torch.Tensor, meta: Dict[str,int]) -> torch.Tensor:
        bits = int(meta["bits"])
        if _HAS_TRITON and packed.is_cuda and bits in TRITON_PACK_BITS:
            return unpack_last_dim_from_int32_triton(packed, meta)
        else:
            return unpack_last_dim_from_int32_python(packed, meta)

    # ---------- Main quantization process ----------
    def quant(self, item: torch.Tensor):
        m = self.quant_method
        self.quant_meta = None
        self._replace_cache_on_next_cache = False
        self._cur_scale_L_counts = []
        red = self.dim_map[m]

        if m in ('VARQ', 'G_SCALE_HEAD_DIM', 'G_TOKEN', 'G_TOKEN_HEAD', 'G_SCALE'):
            if m in ('VARQ', 'G_SCALE_HEAD_DIM', 'G_SCALE') and (self.compression_ratio < 1.0 or self.max_scale_seq_len is not None):
                q, scale, chunk_lengths = self._quantize_compact_chunks(item, red)
                self._cur_scale_L_counts = list(chunk_lengths)
                self.quant_meta = {
                    "scheme": "COMPACT_SCALE",
                    "group_lengths": list(chunk_lengths),
                    "qkv_format": self.qkv_format,
                    "orig_shape": tuple(int(v) for v in item.shape),
                }
            else:
                scale = self._compute_scale(item, red, keepdim=True)
                q = self._quantize_to_int8(item, scale)
                if m in self._COMPACT_SCALE_METHODS:
                    self._cur_scale_L_counts = [q.size(self.dim_cat)]

        elif m in ('G_HEAD_DIM', 'G_TENSOR'):
            if self.cached_item is not None and self.cached_scale is not None and self.cached_len > 0:
                cached_deq = self.dequant_all()
                cat = torch.cat([cached_deq, item], dim=self.dim_cat)
                scale = self._compute_scale(cat, red, keepdim=True)
                q = self._quantize_to_int8(cat, scale)
            else:
                scale = self._compute_scale(item, red, keepdim=True)
                q = self._quantize_to_int8(item, scale)
        else:
            raise ValueError(f"[VAR-Q]: Invalid quantization method: {m}")

        if self.pack_to_int32:
            q_packed, meta = self._pack_last_dim_to_int32(q, self.quant_bits)
            if self._pack_meta is not None:
                assert meta['bits'] == self._pack_meta['bits']
                assert meta['vals_per_word'] == self._pack_meta['vals_per_word']
                assert meta['orig_c'] == self._pack_meta['orig_c']
                assert meta['pad_len'] == self._pack_meta['pad_len']
            self._pack_meta = meta
            self.quantized_item = q_packed
        else:
            self.quantized_item = q

        self.scale = scale

    # ---------- Reconstruct compact scale to full L dimension ----------
    _COMPACT_SCALE_METHODS = frozenset(('VARQ', 'G_SCALE_HEAD_DIM', 'G_SCALE', 'G_TENSOR'))

    def _reconstruct_scale_for_L(self, scale: torch.Tensor, target_L: int) -> torch.Tensor:
        """Expand compact scale from (num_scales) to (L_total) along L-dim via repeat_interleave.

        For compact-scale methods the cached scale has shape (..., num_scales, ...)
        where num_scales == number of AR steps, while the quantised data has L_total
        tokens.  This rebuilds the full-L scale as a *temporary* tensor so the
        permanent cached_scale stays small.
        """
        L_dim = self.dim_cat
        if scale.size(L_dim) >= target_L:
            return scale
        counts = torch.tensor(self._scale_L_counts, device=scale.device, dtype=torch.long)
        return scale.repeat_interleave(counts, dim=L_dim)

    # ---------- Write cache ----------
    def cache(self):
        m = self.quant_method
        q_cur = self.quantized_item
        s_cur = self.scale          # compact — do NOT expand along L
        meta_cur = _clone_quant_meta(self.quant_meta)

        if m in ('VARQ', 'G_SCALE_HEAD_DIM', 'G_SCALE', 'G_TOKEN', 'G_TOKEN_HEAD', 'G_TENSOR'):
            self._append_cached_item(q_cur)
            self._append_cached_scale(s_cur)
            if meta_cur is not None:
                self.cached_quant_meta = _clone_quant_meta(meta_cur)
            if m in self._COMPACT_SCALE_METHODS:
                self._scale_L_counts.extend(self._cur_scale_L_counts or [q_cur.size(self.dim_cat)])
                if self.cached_quant_meta is not None:
                    self.cached_quant_meta["group_lengths"] = list(self._scale_L_counts)
        elif m in ('G_HEAD_DIM', 'G_TENSOR'):
            self._reset_cache_buffers()
            self._append_cached_item(q_cur)
            self._append_cached_scale(s_cur)
            self.cached_quant_meta = meta_cur
        else:
            raise ValueError(f"[VAR-Q]: Invalid quantization method: {m}")

    # ---------- Dequantization ----------
    def dequant_current(self) -> torch.Tensor:
        if self.pack_to_int32:
            assert self._pack_meta is not None, "pack meta is missing"
            q_int8 = self._unpack_last_dim_from_int32(self.quantized_item, self._pack_meta)
        else:
            q_int8 = self.quantized_item
        scale = self.scale
        quant_meta = _clone_quant_meta(self.quant_meta)
        if self.quant_method in self._COMPACT_SCALE_METHODS and scale.size(self.dim_cat) < q_int8.size(self.dim_cat):
            quant_meta = quant_meta or {"scheme": "COMPACT_SCALE", "qkv_format": self.qkv_format}
            quant_meta["group_lengths"] = list(self._cur_scale_L_counts or [q_int8.size(self.dim_cat)])
        return self._write_dequant_into_workspace(q_int8, scale, quant_meta)

    def dequant_all(self) -> torch.Tensor:
        if self.cached_item is None or self.cached_len == 0:
            return self.dequant_current()

        scale = self._valid_cached_scale()
        q_cached = self._valid_cached_item()

        if self.pack_to_int32:
            assert self._pack_meta is not None, "pack meta is missing"
            q_int8 = self._unpack_last_dim_from_int32(q_cached, self._pack_meta)
        else:
            q_int8 = q_cached
        quant_meta = _clone_quant_meta(self.cached_quant_meta)
        if self.quant_method in self._COMPACT_SCALE_METHODS and scale.size(self.dim_cat) < q_int8.size(self.dim_cat):
            quant_meta = quant_meta or {"scheme": "COMPACT_SCALE", "qkv_format": self.qkv_format}
            quant_meta["group_lengths"] = list(self._scale_L_counts)
        return self._write_dequant_into_workspace(q_int8, scale, quant_meta)

    # Main external interface
    def quant_and_cache(self, item: torch.Tensor):
        self.quant(item)
        self.cache()

    def use_var_q(self, item: torch.Tensor, cache_current: bool = True):
        if not cache_current:
            self.quant(item)
            if self.cached_item is None or self.cached_len == 0:
                result = self.dequant_current()
            else:
                q_cached = self._valid_cached_item()
                if self.pack_to_int32:
                    assert self._pack_meta is not None, "pack meta is missing"
                    q_cached = self._unpack_last_dim_from_int32(q_cached, self._pack_meta)
                    q_current = self._unpack_last_dim_from_int32(self.quantized_item, self._pack_meta)
                else:
                    q_current = self.quantized_item
                q_int8 = torch.cat([q_cached, q_current], dim=self.dim_cat)

                cached_scale = self._valid_cached_scale()
                scale = torch.cat([cached_scale, self.scale.to(self.scale_dtype)], dim=self.dim_cat)
                quant_meta = _clone_quant_meta(self.cached_quant_meta)
                if self.quant_method in self._COMPACT_SCALE_METHODS:
                    quant_meta = quant_meta or {"scheme": "COMPACT_SCALE", "qkv_format": self.qkv_format}
                    quant_meta["group_lengths"] = list(self._scale_L_counts) + list(self._cur_scale_L_counts)
                result = self._write_dequant_into_workspace(q_int8, scale, quant_meta)
            if self.debug:
                self.log_memory_breakdown(prefix=f"[VAR-Q:{self.kv_role}] ")
            return result
        self.quant_and_cache(item)
        result = self.dequant_all()
        if self.debug:
            self.log_memory_breakdown(prefix=f"[VAR-Q:{self.kv_role}] ")
        return result

    @staticmethod
    def _tensor_bytes(tensor: Optional[torch.Tensor], valid_len: Optional[int] = None, dim: int = 0) -> int:
        if tensor is None:
            return 0
        if valid_len is None:
            return int(tensor.numel() * tensor.element_size())
        shape = list(tensor.shape)
        shape[dim] = min(int(valid_len), int(tensor.size(dim)))
        numel = 1
        for size in shape:
            numel *= int(size)
        return int(numel * tensor.element_size())

    def memory_breakdown(self) -> Dict[str, int]:
        packed_kv_bytes = self._tensor_bytes(self.cached_item, self.cached_len, self.dim_cat)
        scale_bytes = self._tensor_bytes(self.cached_scale, self.cached_scale_len, self.dim_cat)
        dequant_workspace_bytes = self._tensor_bytes(self._dequant_workspace)
        stats = {
            "packed_kv_bytes": packed_kv_bytes,
            "scale_bytes": scale_bytes,
            "dequant_workspace_bytes": dequant_workspace_bytes,
            "dequant_workspace_peak_bytes": int(self._dequant_workspace_peak_bytes),
            "packed_cache_allocated_bytes": self._tensor_bytes(self.cached_item),
            "scale_cache_allocated_bytes": self._tensor_bytes(self.cached_scale),
        }
        if torch.cuda.is_available():
            stats.update(
                {
                    "cuda_memory_allocated": int(torch.cuda.memory_allocated()),
                    "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                    "cuda_memory_reserved": int(torch.cuda.memory_reserved()),
                    "cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved()),
                }
            )
        else:
            stats.update(
                {
                    "cuda_memory_allocated": 0,
                    "cuda_max_memory_allocated": 0,
                    "cuda_memory_reserved": 0,
                    "cuda_max_memory_reserved": 0,
                }
            )
        return stats

    def log_memory_breakdown(self, prefix: str = "") -> Dict[str, int]:
        stats = self.memory_breakdown()
        print(
            prefix
            + "packed_kv_bytes={packed_kv_bytes} scale_bytes={scale_bytes} "
            + "dequant_workspace_bytes={dequant_workspace_bytes} "
            + "dequant_workspace_peak_bytes={dequant_workspace_peak_bytes} "
            + "cuda_memory_allocated={cuda_memory_allocated} "
            + "cuda_max_memory_allocated={cuda_max_memory_allocated} "
            + "cuda_memory_reserved={cuda_memory_reserved} "
            + "cuda_max_memory_reserved={cuda_max_memory_reserved}"
        .format(**stats))
        return stats

    def rescale_qk(self, q: torch.Tensor, k: torch.Tensor, return_theta: bool = False):
        theta = None
        if self.rescale_qk_enabled:
            # q/k shape: (B, H, L, c), compute one rescale factor per head.
            # range_q = q.abs().amax(dim=(0, 2, 3), keepdim=True)
            # range_k = k.abs().amax(dim=(0, 2, 3), keepdim=True)
            range_q = q.abs().max()
            range_k = k.abs().max()
            theta = torch.sqrt((range_k + 1e-6) / (range_q + 1e-6))
            q = q * theta
            k = k / theta
        else:
            pass # do nothing
        if return_theta:
            return q, k, theta
        return q, k


SUPPORTED_QUANT_BITS = (2, 3, 4, 6, 8)
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


def validate_quantization_args(
    quant_bits: int,
    qkv_format: str,
    quant_method: str,
    pack_to_int32: bool,
    kv_role: str = "k",
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
    max_scale_seq_len: Optional[int] = None,
) -> None:
    normalized_method = normalize_quant_method(quant_method)
    if pack_to_int32:
        if quant_bits not in SUPPORTED_QUANT_BITS:
            raise ValueError(
                f"Unsupported quant_bits={quant_bits} with pack_to_int32=True. "
                f"Pack-supported values: {SUPPORTED_QUANT_BITS}"
            )
    else:
        if not (1 < int(quant_bits) <= 8):
            raise ValueError(
                f"Unsupported quant_bits={quant_bits}. Without pack_to_int32, "
                f"any integer in (1, 8] is allowed."
            )
    if qkv_format not in SUPPORTED_QKV_FORMATS:
        raise ValueError(
            f"Unsupported qkv_format={qkv_format}. Supported values: {SUPPORTED_QKV_FORMATS}"
        )
    if normalized_method not in CANONICAL_QUANT_METHODS:
        raise ValueError(
            f"Unsupported quant_method={quant_method}. Supported values: {SUPPORTED_QUANT_METHODS}"
        )
    _normalize_kv_role(kv_role)
    if not (0.0 < float(compression_ratio)):
        raise ValueError(f"compression_ratio must be positive, got {compression_ratio}")
    if max_scale_seq_len is not None and int(max_scale_seq_len) <= 0:
        raise ValueError(f"max_scale_seq_len must be positive when provided, got {max_scale_seq_len}")


def quantize_tensor(
    item: torch.Tensor,
    quant_bits: int,
    qkv_format: str,
    quant_method: str,
    pack_to_int32: bool,
    dequant_dtype: str | torch.dtype,
    kv_role: str = "k",
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
    max_scale_seq_len: Optional[int] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    validate_quantization_args(
        quant_bits=quant_bits,
        qkv_format=qkv_format,
        quant_method=quant_method,
        pack_to_int32=pack_to_int32,
        kv_role=kv_role,
        compression_ratio=compression_ratio,
        max_scale_seq_len=max_scale_seq_len,
    )
    quantizer = VAR_Q(
        quant_bits=quant_bits,
        qkv_format=qkv_format,
        quant_method=quant_method,
        kv_role=kv_role,
        compression_ratio=compression_ratio,
        max_scale_seq_len=max_scale_seq_len,
        pack_to_int32=pack_to_int32,
        dequant_dtype=_dequant_dtype_name(dequant_dtype),
    )
    quantizer.quant(item)
    return {
        "packed": quantizer.quantized_item,
        "scale": quantizer.scale,
        "pack_meta": quantizer._pack_meta if pack_to_int32 else None,
        "quant_meta": _clone_quant_meta(quantizer.quant_meta),
    }


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
            raise ValueError(
                f"Expected packed int32 tensor when pack_meta is present, got {packed.dtype}"
            )
        bits = int(pack_meta["bits"])
        if _HAS_TRITON and packed.is_cuda and bits in TRITON_PACK_BITS:
            q_int8 = unpack_last_dim_from_int32_triton(packed, pack_meta)
        else:
            q_int8 = unpack_last_dim_from_int32_python(packed, pack_meta)
    else:
        q_int8 = packed
    if _is_compact_scale_meta(quant_meta):
        scale = _expand_compact_scale_for_shape(scale, quant_meta, q_int8.shape)
    return (q_int8.to(torch.float32) * scale).to(target_dtype)


def build_kv_cache_quantizer(
    quant_bits: int,
    qkv_format: str,
    quant_method: str,
    kv_role: str = "k",
    blk_idx: int = 0,
    pack_to_int32: bool = True,
    eps: float = 1e-12,
    debug: bool = False,
    rescale_qk: bool = False,
    dequant_dtype: str = "bf16",
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
    max_scale_seq_len: Optional[int] = None,
    **_ignored: Any,
) -> "VAR_Q":
    validate_quantization_args(
        quant_bits=quant_bits,
        qkv_format=qkv_format,
        quant_method=quant_method,
        pack_to_int32=pack_to_int32,
        kv_role=kv_role,
        compression_ratio=compression_ratio,
        max_scale_seq_len=max_scale_seq_len,
    )
    return VAR_Q(
        quant_bits=quant_bits,
        qkv_format=qkv_format,
        quant_method=quant_method,
        kv_role=kv_role,
        compression_ratio=compression_ratio,
        max_scale_seq_len=max_scale_seq_len,
        blk_idx=blk_idx,
        pack_to_int32=pack_to_int32,
        eps=eps,
        debug=debug,
        rescale_qk=rescale_qk,
        dequant_dtype=dequant_dtype,
    )


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
        quant_method: str = DEFAULT_QUANT_METHOD,
        kv_role: str = "k",
        pack_to_int32: bool = True,
        eps: float = 1e-12,
        debug: bool = False,
        rescale_qk: bool = False,
        dequant_dtype: str = "bf16",
        compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
        max_scale_seq_len: Optional[int] = None,
    ):
        self.quant_bits = quant_bits
        self.qkv_format = qkv_format
        self.quant_method = quant_method
        self.kv_role = _normalize_kv_role(kv_role)
        self.pack_to_int32 = pack_to_int32
        self.eps = eps
        self.debug = debug
        self.rescale_qk = rescale_qk
        self.dequant_dtype = dequant_dtype
        self.compression_ratio = float(compression_ratio)
        self.max_scale_seq_len = int(max_scale_seq_len) if max_scale_seq_len is not None else None

        self._scale_quantizers: Dict[ScaleId, VAR_Q] = {}

    def _new_quantizer(self) -> VAR_Q:
        return VAR_Q(
            quant_bits=self.quant_bits,
            qkv_format=self.qkv_format,
            quant_method=self.quant_method,
            kv_role=self.kv_role,
            pack_to_int32=self.pack_to_int32,
            eps=self.eps,
            debug=self.debug,
            rescale_qk=self.rescale_qk,
            dequant_dtype=self.dequant_dtype,
            compression_ratio=self.compression_ratio,
            max_scale_seq_len=self.max_scale_seq_len,
        )

    def _ensure_quantizer(self, scale_id: ScaleId) -> VAR_Q:
        if scale_id not in self._scale_quantizers:
            self._scale_quantizers[scale_id] = self._new_quantizer()
        return self._scale_quantizers[scale_id]

    @staticmethod
    def _reset_quantizer_cache(q: VAR_Q) -> None:
        q._reset_cache_buffers()
        q.quantized_item = None
        q.scale = None
        q.quant_meta = None
        q._pack_meta = None
        q._cur_scale_L_counts = []

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
        return q is not None and q.cached_item is not None and q.cached_len > 0

    def get_scale(self, scale_id: ScaleId) -> torch.Tensor:
        q = self._scale_quantizers.get(scale_id)
        if q is None or q.cached_item is None or q.cached_len == 0:
            raise KeyError(f"scale_id={scale_id} is not cached")
        return q.dequant_all()

    def get_scale_quantized(self, scale_id: ScaleId):
        """Return raw cached quantized tensor, scale tensor and pack meta for fused kernels."""
        q = self._scale_quantizers.get(scale_id)
        if q is None or q.cached_item is None or q.cached_scale is None or q.cached_len == 0:
            raise KeyError(f"scale_id={scale_id} is not cached")
        return q._valid_cached_item(), q._valid_cached_scale(), q._pack_meta

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
        workspace_bytes = 0
        workspace_peak_bytes = 0
        for q in self._scale_quantizers.values():
            if q.cached_item is not None:
                packed_bytes += q.memory_breakdown()["packed_kv_bytes"]
            if q.cached_scale is not None:
                scale_bytes += q.memory_breakdown()["scale_bytes"]
            stats = q.memory_breakdown()
            workspace_bytes += stats["dequant_workspace_bytes"]
            workspace_peak_bytes += stats.get("dequant_workspace_peak_bytes", 0)
        return {
            "packed_bytes": packed_bytes,
            "scale_bytes": scale_bytes,
            "dequant_workspace_bytes": workspace_bytes,
            "dequant_workspace_peak_bytes": workspace_peak_bytes,
            "total_bytes": packed_bytes + scale_bytes + workspace_bytes,
        }

    def live_scale_ids(self) -> List[ScaleId]:
        return sorted(list(self._scale_quantizers.keys()), key=lambda x: str(x))


def build_infinitystar_cache_quantizer(
    quant_bits: int,
    qkv_format: str,
    quant_method: str,
    kv_role: str = "k",
    pack_to_int32: bool = True,
    eps: float = 1e-12,
    debug: bool = False,
    rescale_qk: bool = False,
    dequant_dtype: str = "bf16",
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
    max_scale_seq_len: Optional[int] = None,
    **_ignored: Any,
) -> InfinityStarVARQ:
    validate_quantization_args(
        quant_bits=quant_bits,
        qkv_format=qkv_format,
        quant_method=quant_method,
        pack_to_int32=pack_to_int32,
        kv_role=kv_role,
        compression_ratio=compression_ratio,
        max_scale_seq_len=max_scale_seq_len,
    )
    return InfinityStarVARQ(
        quant_bits=quant_bits,
        qkv_format=qkv_format,
        quant_method=quant_method,
        kv_role=kv_role,
        pack_to_int32=pack_to_int32,
        eps=eps,
        debug=debug,
        rescale_qk=rescale_qk,
        dequant_dtype=dequant_dtype,
        compression_ratio=compression_ratio,
        max_scale_seq_len=max_scale_seq_len,
    )


HAS_TRITON_PACK = _HAS_TRITON

if __name__ == "__main__":
    import time, torch
    from pack_unpack import (
        pack_last_dim_to_int32_triton, unpack_last_dim_from_int32_triton,
        pack_last_dim_to_int32_python, unpack_last_dim_from_int32_python
    )

    def bench_one(bits, runs=50, warmup=10):
        B,L,H,c = 100,680,20,64
        x = torch.randint(-(1<<(bits-1)), (1<<(bits-1)), (B,L,H,c),
                        dtype=torch.int8, device='cuda')

        # --- Triton: recompile & warmup---
        for _ in range(warmup):
            p, meta = pack_last_dim_to_int32_triton(x, bits)
            xr = unpack_last_dim_from_int32_triton(p, meta)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(runs):
            p, meta = pack_last_dim_to_int32_triton(x, bits)
            xr = unpack_last_dim_from_int32_triton(p, meta)
        torch.cuda.synchronize()
        t1 = time.time()
        triton_ms = (t1 - t0) * 1000 / runs

        # --- PyTorch fallback---
        for _ in range(warmup):
            p, meta = pack_last_dim_to_int32_python(x, bits)
            xr = unpack_last_dim_from_int32_python(p, meta)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(runs):
            p, meta = pack_last_dim_to_int32_python(x, bits)
            xr = unpack_last_dim_from_int32_python(p, meta)
        torch.cuda.synchronize()
        t1 = time.time()
        torch_ms = (t1 - t0) * 1000 / runs

        print(f"{bits}bit  Triton: {triton_ms:.3f} ms/iter   PyTorch: {torch_ms:.3f} ms/iter")

    for b in (2,4,8):
        bench_one(b)

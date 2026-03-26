import torch
from typing import Optional, Tuple, Dict
try:
    from VAR_Q.pack_unpack import pack_last_dim_to_int32_triton, unpack_last_dim_from_int32_triton
    _HAS_TRITON = True
except Exception:
    from VAR_Q.pack_unpack import pack_last_dim_to_int32_python, unpack_last_dim_from_int32_python
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

class VAR_Q:
    def __init__(
        self,
        quant_bits: int = 8,
        qkv_format: str = 'BLHc',  # (B,L,H,c) or (B,H,L,c)
        quant_method: str = 'G_SCALE_HEAD_DIM',  # ['G_TENSOR','G_SCALE_HEAD_DIM','G_HEAD_DIM','G_SCALE','G_TOKEN','G_TOKEN_HEAD']
        blk_idx: int = 0,
        pack_to_int32: bool = True,
        eps: float = 1e-12,
        debug: bool = False,
        rescale_qk = False,
        dequant_dtype: str = "bf16",
        outlier_ratio: float = 0.0,
        outlier_mode: str = 'ratio',
        outlier_n_sigma: float = 3.0,
    ):
        assert qkv_format in ('BLHc','BHLc'), f"Invalid qkv_format: {qkv_format}"
        assert quant_method in ('G_TENSOR','G_SCALE_HEAD_DIM','G_HEAD_DIM','G_SCALE','G_TOKEN','G_TOKEN_HEAD'), \
            f"Invalid quant_method: {quant_method}"
        assert 1 < quant_bits <= 8, "Only support (2..8] bits (common 2/4/8 bits); bit-pack implementation covers 2/4/8 bits"
        assert outlier_mode in ('ratio', 'sigma'), f"Invalid outlier_mode: {outlier_mode}"
        self.quant_bits = quant_bits
        self.quant_method = quant_method
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

        self._qkv_format = None
        self.set_qkv_format(qkv_format)

        self.bound_min = -(2 ** (quant_bits - 1))
        self.bound_max =  (2 ** (quant_bits - 1)) - 1

        # Cache: maybe int8 or packed int32; scale is fp32 (broadcastable)
        self.cached_item: Optional[torch.Tensor] = None
        self.cached_scale: Optional[torch.Tensor] = None

        # Current step
        self.quantized_item: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None

        # pack/unpack meta (assuming c is constant across steps, so meta is constant)
        self._pack_meta: Optional[Dict[str,int]] = None  # {'orig_c','vals_per_word','pad_len','bits'}

        # Outlier preservation (KVQuant-style sparse storage)
        self.outlier_ratio = outlier_ratio
        self.outlier_mode = outlier_mode
        self.outlier_n_sigma = outlier_n_sigma
        self._outlier_enabled = outlier_ratio > 0.0
        # Current step outlier
        self._cur_outlier_indices: Optional[torch.Tensor] = None   # (N, 4) int32
        self._cur_outlier_values: Optional[torch.Tensor] = None    # (N,) bf16
        # Cached outlier across all steps
        self.outlier_cached_indices: Optional[torch.Tensor] = None # (M, 4) int32
        self.outlier_cached_values: Optional[torch.Tensor] = None  # (M,) bf16
        self._L_offset: int = 0

        # Per-step token counts for compact-scale methods (G_SCALE_HEAD_DIM, G_SCALE, G_TENSOR).
        # These methods produce a single scale vector (L-dim=1) per step; we store it compactly
        # and use repeat_interleave to reconstruct full-L scale only during dequantization.
        self._scale_L_counts: list = []

    def set_qkv_format(self, fmt: str):
        """Update qkv_format and synchronize dim_cat / dim_map accordingly."""
        if fmt == self._qkv_format:
            return
        self._qkv_format = fmt
        if fmt == 'BLHc':
            self.dim_cat = 1
            self.dim_map = {
                'G_TENSOR':        (0,1,2,3),
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
        return scale

    def _quantize_to_int8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)
        q = torch.round(x32 / scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
        return q

    def _dequantize_from_int8(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (q.to(torch.float32) * scale).to(self.dequant_dtype)
    
    # ---------- Pack/unpack last dimension to/from int32 ----------
    def _pack_last_dim_to_int32(self, q_int8: torch.Tensor, bits: int) -> Tuple[torch.Tensor, Dict[str,int]]:
        if _HAS_TRITON:
            return pack_last_dim_to_int32_triton(q_int8, bits)
        else:
            return pack_last_dim_to_int32_python(q_int8, bits)

    def _unpack_last_dim_from_int32(self, packed: torch.Tensor, meta: Dict[str,int]) -> torch.Tensor:
        if _HAS_TRITON:
            return unpack_last_dim_from_int32_triton(packed, meta)
        else:
            return unpack_last_dim_from_int32_python(packed, meta)

    # ---------- Outlier extraction (KVQuant-style) ----------
    def _extract_outliers(self, item: torch.Tensor) -> torch.Tensor:
        """Extract outliers from item, store them sparsely, zero-out in-place copy. Returns modified item."""
        if not self._outlier_enabled:
            self._cur_outlier_indices = None
            self._cur_outlier_values = None
            return item

        abs_vals = item.abs().float()

        if self.outlier_mode == 'ratio':
            numel = abs_vals.numel()
            k = max(1, int(numel * self.outlier_ratio))
            threshold = abs_vals.reshape(-1).topk(k).values[-1]
            outlier_mask = abs_vals >= threshold
        else:  # 'sigma'
            mean_val = abs_vals.mean()
            std_val = abs_vals.std()
            outlier_mask = abs_vals >= (mean_val + self.outlier_n_sigma * std_val)

        if outlier_mask.any():
            indices = torch.nonzero(outlier_mask, as_tuple=False).to(torch.int32)  # (N, 4)
            item = item.clone()
            values = item[outlier_mask].to(torch.bfloat16)
            item[outlier_mask] = 0.0
            self._cur_outlier_indices = indices
            self._cur_outlier_values = values
        else:
            self._cur_outlier_indices = None
            self._cur_outlier_values = None

        return item

    def _scatter_outliers(self, result: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Write outlier values back into the dequantized tensor."""
        idx = indices.long().unbind(1)
        result[idx[0], idx[1], idx[2], idx[3]] = values.to(result.dtype)
        return result

    # ---------- Main quantization process ----------
    def quant(self, item: torch.Tensor):
        m = self.quant_method
        red = self.dim_map[m]

        # Extract and zero-out outliers before quantization
        item = self._extract_outliers(item)

        if m in ('G_SCALE_HEAD_DIM', 'G_TOKEN', 'G_TOKEN_HEAD', 'G_SCALE'):
            scale = self._compute_scale(item, red, keepdim=True)
            q = self._quantize_to_int8(item, scale)

        elif m in ('G_HEAD_DIM', 'G_TENSOR'):
            if self.cached_item is not None and self.cached_scale is not None:
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
    _COMPACT_SCALE_METHODS = frozenset(('G_SCALE_HEAD_DIM', 'G_SCALE', 'G_TENSOR'))

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

        if m in ('G_SCALE_HEAD_DIM', 'G_SCALE', 'G_TOKEN', 'G_TOKEN_HEAD', 'G_TENSOR'):
            if self.cached_item is None:
                self.cached_item, self.cached_scale = q_cur, s_cur
            else:
                self.cached_item  = torch.cat([self.cached_item,  q_cur], dim=self.dim_cat)
                self.cached_scale = torch.cat([self.cached_scale, s_cur], dim=self.dim_cat)
            if m in self._COMPACT_SCALE_METHODS:
                self._scale_L_counts.append(q_cur.size(self.dim_cat))
        elif m in ('G_HEAD_DIM', 'G_TENSOR'):
            self.cached_item, self.cached_scale = q_cur, s_cur
        else:
            raise ValueError(f"[VAR-Q]: Invalid quantization method: {m}")

        # Cache outlier indices and values (sparse)
        if self._outlier_enabled:
            self._cache_outliers(q_cur)

    def _cache_outliers(self, q_cur: torch.Tensor):
        """Append current-step outliers to the cached sparse storage with L-offset."""
        L_dim_idx = 1 if self.qkv_format == 'BLHc' else 2
        L_new = q_cur.size(L_dim_idx)

        if self._cur_outlier_indices is not None and self._cur_outlier_indices.numel() > 0:
            shifted = self._cur_outlier_indices.clone()
            shifted[:, L_dim_idx] += self._L_offset
            if self.outlier_cached_indices is None:
                self.outlier_cached_indices = shifted
                self.outlier_cached_values = self._cur_outlier_values
            else:
                self.outlier_cached_indices = torch.cat([self.outlier_cached_indices, shifted], dim=0)
                self.outlier_cached_values = torch.cat([self.outlier_cached_values, self._cur_outlier_values], dim=0)

        if self.quant_method not in ('G_HEAD_DIM',):
            self._L_offset += L_new
        else:
            self._L_offset = 0
            self.outlier_cached_indices = None
            self.outlier_cached_values = None

    # ---------- Dequantization ----------
    def dequant_current(self) -> torch.Tensor:
        if self.pack_to_int32:
            assert self._pack_meta is not None, "pack meta is missing"
            q_int8 = self._unpack_last_dim_from_int32(self.quantized_item, self._pack_meta)
            result = self._dequantize_from_int8(q_int8, self.scale)
        else:
            result = self._dequantize_from_int8(self.quantized_item, self.scale)
        if self._cur_outlier_indices is not None:
            result = self._scatter_outliers(result, self._cur_outlier_indices, self._cur_outlier_values)
        return result

    def dequant_all(self) -> torch.Tensor:
        if self.cached_item is None:
            return self.dequant_current()

        scale = self.cached_scale
        if self.quant_method in self._COMPACT_SCALE_METHODS:
            scale = self._reconstruct_scale_for_L(scale, self.cached_item.size(self.dim_cat))

        if self.pack_to_int32:
            assert self._pack_meta is not None, "pack meta is missing"
            q_int8 = self._unpack_last_dim_from_int32(self.cached_item, self._pack_meta)
            result = self._dequantize_from_int8(q_int8, scale)
        else:
            result = self._dequantize_from_int8(self.cached_item, scale)
        if self.outlier_cached_indices is not None:
            result = self._scatter_outliers(result, self.outlier_cached_indices, self.outlier_cached_values)
        return result

    # Main external interface
    def quant_and_cache(self, item: torch.Tensor):
        self.quant(item)
        self.cache()

    def use_var_q(self, item: torch.Tensor):
        self.quant_and_cache(item)
        return self.dequant_all()

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
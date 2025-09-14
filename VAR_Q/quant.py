import torch
from typing import Optional, Tuple, Dict

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
  This yields `L × H` groups (e.g., `2240 × 20` for VAR), providing the finest granularity among these strategies.  

These grouping strategies allow us to explore different quantization granularities for Q/K/V tensors in AR-based image generation models.

"""

class VAR_Q:
    def __init__(
        self,
        quant_bits: int = 8,
        qkv_format: str = 'BLHc',  # (B,L,H,c) or (B,H,L,c)
        quant_method: str = 'G_SCALE_HEAD_DIM',  # ['G_TENSOR','G_SCALE_HEAD_DIM','G_HEAD_DIM','G_SCALE','G_TOKEN','G_TOKEN_HEAD']
        blk_idx: int = 0,
        pack_to_int32: bool = False,
        eps: float = 1e-12,
        debug: bool = False
    ):
        assert qkv_format in ('BLHc','BHLc'), f"Invalid qkv_format: {qkv_format}"
        assert quant_method in ('G_TENSOR','G_SCALE_HEAD_DIM','G_HEAD_DIM','G_SCALE','G_TOKEN','G_TOKEN_HEAD'), \
            f"Invalid quant_method: {quant_method}"
        assert 1 < quant_bits <= 8, "Only support (2..8] bits (common 2/4/8 bits); bit-pack implementation covers 2/4/8 bits"
        self.quant_bits = quant_bits
        self.quant_method = quant_method
        self.cur_blk_idx = blk_idx
        self.qkv_format = qkv_format
        self.pack_to_int32 = pack_to_int32
        self.eps = eps
        self.debug = debug

        if qkv_format == 'BLHc':
            self.dim_cat = 1
            # These are the dimensions that are reduced, the remaining ones are the grouping dimensions
            self.dim_map = {
                'G_TENSOR':        (0,1,2,3),      # Global
                'G_SCALE_HEAD_DIM':(0,1),          # Keep H,c
                'G_HEAD_DIM':      (0,1),          # Keep H,c
                'G_TOKEN':         (0,2,3),        # Keep L
                'G_TOKEN_HEAD':    (0,3),          # Keep L,H
                'G_SCALE':         (0,1,2,3),      # Every step has a scale
            }
        else:  # 'BHLc'
            self.dim_cat = 2
            self.dim_map = {
                'G_TENSOR':        (0,1,2,3),
                'G_SCALE_HEAD_DIM':(0,2),          # Keep H,c
                'G_HEAD_DIM':      (0,2),          # Keep H,c
                'G_TOKEN':         (0,1,3),        # Keep L
                'G_TOKEN_HEAD':    (0,3),          # Keep H,L
                'G_SCALE':         (0,1,2,3),
            }

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
        return (q.to(torch.float32) * scale).to(torch.float32)

    # ---------- signed<->unsigned width representation (for bit-pack) ----------
    @staticmethod
    def _signed_to_unsigned_width(x_int8: torch.Tensor, bits: int) -> torch.Tensor:
        mask = (1 << bits) - 1
        return x_int8.to(torch.int32) & mask

    @staticmethod
    def _unsigned_to_signed_width(u: torch.Tensor, bits: int) -> torch.Tensor:
        mask = (1 << bits) - 1
        sign = 1 << (bits - 1)
        u = u.to(torch.int32) & mask
        x = torch.where((u & sign) != 0, u - (1 << bits), u)
        return x.to(torch.int8)

    # ---------- Pack the last dimension (c) to int32 ----------
    def _pack_last_dim_to_int32(self, q_int8: torch.Tensor, bits: int) -> Tuple[torch.Tensor, Dict[str,int]]:
        assert bits in (2,4,8), "bit-pack only implements 2/4/8 bits"
        vals = 32 // bits
        *lead, C = q_int8.shape
        pad_len = (-C) % vals
        if pad_len:
            pad = torch.zeros((*lead, pad_len), dtype=q_int8.dtype, device=q_int8.device)
            q_int8 = torch.cat([q_int8, pad], dim=-1)
            C_padded = C + pad_len
        else:
            C_padded = C

        new_last = C_padded // vals
        q_u = self._signed_to_unsigned_width(q_int8, bits)  # -> int32 non-negative
        q_u = q_u.view(*lead, new_last, vals).to(torch.int32)

        shifts = (torch.arange(vals, device=q_int8.device, dtype=torch.int32) * bits).view(
            *([1] * (q_u.dim() - 1)), vals
        )
        packed = (q_u << shifts).sum(dim=-1).to(torch.int32)  # [..., new_last]

        meta = {'orig_c': C, 'vals_per_word': vals, 'pad_len': pad_len, 'bits': bits}
        return packed, meta

    def _unpack_last_dim_from_int32(self, packed: torch.Tensor, meta: Dict[str,int]) -> torch.Tensor:
        bits = meta['bits']
        vals = meta['vals_per_word']
        pad_len = meta['pad_len']
        orig_c = meta['orig_c']
        mask = (1 << bits) - 1

        shifts = torch.arange(vals, device=packed.device, dtype=torch.int32) * bits
        pieces_u = [(packed >> s) & mask for s in shifts]
        u_stack = torch.stack(pieces_u, dim=-1)  # [..., vals]

        unpacked_u = u_stack.reshape(*packed.shape[:-1], packed.shape[-1] * vals)  # [..., C_padded]
        if pad_len:
            unpacked_u = unpacked_u[..., :orig_c]

        q_int8 = self._unsigned_to_signed_width(unpacked_u, bits)
        return q_int8

    # ---------- Main quantization process ----------
    def quant(self, item: torch.Tensor):
        m = self.quant_method
        red = self.dim_map[m]

        if m in ('G_SCALE_HEAD_DIM', 'G_TOKEN', 'G_TOKEN_HEAD', 'G_SCALE'):
            # Group quantization (e.g. [1,1,H,c] / [1,L,1,1] / [1,L,H,1])
            scale = self._compute_scale(item, red, keepdim=True)
            q = self._quantize_to_int8(item, scale)

        elif m in ('G_HEAD_DIM', 'G_TENSOR'):
            # If there is a cache: first dequantize the cache, concatenate with the current step, 
            # and then quantize the entire result along the head and dimension axes
            if self.cached_item is not None and self.cached_scale is not None:
                cached_deq = self.dequant_all()
                cat = torch.cat([cached_deq, item], dim=self.dim_cat)
                scale = self._compute_scale(cat, red, keepdim=True)   # reduce(B,L), keep(H,c)
                q = self._quantize_to_int8(cat, scale)
            else:
                scale = self._compute_scale(item, red, keepdim=True)
                q = self._quantize_to_int8(item, scale)
        else:
            raise ValueError(f"[VAR-Q]: Invalid quantization method: {m}")

        # (Optional) Pack the last dimension (c) to int32
        if self.pack_to_int32:
            q_packed, meta = self._pack_last_dim_to_int32(q, self.quant_bits)
            if self._pack_meta is not None:
                # Protection: if they don't match, it means the upstream shape has changed (which should rarely happen)
                assert meta['bits'] == self._pack_meta['bits']
                assert meta['vals_per_word'] == self._pack_meta['vals_per_word']
                assert meta['orig_c'] == self._pack_meta['orig_c']
                assert meta['pad_len'] == self._pack_meta['pad_len']
            self._pack_meta = meta
            self.quantized_item = q_packed
        else:
            self.quantized_item = q

        self.scale = scale

    # ---------- Align scale to L dimension ----------
    def _expand_scale_like_L(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Only expand when the L dimension of scale is 1; the other dimensions rely on broadcasting
        if self.qkv_format == 'BLHc':
            if scale.size(1) == 1 and q.size(1) > 1:
                scale = scale.expand(-1, q.size(1), -1, -1)
        else:  # 'BHLc'
            if scale.size(2) == 1 and q.size(2) > 1:
                scale = scale.expand(-1, -1, q.size(2), -1)
        return scale

    # ---------- Write cache ----------
    def cache(self):
        m = self.quant_method
        q_cur = self.quantized_item
        s_cur = self._expand_scale_like_L(q_cur, self.scale)

        if m in ('G_SCALE_HEAD_DIM', 'G_SCALE', 'G_TOKEN', 'G_TOKEN_HEAD', 'G_TENSOR'):
            if self.cached_item is None:
                self.cached_item, self.cached_scale = q_cur, s_cur
            else:
                self.cached_item  = torch.cat([self.cached_item,  q_cur], dim=self.dim_cat)
                self.cached_scale = torch.cat([self.cached_scale, s_cur], dim=self.dim_cat)
        elif m in ('G_HEAD_DIM', 'G_TENSOR'):
            self.cached_item, self.cached_scale = q_cur, s_cur
        else:
            raise ValueError(f"[VAR-Q]: Invalid quantization method: {m}")

    # ---------- Dequantization ----------
    def dequant_current(self) -> torch.Tensor:
        if self.pack_to_int32:
            assert self._pack_meta is not None, "pack meta is missing"
            q_int8 = self._unpack_last_dim_from_int32(self.quantized_item, self._pack_meta)
            return self._dequantize_from_int8(q_int8, self.scale)
        else:
            return self._dequantize_from_int8(self.quantized_item, self.scale)

    def dequant_all(self) -> torch.Tensor:
        if self.cached_item is None:
            return self.dequant_current()
        if self.pack_to_int32:
            assert self._pack_meta is not None, "pack meta is missing"
            q_int8 = self._unpack_last_dim_from_int32(self.cached_item, self._pack_meta)
            return self._dequantize_from_int8(q_int8, self.cached_scale)
        else:
            return self._dequantize_from_int8(self.cached_item, self.cached_scale)

    # Main external interface
    def use_var_q(self, item: torch.Tensor):
        self.quant(item)
        self.cache()
        return self.dequant_all()
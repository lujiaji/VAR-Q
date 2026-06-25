# VAR_Q/pack_unpack.py
import torch
from typing import Tuple, Dict, Optional

# ===== Optional Triton =====
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False
    
TRITON_PACK_BITS = (2, 3, 4, 6, 8)

# ===========================
# Triton kernels (vectorized)
# ===========================
if _HAS_TRITON:
    # --- Triton kernels (2D tiling: BLOCK_ROWS x BLOCK_WORDS) ---
    @triton.jit
    def _pack2d_kernel(
        q_ptr, out_ptr,
        N_ROWS, C, C_OUT,
        BITS: tl.constexpr,          # 2/4/8
        VALS: tl.constexpr,          # 32//BITS
        BLOCK_VALS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,    # tile height  (rows per program)
        BLOCK_WORDS: tl.constexpr,   # tile width   (int32 words per program)
    ):
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)

        row_idx   = pid_r * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)         # [BR]
        word_idx  = pid_c * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)       # [BW]
        mask_row  = row_idx < N_ROWS
        mask_word = word_idx < C_OUT

        # indices
        offs      = tl.arange(0, BLOCK_VALS)                              # [BV]
        row_in_b  = row_idx[:, None, None] * C                            # [BR,1,1]
        row_out_b = row_idx[:, None] * C_OUT                              # [BR,1]
        base_cols = word_idx[None, :, None] * VALS + offs[None, None, :]  # [1,BW,VALS]

        mask_vals = offs[None, None, :] < VALS
        mask_cols = mask_row[:, None, None] & mask_word[None, :, None] & mask_vals & (base_cols < C)

        vals_i8 = tl.load(q_ptr + row_in_b + base_cols, mask=mask_cols, other=0).to(tl.int32)  # [BR,BW,VALS]
        mask_bits = (1 << BITS) - 1
        vals_u = vals_i8 & mask_bits

        shifts = (offs * BITS)[None, None, :]           # [1,1,VALS]
        shifted = vals_u << shifts                      # [BR,BW,VALS]
        acc = tl.sum(shifted, axis=2)                   # [BR,BW] int32

        tl.store(out_ptr + row_out_b + word_idx[None, :], acc, mask=mask_row[:, None] & mask_word[None, :])


    @triton.jit
    def _unpack2d_kernel(
        packed_ptr, out_ptr,
        N_ROWS, C, C_OUT,
        BITS: tl.constexpr,
        VALS: tl.constexpr,
        BLOCK_VALS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_WORDS: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)

        row_idx   = pid_r * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)         # [BR]
        word_idx  = pid_c * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)       # [BW]
        mask_row  = row_idx < N_ROWS
        mask_word = word_idx < C_OUT

        row_in_b  = row_idx[:, None] * C_OUT                               # [BR,1]
        row_out_b = row_idx[:, None, None] * C                             # [BR,1,1]

        words = tl.load(packed_ptr + row_in_b + word_idx[None, :], mask=mask_row[:, None] & mask_word[None, :], other=0)  # [BR,BW]

        offs = tl.arange(0, BLOCK_VALS)                                   # [BV]
        words_mat = words[:, :, None]                                     # [BR,BW,1]

        mask_bits = (1 << BITS) - 1
        sign_bit  = 1 << (BITS - 1)
        pieces_u  = (words_mat >> (offs[None, None, :] * BITS)) & mask_bits   # [BR,BW,VALS]
        pieces_s  = tl.where((pieces_u & sign_bit) != 0, pieces_u - (1 << BITS), pieces_u).to(tl.int8)

        # output columns for each small piece
        out_cols  = word_idx[None, :, None] * VALS + offs[None, None, :]       # [1,BW,VALS]
        mask_cols = mask_row[:, None, None] & mask_word[None, :, None] & (offs[None, None, :] < VALS) & (out_cols < C)

        tl.store(out_ptr + row_out_b + out_cols, pieces_s, mask=mask_cols)

    @triton.jit
    def _unpack_dequant2d_kernel(
        packed_ptr, scale_ptr, scale_group_ids_ptr, out_ptr,
        N_ROWS, C, C_OUT,
        B, L, H,
        OUT_S0, OUT_S1, OUT_S2, OUT_S3,
        SCALE_S0, SCALE_S1, SCALE_S2, SCALE_S3,
        SCALE_D0, SCALE_D1, SCALE_D2, SCALE_D3,
        BITS: tl.constexpr,
        VALS: tl.constexpr,
        BLOCK_VALS: tl.constexpr,
        QKV_FORMAT: tl.constexpr,
        USE_SCALE_GROUP_IDS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_WORDS: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)

        row_idx = pid_r * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        word_idx = pid_c * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
        mask_row = row_idx < N_ROWS
        mask_word = word_idx < C_OUT

        row_in_b = row_idx[:, None] * C_OUT
        words = tl.load(
            packed_ptr + row_in_b + word_idx[None, :],
            mask=mask_row[:, None] & mask_word[None, :],
            other=0,
        )

        offs = tl.arange(0, BLOCK_VALS)
        out_cols = word_idx[None, :, None] * VALS + offs[None, None, :]
        valid = mask_row[:, None, None] & mask_word[None, :, None] & (offs[None, None, :] < VALS) & (out_cols < C)

        mask_bits = (1 << BITS) - 1
        sign_bit = 1 << (BITS - 1)
        pieces_u = (words[:, :, None] >> (offs[None, None, :] * BITS)) & mask_bits
        pieces_s = tl.where((pieces_u & sign_bit) != 0, pieces_u - (1 << BITS), pieces_u).to(tl.float32)

        if QKV_FORMAT == 0:  # BLHc rows are ((b * L) + l) * H + h
            b_idx = row_idx // (L * H)
            rem = row_idx - b_idx * L * H
            l_idx = rem // H
            h_idx = rem - l_idx * H
        else:  # BHLc rows are ((b * H) + h) * L + l
            b_idx = row_idx // (H * L)
            rem = row_idx - b_idx * H * L
            h_idx = rem // L
            l_idx = rem - h_idx * L

        if QKV_FORMAT == 0:
            out_offsets = (
                b_idx[:, None, None] * OUT_S0
                + l_idx[:, None, None] * OUT_S1
                + h_idx[:, None, None] * OUT_S2
                + out_cols * OUT_S3
            )
        else:
            out_offsets = (
                b_idx[:, None, None] * OUT_S0
                + h_idx[:, None, None] * OUT_S1
                + l_idx[:, None, None] * OUT_S2
                + out_cols * OUT_S3
            )

        scale_l_idx = l_idx
        if USE_SCALE_GROUP_IDS:
            scale_l_idx = tl.load(scale_group_ids_ptr + l_idx, mask=mask_row, other=0)

        sb = tl.where(SCALE_D0 == 1, 0, b_idx)
        sl = tl.where(SCALE_D1 == 1, 0, scale_l_idx)
        sh = tl.where(SCALE_D2 == 1, 0, h_idx)
        sc = tl.where(SCALE_D3 == 1, 0, out_cols)
        scale_offsets = (
            sb[:, None, None] * SCALE_S0
            + sl[:, None, None] * SCALE_S1
            + sh[:, None, None] * SCALE_S2
            + sc * SCALE_S3
        )
        scale = tl.load(scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.float32)
        tl.store(out_ptr + out_offsets, pieces_s * scale, mask=valid)

    def _vals_per_word(bits: int) -> int:
        return 10 if bits == 3 else 32 // bits

    def _block_vals(vals: int) -> int:
        block_vals = 1
        while block_vals < vals:
            block_vals <<= 1
        return block_vals

    def _pick_tiles(N_ROWS: int, C_out: int):
        bw = 1
        while (bw << 1) <= C_out and bw < 128:
            bw <<= 1
        bw = max(bw, 4)
        br = 64 if N_ROWS >= 1<<18 else 32
        if bw <= 16:
            warps = 1
        elif bw <= 32:
            warps = 2
        else:
            warps = 4
        return br, bw, warps
    
# ==============
# Triton wrappers
# ==============
def pack_last_dim_to_int32_triton(q_int8: torch.Tensor, bits: int):
    assert _HAS_TRITON, "Triton is not available"
    assert q_int8.is_cuda, "Triton version requires CUDA tensor"
    assert q_int8.dtype == torch.int8
    assert bits in TRITON_PACK_BITS
    x = q_int8.contiguous()
    *lead, C = x.shape
    N_ROWS = int(x.numel() // C)
    VALS   = _vals_per_word(bits)
    BLOCK_VALS = _block_vals(VALS)
    C_out  = (C + VALS - 1) // VALS
    pad_len= C_out * VALS - C

    x2d = x.view(N_ROWS, C)
    out2d = torch.empty((N_ROWS, C_out), dtype=torch.int32, device=x.device)

    BR, BW, warps = _pick_tiles(N_ROWS, C_out)
    grid = (triton.cdiv(N_ROWS, BR), triton.cdiv(C_out, BW))

    _pack2d_kernel[grid](
        x2d, out2d,
        N_ROWS, C, C_out,
        BITS=bits, VALS=VALS, BLOCK_VALS=BLOCK_VALS,
        BLOCK_ROWS=BR, BLOCK_WORDS=BW,
        num_warps=warps, num_stages=2
    )
    packed = out2d.view(*lead, C_out)
    meta = {'orig_c': int(C), 'vals_per_word': int(VALS), 'pad_len': int(pad_len), 'bits': int(bits)}
    return packed, meta

def unpack_last_dim_from_int32_triton(packed: torch.Tensor, meta: dict) -> torch.Tensor:
    assert _HAS_TRITON, "Triton is not available"
    assert packed.is_cuda, "Triton version requires CUDA tensor"
    assert packed.dtype == torch.int32

    bits  = int(meta['bits'])
    assert bits in TRITON_PACK_BITS
    VALS  = int(meta['vals_per_word'])
    BLOCK_VALS = _block_vals(VALS)
    C_out = packed.shape[-1]
    orig_c= int(meta['orig_c'])
    y = packed.contiguous()
    *lead, C_out = y.shape
    N_ROWS = int(y.numel() // C_out)

    out2d = torch.empty((N_ROWS, orig_c), dtype=torch.int8, device=y.device)
    y2d   = y.view(N_ROWS, C_out)

    BR, BW, warps = _pick_tiles(N_ROWS, C_out)
    grid = (triton.cdiv(N_ROWS, BR), triton.cdiv(C_out, BW))

    _unpack2d_kernel[grid](
        y2d, out2d,
        N_ROWS, orig_c, C_out,
        BITS=bits, VALS=VALS, BLOCK_VALS=BLOCK_VALS,
        BLOCK_ROWS=BR, BLOCK_WORDS=BW,
        num_warps=warps, num_stages=2
    )
    return out2d.view(*lead, orig_c)

def unpack_dequant_last_dim_from_int32_triton(
    packed: torch.Tensor,
    scale: torch.Tensor,
    meta: dict,
    out: torch.Tensor,
    qkv_format: str,
    scale_group_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert _HAS_TRITON, "Triton is not available"
    assert packed.is_cuda and scale.is_cuda and out.is_cuda, "Triton version requires CUDA tensors"
    assert packed.dtype == torch.int32
    assert out.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert qkv_format in ("BLHc", "BHLc")

    bits = int(meta["bits"])
    assert bits in TRITON_PACK_BITS
    vals = int(meta["vals_per_word"])
    block_vals = _block_vals(vals)
    orig_c = int(meta["orig_c"])
    y = packed.contiguous()
    s = scale.contiguous()
    if out.shape[-1] != orig_c:
        raise ValueError(f"out last dimension must be {orig_c}, got {out.shape[-1]}")
    if out.shape[:-1] != y.shape[:-1]:
        raise ValueError(f"out leading shape must match packed leading shape, got {out.shape[:-1]} vs {y.shape[:-1]}")
    if out.ndim != 4:
        raise ValueError(f"fused dequant expects 4D KV tensors, got ndim={out.ndim}")

    if qkv_format == "BLHc":
        B, L, H, _ = out.shape
    else:
        B, H, L, _ = out.shape
    use_scale_group_ids = scale_group_ids is not None
    if use_scale_group_ids:
        assert scale_group_ids is not None
        if not scale_group_ids.is_cuda:
            raise ValueError("scale_group_ids must be a CUDA tensor")
        if scale_group_ids.device != y.device:
            raise ValueError(f"scale_group_ids must be on {y.device}, got {scale_group_ids.device}")
        if scale_group_ids.dtype != torch.int32:
            raise ValueError(f"scale_group_ids must have dtype torch.int32, got {scale_group_ids.dtype}")
        if scale_group_ids.ndim != 1:
            raise ValueError(f"scale_group_ids must be 1D, got ndim={scale_group_ids.ndim}")
        if int(scale_group_ids.numel()) != int(L):
            raise ValueError(f"scale_group_ids length must be {int(L)}, got {int(scale_group_ids.numel())}")
        scale_group_ids_arg = scale_group_ids.contiguous()
    else:
        scale_group_ids_arg = y
    C_out = y.shape[-1]
    n_rows = int(y.numel() // C_out)
    y2d = y.view(n_rows, C_out)

    scale_shape = tuple(int(v) for v in s.shape)
    if len(scale_shape) != 4:
        raise ValueError(f"scale must be 4D, got shape={scale_shape}")
    if qkv_format == "BLHc":
        scale_strides = tuple(int(v) for v in s.stride())
        scale_dims = scale_shape
    else:
        scale_strides = (int(s.stride(0)), int(s.stride(2)), int(s.stride(1)), int(s.stride(3)))
        scale_dims = (scale_shape[0], scale_shape[2], scale_shape[1], scale_shape[3])
    BR, BW, warps = _pick_tiles(n_rows, C_out)
    grid = (triton.cdiv(n_rows, BR), triton.cdiv(C_out, BW))
    _unpack_dequant2d_kernel[grid](
        y2d,
        s,
        scale_group_ids_arg,
        out,
        n_rows,
        orig_c,
        C_out,
        int(B),
        int(L),
        int(H),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        int(out.stride(3)),
        scale_strides[0],
        scale_strides[1],
        scale_strides[2],
        scale_strides[3],
        scale_dims[0],
        scale_dims[1],
        scale_dims[2],
        scale_dims[3],
        BITS=bits,
        VALS=vals,
        BLOCK_VALS=block_vals,
        QKV_FORMAT=0 if qkv_format == "BLHc" else 1,
        USE_SCALE_GROUP_IDS=use_scale_group_ids,
        BLOCK_ROWS=BR,
        BLOCK_WORDS=BW,
        num_warps=warps,
        num_stages=2,
    )
    return out

# ===========================
# Pure-PyTorch fallback
# ===========================
def _signed_to_unsigned_width(x_int8: torch.Tensor, bits: int) -> torch.Tensor:
    mask = (1 << bits) - 1
    return x_int8.to(torch.int32) & mask

def _unsigned_to_signed_width(u: torch.Tensor, bits: int) -> torch.Tensor:
    sign = 1 << (bits - 1)
    u = u.to(torch.int32) & ((1 << bits) - 1)
    x = torch.where((u & sign) != 0, u - (1 << bits), u)
    return x.to(torch.int8)

def pack_last_dim_to_int32_python(q_int8: torch.Tensor, bits: int) -> Tuple[torch.Tensor, Dict[str,int]]:
    assert bits in (2, 3, 4, 6, 8), "bit-pack only implements 2/3/4/6/8 bits"
    # q3 uses a pure PyTorch fallback with 10 values packed into 30 bits.
    vals = 10 if bits == 3 else (32 // bits)
    *lead, C = q_int8.shape
    pad_len = (-C) % vals
    if pad_len:
        pad = torch.zeros((*lead, pad_len), dtype=q_int8.dtype, device=q_int8.device)
        q_int8 = torch.cat([q_int8, pad], dim=-1)
        C_padded = C + pad_len
    else:
        C_padded = C
    new_last = C_padded // vals
    q_u = _signed_to_unsigned_width(q_int8, bits)  # -> int32 non-negative
    q_u = q_u.view(*lead, new_last, vals).to(torch.int32)
    shifts = (torch.arange(vals, device=q_int8.device, dtype=torch.int32) * bits).view(
        *([1] * (q_u.dim() - 1)), vals
    )
    packed = (q_u << shifts).sum(dim=-1).to(torch.int32)  # [..., new_last]
    meta = {'orig_c': int(C), 'vals_per_word': int(vals), 'pad_len': int(pad_len), 'bits': int(bits)}
    return packed, meta

def unpack_last_dim_from_int32_python(packed: torch.Tensor, meta: Dict[str,int]) -> torch.Tensor:
    bits = int(meta['bits'])
    vals = int(meta['vals_per_word'])
    pad_len = int(meta['pad_len'])
    orig_c = int(meta['orig_c'])
    mask = (1 << bits) - 1
    shifts = torch.arange(vals, device=packed.device, dtype=torch.int32) * bits
    pieces_u = [(packed >> s) & mask for s in shifts]
    u_stack = torch.stack(pieces_u, dim=-1)  # [..., vals]
    unpacked_u = u_stack.reshape(*packed.shape[:-1], packed.shape[-1] * vals)  # [..., C_padded]
    if pad_len:
        unpacked_u = unpacked_u[..., :orig_c]
    q_int8 = _unsigned_to_signed_width(unpacked_u, bits)
    return q_int8

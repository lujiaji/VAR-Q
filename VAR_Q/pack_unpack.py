# VAR_Q/pack_unpack.py
import torch
from typing import Tuple, Dict

# ===== Optional Triton =====
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False
    
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
        offs      = tl.arange(0, VALS)                                    # [VALS]
        row_in_b  = row_idx[:, None, None] * C                            # [BR,1,1]
        row_out_b = row_idx[:, None] * C_OUT                              # [BR,1]
        base_cols = word_idx[None, :, None] * VALS + offs[None, None, :]  # [1,BW,VALS]

        mask_cols = mask_row[:, None, None] & mask_word[None, :, None] & (base_cols < C)

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

        offs = tl.arange(0, VALS)                                         # [VALS]
        words_mat = words[:, :, None]                                     # [BR,BW,1]

        mask_bits = (1 << BITS) - 1
        sign_bit  = 1 << (BITS - 1)
        pieces_u  = (words_mat >> (offs[None, None, :] * BITS)) & mask_bits   # [BR,BW,VALS]
        pieces_s  = tl.where((pieces_u & sign_bit) != 0, pieces_u - (1 << BITS), pieces_u).to(tl.int8)

        # output columns for each small piece
        out_cols  = word_idx[None, :, None] * VALS + offs[None, None, :]       # [1,BW,VALS]
        mask_cols = mask_row[:, None, None] & mask_word[None, :, None] & (out_cols < C)

        tl.store(out_ptr + row_out_b + out_cols, pieces_s, mask=mask_cols)

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
    assert bits in (2, 4, 8)
    x = q_int8.contiguous()
    *lead, C = x.shape
    N_ROWS = int(x.numel() // C)
    VALS   = 32 // bits
    C_out  = (C + VALS - 1) // VALS
    pad_len= C_out * VALS - C

    x2d = x.view(N_ROWS, C)
    out2d = torch.empty((N_ROWS, C_out), dtype=torch.int32, device=x.device)

    BR, BW, warps = _pick_tiles(N_ROWS, C_out)
    grid = (triton.cdiv(N_ROWS, BR), triton.cdiv(C_out, BW))

    _pack2d_kernel[grid](
        x2d, out2d,
        N_ROWS, C, C_out,
        BITS=bits, VALS=VALS,
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
    VALS  = int(meta['vals_per_word'])
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
        BITS=bits, VALS=VALS,
        BLOCK_ROWS=BR, BLOCK_WORDS=BW,
        num_warps=warps, num_stages=2
    )
    return out2d.view(*lead, orig_c)

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

"""Fused dequant + FlashAttention-2 forward (Triton). q8 / VARQ / forward-only."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _fa2_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    H, M, N, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_ptrs = (Q + b * stride_qb + h * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = (K + b * stride_kb + h * stride_kh
                  + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(offs_n[None, :] < N, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = (V + b * stride_vb + h * stride_vh
                  + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = (Out + b * stride_ob + h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < M)


@triton.jit
def _dequant_tile(packed_ptr, scale_ptr, step_ids_ptr,
                  base_b, base_h, h, stride_pn, stride_pw,
                  scale_sb, scale_ss, scale_sh, scale_sd,
                  offs_n, N, D: tl.constexpr, BITS: tl.constexpr):
    """Load a [BLOCK_N, D] fp16 tile from packed int32 + VARQ scale."""
    offs_d = tl.arange(0, D)
    offs_w = tl.arange(0, D // 4)
    p_ptrs = (packed_ptr + base_b + base_h
              + offs_n[:, None] * stride_pn + offs_w[None, :] * stride_pw)
    words = tl.load(p_ptrs, mask=offs_n[:, None] < N, other=0)   # [BN,D/4] int32
    mask_bits = 0xFF
    sign_bit = 0x80
    piece0_u = words & mask_bits
    piece1_u = (words >> 8) & mask_bits
    piece2_u = (words >> 16) & mask_bits
    piece3_u = (words >> 24) & mask_bits
    piece0_s = tl.where((piece0_u & sign_bit) != 0, piece0_u - 256, piece0_u)
    piece1_s = tl.where((piece1_u & sign_bit) != 0, piece1_u - 256, piece1_u)
    piece2_s = tl.where((piece2_u & sign_bit) != 0, piece2_u - 256, piece2_u)
    piece3_s = tl.where((piece3_u & sign_bit) != 0, piece3_u - 256, piece3_u)
    piece02 = tl.interleave(piece0_s, piece2_s)
    piece13 = tl.interleave(piece1_s, piece3_s)
    piece_s = tl.interleave(piece02, piece13).to(tl.float32)
    step = tl.load(step_ids_ptr + offs_n, mask=offs_n < N, other=0)   # [BN]
    s_ptrs = (scale_ptr + step[:, None] * scale_ss + h * scale_sh
              + offs_d[None, :] * scale_sd)
    scale = tl.load(s_ptrs, mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
    return (piece_s * scale).to(tl.float16)


@triton.jit
def _packed_fa2_fwd_kernel(
    Q, KPacked, VPacked, KScale, VScale, StepIds, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kpb, stride_kph, stride_kpn, stride_kpw,
    stride_vpb, stride_vph, stride_vpn, stride_vpw,
    stride_ksb, stride_kss, stride_ksh, stride_ksd,
    stride_vsb, stride_vss, stride_vsh, stride_vsd,
    stride_ob, stride_oh, stride_om, stride_od,
    H, M, N, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D: tl.constexpr, BITS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_ptrs = (Q + b * stride_qb + h * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = _dequant_tile(
            KPacked, KScale + b * stride_ksb, StepIds,
            b * stride_kpb, h * stride_kph, h, stride_kpn, stride_kpw,
            stride_ksb, stride_kss, stride_ksh, stride_ksd,
            offs_n, N, D, BITS,
        )
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(offs_n[None, :] < N, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = _dequant_tile(
            VPacked, VScale + b * stride_vsb, StepIds,
            b * stride_vpb, h * stride_vph, h, stride_vpn, stride_vpw,
            stride_vsb, stride_vss, stride_vsh, stride_vsd,
            offs_n, N, D, BITS,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = (Out + b * stride_ob + h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < M)


@triton.jit
def _two_segment_fa2_fwd_kernel(
    Q, KPacked, VPacked, KScale, VScale, StepIds, KFresh, VFresh, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kpb, stride_kph, stride_kpn, stride_kpw,
    stride_vpb, stride_vph, stride_vpn, stride_vpw,
    stride_ksb, stride_kss, stride_ksh, stride_ksd,
    stride_vsb, stride_vss, stride_vsh, stride_vsd,
    stride_kfb, stride_kfh, stride_kfn, stride_kfd,
    stride_vfb, stride_vfh, stride_vfn, stride_vfd,
    stride_ob, stride_oh, stride_om, stride_od,
    H, M, N_cached, N_fresh, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D: tl.constexpr, BITS: tl.constexpr, EVEN_N_FRESH: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_ptrs = (Q + b * stride_qb + h * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    for start_n in range(0, N_cached, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = _dequant_tile(
            KPacked, KScale + b * stride_ksb, StepIds,
            b * stride_kpb, h * stride_kph, h, stride_kpn, stride_kpw,
            stride_ksb, stride_kss, stride_ksh, stride_ksd,
            offs_n, N_cached, D, BITS,
        )
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(offs_n[None, :] < N_cached, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = _dequant_tile(
            VPacked, VScale + b * stride_vsb, StepIds,
            b * stride_vpb, h * stride_vph, h, stride_vpn, stride_vpw,
            stride_vsb, stride_vss, stride_vsh, stride_vsd,
            offs_n, N_cached, D, BITS,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    for start_n in range(0, N_fresh, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = (KFresh + b * stride_kfb + h * stride_kfh
                  + offs_n[:, None] * stride_kfn + offs_d[None, :] * stride_kfd)
        if EVEN_N_FRESH:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < N_fresh, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if not EVEN_N_FRESH:
            qk = tl.where(offs_n[None, :] < N_fresh, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = (VFresh + b * stride_vfb + h * stride_vfh
                  + offs_n[:, None] * stride_vfn + offs_d[None, :] * stride_vfd)
        if EVEN_N_FRESH:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N_fresh, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = (Out + b * stride_ob + h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < M)


def _plain_attention(q, k, v, block_m=128, block_n=32, num_warps=4, num_stages=2):
    """q,k,v in BHLc [B,H,L,D] fp16. Returns [B,H,Lq,D] fp16."""
    B, H, M, D = q.shape
    N = k.shape[2]
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(M, block_m), B * H)
    _fa2_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, M, N, sm_scale,
        BLOCK_M=block_m, BLOCK_N=block_n, D=D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def _layout_dims(fmt):
    if fmt == "BHLc":
        return 1, 2
    if fmt == "BLHc":
        return 2, 1
    raise ValueError(f"Unsupported qkv layout: {fmt}")


def _scale_strides(scale, H):
    if scale.shape[1] == H:
        return scale.stride(0), scale.stride(2), scale.stride(1), scale.stride(-1)
    if scale.shape[2] == H:
        return scale.stride(0), scale.stride(1), scale.stride(2), scale.stride(-1)
    raise ValueError(f"Scale tensor shape {tuple(scale.shape)} does not match H={H}")


def _packed_attention(
    q, k_packed, v_packed, k_scale, v_scale, step_ids, bits,
    block_m=64, block_n=32, num_warps=4, num_stages=3, fmt="BHLc",
):
    """q in BHLc/BLHc, packed K/V in matching layout. Returns fp16 in q layout."""
    if bits != 8:
        raise NotImplementedError("Task 3 implements q8 packed attention only")
    head_dim, seq_dim = _layout_dims(fmt)
    B = q.shape[0]
    H = q.shape[head_dim]
    M = q.shape[seq_dim]
    D = q.shape[-1]
    N = k_packed.shape[seq_dim]
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(M, block_m), B * H)
    k_scale_strides = _scale_strides(k_scale, H)
    v_scale_strides = _scale_strides(v_scale, H)
    _packed_fa2_fwd_kernel[grid](
        q, k_packed, v_packed, k_scale, v_scale, step_ids, out,
        q.stride(0), q.stride(head_dim), q.stride(seq_dim), q.stride(-1),
        k_packed.stride(0), k_packed.stride(head_dim), k_packed.stride(seq_dim), k_packed.stride(-1),
        v_packed.stride(0), v_packed.stride(head_dim), v_packed.stride(seq_dim), v_packed.stride(-1),
        *k_scale_strides,
        *v_scale_strides,
        out.stride(0), out.stride(head_dim), out.stride(seq_dim), out.stride(-1),
        H, M, N, sm_scale,
        BLOCK_M=block_m, BLOCK_N=block_n, D=D, BITS=bits,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def _two_segment_attention(
    q, k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh, bits,
    block_m=128, block_n=32, num_warps=4, num_stages=2, fmt="BHLc",
):
    """q/fresh in BHLc/BLHc, packed K/V in matching layout. Returns fp16 in q layout."""
    if bits != 8:
        raise NotImplementedError("Task 4 implements q8 two-segment attention only")
    head_dim, seq_dim = _layout_dims(fmt)
    B = q.shape[0]
    H = q.shape[head_dim]
    M = q.shape[seq_dim]
    D = q.shape[-1]
    N_cached = k_packed.shape[seq_dim]
    N_fresh = k_fresh.shape[seq_dim]
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(M, block_m), B * H)
    k_scale_strides = _scale_strides(k_scale, H)
    v_scale_strides = _scale_strides(v_scale, H)
    _two_segment_fa2_fwd_kernel[grid](
        q, k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh, out,
        q.stride(0), q.stride(head_dim), q.stride(seq_dim), q.stride(-1),
        k_packed.stride(0), k_packed.stride(head_dim), k_packed.stride(seq_dim), k_packed.stride(-1),
        v_packed.stride(0), v_packed.stride(head_dim), v_packed.stride(seq_dim), v_packed.stride(-1),
        *k_scale_strides,
        *v_scale_strides,
        k_fresh.stride(0), k_fresh.stride(head_dim), k_fresh.stride(seq_dim), k_fresh.stride(-1),
        v_fresh.stride(0), v_fresh.stride(head_dim), v_fresh.stride(seq_dim), v_fresh.stride(-1),
        out.stride(0), out.stride(head_dim), out.stride(seq_dim), out.stride(-1),
        H, M, N_cached, N_fresh, sm_scale,
        BLOCK_M=block_m, BLOCK_N=block_n, D=D, BITS=bits,
        EVEN_N_FRESH=(N_fresh % block_n) == 0,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def _build_step_ids(group_lengths, device):
    ids = torch.empty(sum(group_lengths), dtype=torch.int32, device=device)
    pos = 0
    for s, n in enumerate(group_lengths):
        ids[pos:pos + n] = s
        pos += n
    return ids


def fused_dequant_attention(q, k_quant, v_quant, k_fresh, v_fresh,
                            qkv_format="BHLc"):
    """Fused dequant + FA2 for one AR step.

    q, k_fresh, v_fresh: fp16 in qkv_format. k_quant/v_quant: VAR_Q with a
    packed VARQ cache. Returns attention output in qkv_format, fp16.
    """
    bits = int(k_quant.quant_bits)
    k_packed = k_quant._valid_cached_item().contiguous()
    v_packed = v_quant._valid_cached_item().contiguous()
    k_scale = k_quant._valid_cached_scale().contiguous().to(torch.float16)
    v_scale = v_quant._valid_cached_scale().contiguous().to(torch.float16)
    group_lengths = list(k_quant._scale_L_counts)
    step_ids = _build_step_ids(group_lengths, q.device)
    return _two_segment_attention(
        q, k_packed, v_packed, k_scale, v_scale, step_ids,
        k_fresh, v_fresh, bits, fmt=qkv_format,
    )

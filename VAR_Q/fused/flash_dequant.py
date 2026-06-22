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
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = (K + b * stride_kb + h * stride_kh
                  + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(offs_n[None, :] < N, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = (V + b * stride_vb + h * stride_vh
                  + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = (Out + b * stride_ob + h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < M)


def _plain_attention(q, k, v, block_m=64, block_n=64):
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
    )
    return out


def fused_dequant_attention(*args, **kwargs):  # filled in Task 6
    raise NotImplementedError("implemented in Task 6")

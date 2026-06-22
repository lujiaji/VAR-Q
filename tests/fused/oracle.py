"""Pure-torch reference for fused dequant attention tests. No Triton."""
import torch
import torch.nn.functional as F

from VAR_Q.quant import VAR_Q

# Infinity-8B first-12-scale patch sides (cached); last scale (64) stays fresh.
CACHE_PATCH = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48]
LAST_PATCH = 64


def make_kv_tensor(B, H, tokens, D, fmt, device, dtype=torch.float16):
    if fmt == "BHLc":
        return torch.randn(B, H, tokens, D, device=device, dtype=dtype)
    return torch.randn(B, tokens, H, D, device=device, dtype=dtype)


def build_varq_cache(patch_list, B, H, D, bits, fmt, device, kv_role):
    """Build a VAR_Q packed cache over patch_list scales.

    Returns (quantizer, ref_fp16_cache) where ref_fp16_cache is the exact
    dequantized cache the kernel must reproduce, in `fmt` layout.
    """
    q = VAR_Q(
        quant_bits=bits, qkv_format=fmt, quant_method="VARQ",
        kv_role=kv_role, pack_to_int32=True, dequant_dtype="fp16",
    )
    for p in patch_list:
        x = make_kv_tensor(B, H, p * p, D, fmt, device)
        q.use_var_q(x, cache_current=True)
    ref = q.dequant_all().clone()  # fp16, [B,H,L,D] or [B,L,H,D]
    return q, ref


def extract_packed(q):
    """Pull the raw packed buffers a fused kernel consumes from a VAR_Q cache.

    Returns dict with packed int32 cache [B,H,L,W] (W=D//4 for q8), compact
    scale [B,H,num_steps,D], and per-step token counts (group lengths).
    """
    packed = q._valid_cached_item().contiguous()   # int32
    scale = q._valid_cached_scale().contiguous()    # fp16/bf16 compact
    group_lengths = list(q._scale_L_counts)
    return {
        "packed": packed,
        "scale": scale.to(torch.float16),
        "group_lengths": group_lengths,
        "pack_meta": dict(q._pack_meta),
    }


def step_ids_from_groups(group_lengths, device):
    """[L] int32 mapping each token to its step index (for scale lookup)."""
    ids = torch.empty(sum(group_lengths), dtype=torch.int32, device=device)
    pos = 0
    for s, n in enumerate(group_lengths):
        ids[pos:pos + n] = s
        pos += n
    return ids


def to_bhld(t, fmt):
    """Normalize a KV/Q tensor to [B,H,L,D] for the reference math."""
    if fmt == "BHLc":
        return t
    return t.transpose(1, 2).contiguous()  # BLHc -> BHLc


def ref_attention(q, k, v, fmt):
    """Non-causal full attention reference in fp32 accumulation.

    q,k,v are in `fmt` layout. Returns output in `fmt` layout, fp16.
    """
    qb = to_bhld(q, fmt).float()
    kb = to_bhld(k, fmt).float()
    vb = to_bhld(v, fmt).float()
    out = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False)
    out = out.to(torch.float16)
    if fmt == "BLHc":
        out = out.transpose(1, 2).contiguous()
    return out

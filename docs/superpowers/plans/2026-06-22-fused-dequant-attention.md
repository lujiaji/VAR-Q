# Fused Dequant + FlashAttention-2 Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Triton FlashAttention-2 forward kernel that dequantizes the VAR-Q packed KV cache inside the K/V tile loop (read int32 + scale, dequant in SRAM, never materialize fp16 K/V to HBM), recovering the ~2.7 ms dequant tax as throughput at Infinity-8B's heaviest AR step.

**Architecture:** Stage in TDD order — (1) a pure-torch reference oracle, (2) a plain Triton FA2 forward matching it, (3) inject dequant-on-load for a single packed KV segment, (4) extend to the two-segment heaviest step (cached packed + fresh fp16) in one online-softmax pass, (5) both qkv layouts, (6) a Python wrapper that feeds the kernel from VAR_Q quantizers, (7) integration into the Infinity forward hook gated by `enable_fused_kv_flashattn`, (8) benchmark vs baselines on A100.

**Tech Stack:** Triton (kernel), PyTorch fp16, the existing `VAR_Q` quantizer + `pack_unpack.py` bit layout, pytest. Hardware: A100 / CUDA, forward-only inference.

**Scope (from spec `docs/superpowers/specs/2026-06-22-fused-dequant-attention-design.md`):** q8 · VARQ · {BHLc, BLHc} · forward-only. Out of scope: q4/q3/q2, other 6 grouping methods, backward, non-Infinity models, CUTLASS Phase B.

---

## Remote execution (read first)

There is **no local GPU**. The kernel and all CUDA tests run on the A100 box,
inside the `kivi_bench` container (torch 2.7.0a0, Triton bundled, flash_attn
2.7.3, CUDA, A100 80GB). Code is edited locally and synced. The first task
creates a helper so every later "Run" step is one command.

The smoke/oracle-only tests (no Triton) run locally on CPU; CUDA tests run remote.

## Packing facts the kernel depends on (confirmed in `VAR_Q/pack_unpack.py`)

- q8: `bits=8`, `vals_per_word = 32 // 8 = 4`. Output channel `c` of a row comes
  from packed word `c // 4`, slot `c % 4`:
  `out_cols = word_idx * VALS + slot` (pack_unpack.py:90,126).
- Per-channel dequant (lifted from `_unpack_dequant2d_kernel`, pack_unpack.py:129-132):
  ```
  mask_bits = (1 << 8) - 1            # 0xFF
  sign_bit  = 1 << 7                  # 0x80
  piece_u = (word >> (slot * 8)) & mask_bits
  piece_s = where((piece_u & sign_bit) != 0, piece_u - 256, piece_u)
  value   = float(piece_s) * scale
  ```
- VARQ scale: one vector per `(step, head, head_dim channel)`, broadcast over the
  tokens of that step. Cached compact scale is concatenated per step along the
  seq dim; `kq._scale_L_counts` holds the per-step token counts (group lengths).
  Token `l` → step `s` via the prefix sums of `_scale_L_counts`.
- A q8 packed cache row holds `D // 4 = 32` int32 words for `D = 128`.
- Packed layout is identical for BHLc and BLHc; only outer strides differ.

## File structure

- Create `VAR_Q/fused/__init__.py` — exports `fused_dequant_attention`.
- Create `VAR_Q/fused/flash_dequant.py` — the Triton kernel(s) + the Python
  wrapper. One responsibility: fused dequant attention. This is the only file
  holding Triton code.
- Create `tests/fused/__init__.py` — empty package marker.
- Create `tests/fused/oracle.py` — pure-torch reference attention + a VAR-Q cache
  builder shared by all fused tests. No Triton.
- Create `tests/fused/test_fused_dequant_attention.py` — the test suite.
- Create `scripts/bench/remote_test.sh` — sync + run-in-container helper.
- Create `scripts/bench/microbench_fused_vs_baseline.py` — final benchmark.
- Modify `VAR_Q/hooks/runtime.py` — route the Infinity attention branch to the
  fused kernel when `enable_fused_kv_flashattn` is on.

---

### Task 1: Remote test helper, reference oracle, and a red test

**Files:**
- Create: `scripts/bench/remote_test.sh`
- Create: `tests/fused/__init__.py`
- Create: `tests/fused/oracle.py`
- Create: `tests/fused/test_fused_dequant_attention.py`
- Create: `VAR_Q/fused/__init__.py`

- [ ] **Step 1: Write the remote-test helper**

Create `scripts/bench/remote_test.sh`:

```bash
#!/usr/bin/env bash
# Sync the repo to the A100 box and run a command inside the kivi_bench container.
# Usage: scripts/bench/remote_test.sh '<command run from /work/VAR-Q>'
set -euo pipefail
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)/"
rsync -az -e 'ssh -o BatchMode=yes' \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='scripts/output' \
  "$LOCAL_DIR" A100:work/kv-quant-eval/VAR-Q/
ssh -o BatchMode=yes A100 \
  "docker exec -e CUDA_VISIBLE_DEVICES=0 kivi_bench bash -lc 'cd /work/VAR-Q && ${1}'"
```

Make it executable: `chmod +x scripts/bench/remote_test.sh`

- [ ] **Step 2: Write the package markers**

Create `tests/fused/__init__.py` (empty file).

Create `VAR_Q/fused/__init__.py`:

```python
from .flash_dequant import fused_dequant_attention

__all__ = ["fused_dequant_attention"]
```

- [ ] **Step 3: Write the reference oracle**

Create `tests/fused/oracle.py`:

```python
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
```

- [ ] **Step 4: Write the first (red) test**

Create `tests/fused/test_fused_dequant_attention.py`:

```python
import pytest
import torch

from tests.fused import oracle


def test_import_exists():
    # Red until VAR_Q/fused/flash_dequant.py defines the symbol.
    from VAR_Q.fused import fused_dequant_attention
    assert callable(fused_dequant_attention)


def test_oracle_shapes_cpu():
    B, H, D = 1, 2, 128
    fmt = "BHLc"
    q = oracle.make_kv_tensor(B, H, 16, D, fmt, "cpu")
    k = oracle.make_kv_tensor(B, H, 40, D, fmt, "cpu")
    v = oracle.make_kv_tensor(B, H, 40, D, fmt, "cpu")
    out = oracle.ref_attention(q, k, v, fmt)
    assert out.shape == (B, H, 16, D)
    assert out.dtype == torch.float16
```

- [ ] **Step 5: Run the oracle test locally (CPU), confirm import test fails**

Run: `python -m pytest tests/fused/test_fused_dequant_attention.py -v`
Expected: `test_oracle_shapes_cpu` PASS; `test_import_exists` FAIL with
`ModuleNotFoundError`/`ImportError` (flash_dequant.py not written yet).

- [ ] **Step 6: Commit**

```bash
git add scripts/bench/remote_test.sh tests/fused VAR_Q/fused
git commit -m "test: fused-attention oracle + remote-test helper (red)"
```

---

### Task 2: Plain Triton FA2 forward (fp16, single KV segment)

Build the attention foundation with no dequant yet: a standard FA2 forward that
matches the oracle on fp16 K/V. Dequant is injected in Task 3.

**Files:**
- Create: `VAR_Q/fused/flash_dequant.py`
- Test: `tests/fused/test_fused_dequant_attention.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/fused/test_fused_dequant_attention.py`:

```python
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("Lq,Lkv", [(64, 64), (128, 300), (4096, 4096)])
def test_plain_fa2_matches_oracle(Lq, Lkv):
    from VAR_Q.fused.flash_dequant import _plain_attention
    B, H, D, fmt = 1, 4, 128, "BHLc"
    dev = "cuda"
    torch.manual_seed(0)
    q = oracle.make_kv_tensor(B, H, Lq, D, fmt, dev)
    k = oracle.make_kv_tensor(B, H, Lkv, D, fmt, dev)
    v = oracle.make_kv_tensor(B, H, Lkv, D, fmt, dev)
    out = _plain_attention(q, k, v)            # [B,H,Lq,D] fp16
    ref = oracle.ref_attention(q, k, v, fmt)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
```

- [ ] **Step 2: Write the plain FA2 kernel + wrapper**

Create `VAR_Q/fused/flash_dequant.py`:

```python
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
```

- [ ] **Step 3: Run on A100, expect FAIL→PASS iteration**

Run: `scripts/bench/remote_test.sh 'python -m pytest tests/fused/test_fused_dequant_attention.py::test_plain_fa2_matches_oracle -v'`
Expected: PASS for all three `(Lq,Lkv)` cases. If a case fails on tolerance,
debug the online-softmax rescale (the `alpha` update) — do not loosen tolerance
beyond `2e-2`.

- [ ] **Step 4: Commit**

```bash
git add VAR_Q/fused/flash_dequant.py tests/fused/test_fused_dequant_attention.py
git commit -m "feat: plain Triton FA2 forward matching oracle"
```

---

### Task 3: Inject dequant-on-load for a single packed KV segment

Now make the kernel read packed int32 K/V + compact VARQ scale and dequant each
tile in SRAM. Single segment (whole KV is the packed cache, no fresh fp16 yet).

**Files:**
- Modify: `VAR_Q/fused/flash_dequant.py`
- Modify: `tests/fused/oracle.py` (add a packed-input extractor)
- Test: `tests/fused/test_fused_dequant_attention.py`

- [ ] **Step 1: Add a packed-cache extractor to the oracle**

Add to `tests/fused/oracle.py`:

```python
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
```

- [ ] **Step 2: Write the failing test**

Add to `tests/fused/test_fused_dequant_attention.py`:

```python
@cuda
def test_dequant_load_single_segment():
    from VAR_Q.fused.flash_dequant import _packed_attention
    B, H, D, fmt, bits = 1, 4, 128, "BHLc", 8
    dev = "cuda"
    torch.manual_seed(0)
    patch = [1, 2, 4, 6, 8]                      # small cache for the unit test
    kq, ref_k = oracle.build_varq_cache(patch, B, H, D, bits, fmt, dev, "k")
    vq, ref_v = oracle.build_varq_cache(patch, B, H, D, bits, fmt, dev, "v")
    Lkv = ref_k.shape[2]
    q = oracle.make_kv_tensor(B, H, Lkv, D, fmt, dev)

    kp = oracle.extract_packed(kq)
    vp = oracle.extract_packed(vq)
    step_ids = oracle.step_ids_from_groups(kp["group_lengths"], dev)

    out = _packed_attention(
        q, kp["packed"], vp["packed"], kp["scale"], vp["scale"], step_ids, bits,
    )
    ref = oracle.ref_attention(q, ref_k, ref_v, fmt)   # ref_k = kq.dequant_all()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
```

- [ ] **Step 3: Add the packed kernel + wrapper**

Add to `VAR_Q/fused/flash_dequant.py`:

```python
@triton.jit
def _dequant_tile(packed_ptr, scale_ptr, step_ids_ptr,
                  base_b, base_h, h, stride_pn, stride_pw,
                  scale_sb, scale_ss, scale_sh, scale_sd,
                  offs_n, N, D: tl.constexpr, BITS: tl.constexpr):
    """Load a [BLOCK_N, D] fp32 tile from packed int32 + VARQ scale."""
    VALS = 32 // BITS                       # q8 -> 4
    offs_d = tl.arange(0, D)
    word_col = offs_d // VALS               # [D]
    slot = offs_d % VALS                    # [D]
    p_ptrs = (packed_ptr + base_b + base_h
              + offs_n[:, None] * stride_pn + word_col[None, :] * stride_pw)
    words = tl.load(p_ptrs, mask=offs_n[:, None] < N, other=0)   # [BN,D] int32
    mask_bits = (1 << BITS) - 1
    sign_bit = 1 << (BITS - 1)
    piece_u = (words >> (slot[None, :] * BITS)) & mask_bits
    piece_s = tl.where((piece_u & sign_bit) != 0, piece_u - (1 << BITS), piece_u)
    piece_s = piece_s.to(tl.float32)
    step = tl.load(step_ids_ptr + offs_n, mask=offs_n < N, other=0)   # [BN]
    s_ptrs = (scale_ptr + step[:, None] * scale_ss + h * scale_sh
              + offs_d[None, :] * scale_sd)
    scale = tl.load(s_ptrs, mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
    return piece_s * scale
```

Then a `_packed_fa2_fwd_kernel` mirroring `_fa2_fwd_kernel` but replacing the K
and V `tl.load(...).to(tl.float32)` lines with `_dequant_tile(...)` calls (K and
V get their own packed/scale pointers and strides), and a `_packed_attention`
wrapper mirroring `_plain_attention` that passes packed/scale/step_ids and
`BITS=bits`. The scale tensor is `[B,H,num_steps,D]`; pass `scale.stride(...)`
for `scale_sb/ss/sh/sd`. `stride_pw = packed.stride(3)` (word dim),
`stride_pn = packed.stride(2)` (seq dim) for BHLc.

- [ ] **Step 4: Run on A100, iterate to PASS**

Run: `scripts/bench/remote_test.sh 'python -m pytest tests/fused/test_fused_dequant_attention.py::test_dequant_load_single_segment -v'`
Expected: PASS. If values are off by a constant per-channel factor, check the
`step` lookup and `scale` strides. If off by sign on some channels, check the
sign-extend (`piece_u - (1<<BITS)`). Cross-check one tile against
`kq.dequant_all()` by adding a temporary `_dequant_tile`-only test if needed.

- [ ] **Step 5: Commit**

```bash
git add VAR_Q/fused/flash_dequant.py tests/fused/oracle.py tests/fused/test_fused_dequant_attention.py
git commit -m "feat: dequant-on-load packed KV attention (single segment)"
```

---

### Task 4: Two-segment fused step (cached packed + fresh fp16)

The heaviest AR step: KV = cached packed (first 12 scales, 6425 tok) followed by
fresh fp16 (last scale, 4096 tok), in one online-softmax pass. The fresh segment
is NOT cached (the last-scale trick).

**Files:**
- Modify: `VAR_Q/fused/flash_dequant.py`
- Test: `tests/fused/test_fused_dequant_attention.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/fused/test_fused_dequant_attention.py`:

```python
@cuda
def test_two_segment_matches_oracle():
    from VAR_Q.fused.flash_dequant import _two_segment_attention
    B, H, D, fmt, bits = 1, 4, 128, "BHLc", 8
    dev = "cuda"
    torch.manual_seed(0)
    patch = oracle.CACHE_PATCH                         # 12 scales, 6425 tok
    kq, ref_k = oracle.build_varq_cache(patch, B, H, D, bits, fmt, dev, "k")
    vq, ref_v = oracle.build_varq_cache(patch, B, H, D, bits, fmt, dev, "v")
    fresh_n = oracle.LAST_PATCH ** 2                   # 4096
    fk = oracle.make_kv_tensor(B, H, fresh_n, D, fmt, dev)
    fv = oracle.make_kv_tensor(B, H, fresh_n, D, fmt, dev)
    q = oracle.make_kv_tensor(B, H, fresh_n, D, fmt, dev)

    kp, vp = oracle.extract_packed(kq), oracle.extract_packed(vq)
    step_ids = oracle.step_ids_from_groups(kp["group_lengths"], dev)
    out = _two_segment_attention(
        q, kp["packed"], vp["packed"], kp["scale"], vp["scale"], step_ids,
        fk, fv, bits,
    )
    full_k = torch.cat([ref_k, fk], dim=2)
    full_v = torch.cat([ref_v, fv], dim=2)
    ref = oracle.ref_attention(q, full_k, full_v, fmt)
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
```

- [ ] **Step 2: Implement the two-segment kernel + wrapper**

Add `_two_segment_fa2_fwd_kernel` to `VAR_Q/fused/flash_dequant.py`. It runs the
same online-softmax accumulator across two KV loops sharing `m_i/l_i/acc`:

1. Loop A over `range(0, N_cached, BLOCK_N)`: K/V tiles via `_dequant_tile`.
2. Loop B over `range(0, N_fresh, BLOCK_N)`: K/V tiles via plain fp16
   `tl.load(...).to(tl.float32)` from the fresh tensors.

Both loops use the identical `m_new/alpha/p/acc` update from `_fa2_fwd_kernel`.
Add `_two_segment_attention(q, k_packed, v_packed, k_scale, v_scale, step_ids,
k_fresh, v_fresh, bits)` mirroring earlier wrappers; pass both segments' pointers
and strides and `N_cached`, `N_fresh`. Grid is `(cdiv(Lq, BLOCK_M), B*H)`.

- [ ] **Step 3: Run on A100, iterate to PASS**

Run: `scripts/bench/remote_test.sh 'python -m pytest tests/fused/test_fused_dequant_attention.py::test_two_segment_matches_oracle -v'`
Expected: PASS. If the output looks like only one segment contributed, confirm
both loops mutate the SAME `m_i/l_i/acc` (no re-init between loops) and that the
final `acc / l_i` runs once after loop B.

- [ ] **Step 4: Commit**

```bash
git add VAR_Q/fused/flash_dequant.py tests/fused/test_fused_dequant_attention.py
git commit -m "feat: two-segment fused attention (cached packed + fresh fp16)"
```

---

### Task 5: Support both qkv layouts (BHLc and BLHc)

Packed layout is identical; only strides differ. Drive the kernel by passing the
right strides per format, and parametrize the tests.

**Files:**
- Modify: `VAR_Q/fused/flash_dequant.py`
- Test: `tests/fused/test_fused_dequant_attention.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/fused/test_fused_dequant_attention.py`:

```python
@cuda
@pytest.mark.parametrize("fmt", ["BHLc", "BLHc"])
def test_two_segment_both_layouts(fmt):
    from VAR_Q.fused.flash_dequant import _two_segment_attention
    B, H, D, bits = 1, 4, 128, 8
    dev = "cuda"
    torch.manual_seed(0)
    patch = [1, 2, 4, 6, 8]
    kq, ref_k = oracle.build_varq_cache(patch, B, H, D, bits, fmt, dev, "k")
    vq, ref_v = oracle.build_varq_cache(patch, B, H, D, bits, fmt, dev, "v")
    fk = oracle.make_kv_tensor(B, H, 64, D, fmt, dev)
    fv = oracle.make_kv_tensor(B, H, 64, D, fmt, dev)
    q = oracle.make_kv_tensor(B, H, 64, D, fmt, dev)
    kp, vp = oracle.extract_packed(kq), oracle.extract_packed(vq)
    step_ids = oracle.step_ids_from_groups(kp["group_lengths"], dev)
    out = _two_segment_attention(
        q, kp["packed"], vp["packed"], kp["scale"], vp["scale"], step_ids,
        fk, fv, bits, fmt=fmt,
    )
    ref = oracle.ref_attention(
        q, torch.cat([ref_k, fk], dim=2 if fmt == "BHLc" else 1),
        torch.cat([ref_v, fv], dim=2 if fmt == "BHLc" else 1), fmt)
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
```

- [ ] **Step 2: Add `fmt` handling to the wrapper**

In `_two_segment_attention` (and `_packed_attention`), accept `fmt="BHLc"`. For
`fmt == "BHLc"`, seq stride is `.stride(2)`, head stride `.stride(1)`. For
`fmt == "BLHc"`, seq stride is `.stride(1)`, head stride `.stride(2)`. Pass the
selected strides for Q, K (packed seq/word), V, fresh K/V, and Out. The packed
word dim is always the last dim (`.stride(-1)`). The kernel body is unchanged —
it only sees `stride_*` scalars.

- [ ] **Step 3: Run on A100**

Run: `scripts/bench/remote_test.sh 'python -m pytest tests/fused/test_fused_dequant_attention.py::test_two_segment_both_layouts -v'`
Expected: PASS for both `BHLc` and `BLHc`.

- [ ] **Step 4: Commit**

```bash
git add VAR_Q/fused/flash_dequant.py tests/fused/test_fused_dequant_attention.py
git commit -m "feat: support BHLc and BLHc layouts in fused attention"
```

---

### Task 6: Public wrapper `fused_dequant_attention` from VAR_Q quantizers

A single entry point the hook calls: given `q`, the `k_quant`/`v_quant` VAR_Q
objects (holding the packed cache), and the fresh fp16 K/V, assemble kernel
inputs and return attention output.

**Files:**
- Modify: `VAR_Q/fused/flash_dequant.py`
- Test: `tests/fused/test_fused_dequant_attention.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/fused/test_fused_dequant_attention.py`:

```python
@cuda
def test_public_wrapper_matches_oracle():
    from VAR_Q.fused import fused_dequant_attention
    B, H, D, fmt, bits = 1, 4, 128, "BHLc", 8
    dev = "cuda"
    torch.manual_seed(0)
    patch = [1, 2, 4, 6, 8]
    kq, ref_k = oracle.build_varq_cache(patch, B, H, D, bits, fmt, dev, "k")
    vq, ref_v = oracle.build_varq_cache(patch, B, H, D, bits, fmt, dev, "v")
    fk = oracle.make_kv_tensor(B, H, 64, D, fmt, dev)
    fv = oracle.make_kv_tensor(B, H, 64, D, fmt, dev)
    q = oracle.make_kv_tensor(B, H, 64, D, fmt, dev)
    out = fused_dequant_attention(q, kq, vq, fk, fv, qkv_format=fmt)
    ref = oracle.ref_attention(
        q, torch.cat([ref_k, fk], dim=2), torch.cat([ref_v, fv], dim=2), fmt)
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
```

- [ ] **Step 2: Implement the wrapper**

Replace the `fused_dequant_attention` stub in `VAR_Q/fused/flash_dequant.py`:

```python
from tests.fused import oracle as _o  # NOTE: replace with inlined helpers; see step 3


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
```

- [ ] **Step 3: Inline `_build_step_ids` (no test-package import in library code)**

Add to `VAR_Q/fused/flash_dequant.py` and delete the `from tests.fused import`
line — library code must not import tests:

```python
def _build_step_ids(group_lengths, device):
    ids = torch.empty(sum(group_lengths), dtype=torch.int32, device=device)
    pos = 0
    for s, n in enumerate(group_lengths):
        ids[pos:pos + n] = s
        pos += n
    return ids
```

- [ ] **Step 4: Run on A100**

Run: `scripts/bench/remote_test.sh 'python -m pytest tests/fused/test_fused_dequant_attention.py::test_public_wrapper_matches_oracle -v'`
Expected: PASS.

- [ ] **Step 5: Run the full fused suite on A100**

Run: `scripts/bench/remote_test.sh 'python -m pytest tests/fused/ -v'`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add VAR_Q/fused/flash_dequant.py tests/fused/test_fused_dequant_attention.py
git commit -m "feat: public fused_dequant_attention wrapper from VAR_Q quantizers"
```

---

### Task 7: Integrate into the Infinity forward hook, gated by the flag

Route the Infinity attention branch to the fused kernel when
`enable_fused_kv_flashattn` is on; otherwise unchanged.

**Files:**
- Modify: `VAR_Q/hooks/runtime.py`
- Test: `tests/fused/test_hook_routing.py` (create)

- [ ] **Step 1: Read the current attention branch**

Run: `python - <<'PY'\nimport re,sys\nsrc=open('VAR_Q/hooks/runtime.py').read()\ni=src.index('_wrap_infinity_forward')\nprint(src[i:i+3000])\nPY`
Expected: see where `k`/`v` become attention dtype and where
`flash_attn_func`/SDPA is called (around the `VAR_visible_kvlen` block), plus the
config read at line 834. Note the exact local variable names for q/k/v and the
`cache_current`/last-scale decision.

- [ ] **Step 2: Write the routing unit test**

Create `tests/fused/test_hook_routing.py`:

```python
"""The flag must select the fused path; default must be unchanged."""
from VAR_Q.hooks import runtime


def test_flag_default_off():
    cfg = {}
    assert bool(cfg.get("enable_fused_kv_flashattn", False)) is False


def test_dispatch_helper_exists():
    # A small pure-python dispatcher we can unit-test without a GPU/model.
    assert hasattr(runtime, "_should_use_fused_kv_attn")


def test_dispatch_requires_flag_and_last_scale():
    f = runtime._should_use_fused_kv_attn
    assert f(enabled=True, is_last_scale=True, qkv_format="BHLc", bits=8) is True
    assert f(enabled=False, is_last_scale=True, qkv_format="BHLc", bits=8) is False
    # fusion is defined for the last (two-segment) step only in v1
    assert f(enabled=True, is_last_scale=False, qkv_format="BHLc", bits=8) is False
    # v1 supports q8 only
    assert f(enabled=True, is_last_scale=True, qkv_format="BHLc", bits=4) is False
```

- [ ] **Step 3: Run the routing test, expect FAIL**

Run: `python -m pytest tests/fused/test_hook_routing.py -v`
Expected: `test_dispatch_helper_exists` / `test_dispatch_requires_flag_and_last_scale`
FAIL (`_should_use_fused_kv_attn` not defined).

- [ ] **Step 4: Add the dispatcher helper**

Add to `VAR_Q/hooks/runtime.py` (module level):

```python
def _should_use_fused_kv_attn(enabled, is_last_scale, qkv_format, bits):
    """v1 fused path: only the last (two-segment) AR step, q8, known layout."""
    return bool(enabled) and bool(is_last_scale) and int(bits) == 8 \
        and qkv_format in ("BHLc", "BLHc")
```

- [ ] **Step 5: Wire the dispatcher into the attention branch**

In `_wrap_infinity_forward`, at the point where the last-scale decision and the
fresh fp16 `k`/`v` are available (the `cache_current = not _infinity_is_last_scale(...)`
site), add — before the existing `flash_attn_func`/SDPA call:

```python
if _should_use_fused_kv_attn(
    config.get("enable_fused_kv_flashattn", False),
    is_last_scale=not cache_current,
    qkv_format=self.k_quant.qkv_format,
    bits=self.k_quant.quant_bits,
):
    from VAR_Q.fused import fused_dequant_attention
    attn_out = fused_dequant_attention(
        q, self.k_quant, self.v_quant, k_fresh=k, v_fresh=v,
        qkv_format=self.k_quant.qkv_format,
    )
    # skip the dequant+flash_attn_func path below
else:
    ... existing path ...
```

Match the exact local names found in Step 1 (`q`, `k`, `v`, `config`,
`self.k_quant`). When fused, do NOT also call `_use_quantizer`/dequant for this
step — the kernel consumes the packed cache directly; the fresh k/v are the
pre-quant tensors.

- [ ] **Step 6: Run routing tests + mock hook tests**

Run: `python -m pytest tests/fused/test_hook_routing.py tests/test_runtime_hooks_mock.py -v`
Expected: PASS (these are CPU/mock; no GPU needed).

- [ ] **Step 7: Commit**

```bash
git add VAR_Q/hooks/runtime.py tests/fused/test_hook_routing.py
git commit -m "feat: gate Infinity attention on fused KV kernel via enable_fused_kv_flashattn"
```

---

### Task 8: Benchmark fused vs baselines on A100

Quantify the win and isolate the framework question (Triton-fused vs
Triton-plain) from the production comparison (vs flash_attn_func + dequant_all).

**Files:**
- Create: `scripts/bench/microbench_fused_vs_baseline.py`

- [ ] **Step 1: Write the benchmark**

Create `scripts/bench/microbench_fused_vs_baseline.py` reusing the shapes and
cuda-event timer from `microbench_dequant_vs_fa2.py` (import its `median_ms_cuda_event`
and `BenchShape`, or copy them). Build the 12-scale q8 cache once, then time
three pipelines at the heaviest step (Lq=4096, cached=6425, fresh=4096, H=32,
D=128):

1. **production**: `k_full = kq.dequant_all(); v_full = vq.dequant_all();`
   `cat fresh; flash_attn_func(q, k_full, v_full)`.
2. **fused**: `fused_dequant_attention(q, kq, vq, fk, fv, qkv_format="BHLc")`.
3. **plain-isolated**: `dequant_all` then `_plain_attention` (Triton attention,
   same framework as fused) — isolates the dequant-fusion delta from the
   Triton-vs-CUTLASS attention gap.

Print a table: `pipeline | total_ms | speedup_vs_production`. Assert all three
outputs match within `atol=3e-2` before timing (correctness gate).

- [ ] **Step 2: Run on A100**

Run: `scripts/bench/remote_test.sh 'python scripts/bench/microbench_fused_vs_baseline.py'`
Expected: a table. Target: `fused` total ≈ 4–4.5 ms vs `production` ≈ 6.5–7 ms
(~1.5–1.6×). The `production` vs `plain-isolated` gap reveals how much of any
shortfall is the Triton-vs-CUTLASS attention gap (informs the Phase B / CUTLASS
decision).

- [ ] **Step 3: Record results in the spec**

Append the measured table to
`docs/superpowers/specs/2026-06-22-fused-dequant-attention-design.md` under a new
"## Measured fused result" section. State plainly whether the ~1.6× target was
met and whether Phase B (CUTLASS) is warranted.

- [ ] **Step 4: Commit**

```bash
git add scripts/bench/microbench_fused_vs_baseline.py docs/superpowers/specs/2026-06-22-fused-dequant-attention-design.md
git commit -m "bench: fused vs baseline + record measured result"
```

---

## Self-review notes

- **Spec coverage:** two-segment heaviest step (Task 4), packed dequant-on-load
  (Task 3), both layouts (Task 5), `enable_fused_kv_flashattn` gate (Task 7),
  correctness vs fp16 reference (Tasks 2-6), Triton-fused-vs-Triton-plain
  isolation (Task 8) — all spec requirements mapped.
- **q8 only / VARQ only:** enforced in `_should_use_fused_kv_attn` (Task 7) and
  assumed in `_dequant_tile` (`BITS` constexpr); q4/q3/q2 deferred per scope.
- **No test import in library code:** Task 6 Step 3 explicitly removes the
  `from tests.fused import` line and inlines `_build_step_ids`.
- **Type/name consistency:** `_plain_attention`, `_packed_attention`,
  `_two_segment_attention`, `fused_dequant_attention`, `_dequant_tile`,
  `_build_step_ids`, `_should_use_fused_kv_attn`, `extract_packed`,
  `step_ids_from_groups`, `build_varq_cache`, `ref_attention` used consistently
  across tasks.
- **Open risk to watch during execution:** the per-channel packed load in
  `_dequant_tile` issues a `[BLOCK_N, D]` gather where `word_col` repeats each
  word `VALS` times; if this is bandwidth-bound, a later optimization is to load
  `[BLOCK_N, D//VALS]` words once and unpack in-register. Defer until Task 8
  shows whether it matters.

# Fused Dequant + FlashAttention-2 Kernel — Design Spec

**Date:** 2026-06-22
**Status:** Approved design, pending implementation plan
**Target:** Infinity-8B VAR-Q KV-cache, A100 / CUDA, inference forward only

## Problem

VAR-Q packs the KV cache (int8 → int32 at 2/3/4/6/8 bits), saving HBM, but
throughput did **not** improve — it got a *tax*. The quant/dequant path is not
fused with attention; it runs linearly:

1. dequant the packed cache → a full fp16 K/V tensor materialized to HBM,
2. `flash_attn_func` re-reads that fp16 K/V from HBM.

As long as attention consumes full fp16 K/V, quantization can never raise
throughput — it only saves memory and adds dequant overhead.

## Measured motivation (synthetic microbench, A100 80GB)

Heaviest AR step of Infinity-8B (last scale): `Q=[1,4096,32,128]`,
`KV=[1,10521,32,128]`, fp16. Cached (packed) KV = first 12 scales = 6425 tokens;
last scale (4096 tokens) stays fresh fp16 (not cached — the "last-scale trick").

Script: `scripts/bench/microbench_dequant_vs_fa2.py` (no model weights).

| bits | packedKV | A: last-scale roundtrip | B: cached dequant | C: FA2 attention | fuse_ratio (B+ws)/C | total_tax |
|------|----------|-------------------------|-------------------|------------------|---------------------|-----------|
| q8   | 52.8 MB  | 0.52 ms                 | **2.69 ms**       | 3.81 ms          | 0.71                | 0.85      |
| q4   | 26.5 MB  | 0.45 ms                 | 2.79 ms           | 3.81 ms          | 0.73                | 0.85      |
| q3   | 21.6 MB  | 0.51 ms                 | 2.85 ms           | 3.81 ms          | 0.75                | 0.89      |
| q2   | 13.4 MB  | 0.44 ms                 | 2.55 ms           | 3.81 ms          | 0.67                | 0.79      |

`t_workspace = 0.008 ms`, `t_empty_cache = 0.004 ms` (negligible — the
empty_cache / workspace-churn hypothesis is **debunked**).

**Findings:**
- Cached dequant (B) ≈ **70% of attention time**. The current step costs
  `~3.8ms attn + ~2.7ms dequant + ~0.5ms last-scale ≈ 7ms`; quantization adds
  ~85% overhead. This is the real throughput thief.
- Dequant is expensive because it is a multi-kernel int32→unpack→×scale→fp16
  path that writes ~105 MB of fp16 K/V (cached K+V) to HBM and has attention
  re-read it.
- Conservative projection: fusing dequant into the attention K/V tile loop
  removes most of B → heaviest step **~7ms → ~4–4.5ms (~1.6×)**. B will not
  vanish entirely (packed bytes 13–53 MB still read from HBM), but the fp16
  materialization + re-read is eliminated.

## Goal

A fused kernel where cached KV dequant happens **inside** the FA2 K/V tile
loop: read packed int32 + per-group scale, dequant in SRAM, never materialize
fp16 K/V to HBM. Recover the dequant tax as throughput while keeping the memory
saving.

## Layout facts (confirmed in `VAR_Q/quant.py`)

- **Bit-packing is along the last dim (head_dim `c`)** — `_pack_last_dim_to_int32`
  / `unpack_dequant_last_dim_from_int32`. `dim_cat` (seq dim: 1 for BLHc, 2 for
  BHLc) governs only seq concat/caching, **not** packing.
- Therefore the **packed int32 layout is identical for BHLc and BLHc**; the two
  formats differ only in outer strides (which of dims 1/2 is seq vs head).
  Supporting both in the kernel is a stride concern, not extra arithmetic.
- **VARQ scale granularity:** scale reduces over the seq dim → one scale vector
  per `(step, head, head_dim channel)`, broadcast across tokens within a step.
  Along head_dim the scale varies per channel; along seq it is piecewise
  constant per step (12 cached steps → 12 scale vectors). Kernel-friendly.
- q8: 4 int8 per int32, exact division — simplest packing case.

## Approach

### Kernel framework: Triton primary, CUTLASS as conditional Phase B

To fuse, we must own the attention kernel (`flash_attn_func` accepts only fp16
K/V). The choice is the framework to write it in.

**Triton is the primary choice** for v1:
- The dequant logic (int32 → 4×int8 → ×scale → fp16 in SRAM) is a few lines in
  Triton, awkward in CuTe/CUTLASS templates.
- Forward-only fp16 attention on Ampere at D=128, non-causal, Lq≠Lkv is
  Triton's sweet spot (Triton lags mainly on backward, which we do not need).
- Days to a runnable PoC vs weeks for a CUTLASS Ampere fork (no TMA on Ampere →
  hand-rolled `cp.async`).
- The repo already ships a Triton unpack/dequant kernel to build on.

**CUTLASS / CuTe is Phase B, only if validated as necessary:** if the PoC shows
the Triton attention portion is >15–20% slower than `flash_attn_func` *and* that
gap is the bottleneck, fork the flash_attn Ampere CUTLASS kernel and inject
dequant into its `cp.async` K/V load pipeline to recover full attention perf.
Otherwise the Triton kernel is the final version.

**De-risking the framework question:** the plan measures `Triton-fused` vs
`Triton-plain-attention` (same framework) to isolate the dequant-fusion net win
from the Triton-vs-CUTLASS framework gap. This directly answers "does Triton's
own attention overhead eat the fusion win."

### Kernel: `fused_dequant_attention` (Triton FA2 forward)

KV is **heterogeneous, two segments**, handled in a single online-softmax pass
(no two-call merge):

- **Cached segment** (first 12 scales, 6425 tokens): packed int32 + compact
  scale `[num_steps, H, D]` + `step_boundaries`. Tiles in this segment:
  read int32 → SRAM unpack ×scale → fp16 → QK^T / PV.
- **Fresh segment** (last scale, 4096 tokens): raw fp16 (the trick skips
  caching it). Tiles here: normal fp16 load.

The boundary (6425) is known and tile-aligned; FA2's online softmax streams over
KV tiles naturally, so a single softmax normalization spans both segments.

**Kernel interface (draft):**
- `q`: fp16 `[B, Lq, H, D]` (Lq = 4096)
- cached K, cached V: packed `int32`, plus compact scale `[num_steps, H, D]`,
  plus `step_boundaries` (token → step segment map)
- fresh K, fresh V: fp16 `[B, Lfresh, H, D]` (Lfresh = 4096)
- BHLc vs BLHc selected by passing the right strides (packed layout identical)

### Integration

- Gate on the existing dead flag `enable_fused_kv_flashattn` (read into the
  config dict in `hooks/runtime.py` but currently unused). Off → fall back to
  the current `flash_attn_func` path. No behavior change when off.
- Change is localized to the attention branch of `_wrap_infinity_forward`.

### Correctness

- Kernel output must match the fp16 reference
  `dequant_all() + flash_attn_func` on random tensors, fp16 tolerance.
- Reuse the numerical-test framework already pushed in
  `fix/varq-review-2026q2`.

## Scope (v1)

**In:** q8 · VARQ · {BHLc, BLHc} · forward-only inference · A100 / CUDA ·
Triton kernel · two-segment (cached packed + fresh fp16) heaviest-step path ·
gate via `enable_fused_kv_flashattn` · correctness vs fp16 reference ·
apples-to-apples Triton-fused vs Triton-plain measurement.

**Out (deferred until ~1.6× is validated):** q4/q3/q2 and non-divisible bit
widths · the other 6 grouping methods (G_TENSOR, G_SCALE_HEAD_DIM, G_HEAD_DIM,
G_SCALE, G_TOKEN, G_TOKEN_HEAD) · backward / training · non-Infinity models ·
CUTLASS/CuTe port (Phase B, conditional on PoC result).

## Risks

1. Writing FA2 in Triton + SRAM bit-unpack has debugging cost. Mitigation:
   stage it — first a Triton FA2 that matches `flash_attn_func`, then inject
   dequant.
2. Two-segment tile boundary handling (tiles must align to the cached/fresh
   split). Mitigation: pad/align the boundary to BLOCK_N; the fresh segment can
   start a fresh tile.
3. q8 (4-per-int32) divides evenly; q3 and other non-divisible widths are
   harder — explicitly deferred.
4. Triton attention may not fully match CUTLASS FA2 perf. Mitigation: the
   isolated Triton-fused vs Triton-plain measurement keeps the fusion verdict
   valid regardless, and Phase B (CUTLASS) exists if the gap matters.

## Validation plan

- Microbench already establishes the dequant tax (B) and attention floor (C).
- After the kernel: measure `Triton-fused` end-to-end vs both
  `flash_attn_func + dequant_all` (current production) and `Triton-plain + dequant`
  (framework-isolated), on A100, 1 card, synthetic tensors (no weights).
- Gate the throughput claim on the measured numbers, not projections.

## Measured fused result

Run: `python scripts/bench/microbench_fused_vs_baseline.py` through
`scripts/bench/remote_test.sh` on NVIDIA A100 80GB PCIe, torch
`2.7.0a0+79aa17489c.nv25.04`, flash-attn available. Correctness gate: PASS.

| pipeline | total_ms | speedup_vs_production |
|----------|----------|-----------------------|
| production | 7.448 | 1.000x |
| fused | 153.202 | 0.049x |
| plain-isolated | 59.680 | 0.125x |

The projected ~1.6x throughput win was not met; the fused path is 20.6x slower
than production. The production-vs-plain-isolated result shows the current
Triton attention path is already about 8.0x slower than the production
FlashAttention/CUTLASS path, far beyond the 15-20% Phase B threshold. The
fused-vs-plain-isolated comparison is also negative (`153.202 ms` vs
`59.680 ms`), so this Triton fused implementation does not yet isolate a
positive dequant-fusion delta. Phase B is warranted if the fused path is still
worth pursuing for throughput.

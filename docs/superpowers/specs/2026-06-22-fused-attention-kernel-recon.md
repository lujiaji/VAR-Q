# Fused Quantized-KV Attention Kernel — Build-vs-Adopt Recon

**Date:** 2026-06-22
**Status:** Recon / recommendation (no code changes)
**Target kernel:** A100 (Ampere, sm80), fp16 Q, forward-only inference. PTQ KV-cache quant.
Heaviest step: `Q=[1,4096,32,128]` attends non-causally to `KV=[1,10521,32,128]`.
**Our KV format:** int8 (q8), packed 4-per-int32 along **head_dim**, scale **per-(step, head, head_dim-channel)** fp16 — i.e. PER-CHANNEL along head_dim, piecewise-constant per step along sequence. Two segments: 6425 cached/packed tokens + 4096 fresh fp16 ("last-scale trick").
**Perf context:** production (dequant→flash) = 7.45 ms; flash-attn alone on this shape = 5.2 ms; realistic fused ceiling ~1.35x. The repo's current Triton fused PoC measured **20x SLOWER** than production (the Triton attention itself is ~8x slower than CUTLASS flash on this shape) — so "stay Triton" is not currently viable without major Triton-perf work.

---

## Why our format is the load-bearing constraint

Every candidate's scale axis is the **orthogonal** of ours. The whole quantized-attention ecosystem reduces absmax **over** head_dim to get one scale per (token, head) — a *per-token* scalar. We keep a scale vector **per channel along** head_dim, constant per step along tokens. That single mismatch (plus our 4-int8-per-int32 packing and int8 bit-width, where the fused world is overwhelmingly int4) is what rules out drop-in reuse across the board.

---

## TRACK A — Existing fused quantized-KV attention kernels

| Project | Fuses dequant into attn? | KV quant format | Scale granularity | sm80 / A100? | Per-channel (head_dim) int8? | License | Adoptable for our per-(step,head,channel) int8 packed KV? |
|---|---|---|---|---|---|---|---|
| **FlashInfer** (fp8 / low-prec KV, BatchPrefill/append) | Yes (fp8 dequant in-kernel) | **fp8 (e4m3/e5m2), NVFP4** — no int8 | per-tensor / per-head; NVFP4 = fixed 16-elem head_dim blocks | fp8 attn path **Hopper-gated (sm90a)** in practice; could not confirm working fp8-KV on sm80 | No | Apache-2.0 | **No** — no int8; would re-quantize to fp8/per-head and likely won't run fused on A100 |
| **flash-attn v3 fp8** (Dao-AILab) | Descales fp8 GEMM operands (not a stored KV cache) | fp8 GEMM inputs only | per-(batch, kv_head) descale | **sm90 only** (FP8 needs H100) | No | BSD-3 | **No** — Hopper-only; no stored-quant-KV concept |
| **flash_attn_with_kvcache** | No (fp16/bf16 KV only) | fp16/bf16 | n/a | Yes (fp16/bf16) | No | BSD-3 | **No** — must pre-dequant to fp16 (defeats the goal) |
| **KIVI** (jy-yuan/KIVI) | Yes, but **CUDA-core GEMV**, not Tensor-Core flash | int2 / int4 fused kernels (8-bit packs but **no 8-bit fused kernel**) | per-group G=32, **asymmetric** (scale+zero); **K is per-channel** | Inferred sm80+ (`-DENABLE_BF16`); not documented | Closest (K per-channel) but 2/4-bit + per-group, not per-single-channel | MIT | **No** — no 8-bit fused GEMV; per-group+asymmetric; CUDA-core not TC. Best *conceptual* reference for per-channel-K |
| **QServe / QoQ** (mit-han-lab/omniserve) | Yes (KV4 + a KV8 path; int4 dequant in SRAM) | int4 (KV4); separate KV8 | per-token + per-head, asymmetric | **Yes, explicit** (`SUPPORTED_ARCHS={8.0,8.6,8.7,8.9,9.0}`, A100-benched) | No (per-token/per-head) | Apache-2.0 | **No** as-is — KV8 path closest, but per-token/per-head scale + different packing. **Best sm80 reference** |
| **Atom** (efeslab/Atom) | Yes (FlashInfer-derived) | int4, asymmetric | per-head | **No** — "kernels only optimized for RTX4090" (sm89) | No | **None (no LICENSE file)** | **No** — int4, RTX4090-only, legally unusable |
| **QuaRot** (spcl/QuaRot) | Wraps **FlashInfer's** kernel; owns only a CUTLASS W4A4 GEMM | int4, group-128, asymmetric (FlashInfer paged layout) | per-group 128 | GEMM tagged Sm80; attn = FlashInfer; sm80 run unverified | No | Apache-2.0 | **No** — int4 + paged layout; target FlashInfer directly if anything |
| **SageAttention** (thu-ml) | Yes (int8 QK fused) but **quantizes internally every call** | int8 Q/K (V fp16 on sm80) | per-block / per-warp **along tokens** | Yes — `sageattn_qk_int8_pv_fp16_cuda` is sm80 | No (per-block along tokens; per-channel on QK^T contraction axis is mathematically disallowed) | Apache-2.0 | **No** — no entry point for pre-quantized int8 / external scales; wrong scale axis. (This is the `using_sageattn` flag in `hooks/runtime.py` — it's an alternative attention, not a quant-KV ingestor) |
| **vLLM** quantized paged-attn | Yes (`scaled_convert` at K/V load) | fp8 + int8 (`int8_per_token_head`, 2026) | per-tensor / per-head / per-token-head (absmax **over** head_dim) | fp8 emulated on sm80; int8 = Triton path | No | Apache-2.0 | **No** — per-token-head scale, plain uint8 packing, welded to block-tables/paged manager. Reference only |
| **LMDeploy / TurboMind** kv_quant | Yes (`ConvertKvCache` in `Transform()`) | int8 (uint8) / int4 (8-per-uint32) | per-head, per-token, asymmetric | **Yes** (`iterator_sm80.h`, `impl_16816.h`) | No (per-token-head) | Apache-2.0 | **No** — scalar scale across head_dim vector; paged/decode-oriented; no PyTorch binding. Reference only |

**Track A conclusion:** **No project ingests an externally-quantized, per-channel-along-head_dim, int8, 4-per-int32 KV cache and runs fused attention on it.** All fused paths are int4 + per-token/per-head/per-group; the one int8 fused kernel (SageAttention) quantizes internally and uses a per-token axis; KIVI is the lone per-channel-K design but is 2/4-bit CUDA-core GEMV. Best *reference* implementations on sm80: **QServe/omniserve** (Apache-2.0, explicit A100, fused int4/int8, register-level dequant) and **KIVI** (MIT, per-channel-K pattern). vLLM `scaled_convert` and TurboMind `ConvertKvCache` are clean reference patterns for register-level fused fetch+dequant.

---

## TRACK B — Fork flash-attention's CUDA (CUTLASS) Ampere kernel

**Container (`kivi_bench` on A100, verified by inspection — no GPU workload run):**
- 8x A100 80GB PCIe, **compute_cap 8.0** (sm80). GPU 0 in use — not touched.
- nvcc **12.9**, torch **2.7.0a0 (CUDA 12.9)**, **ninja 1.11**, **git 2.43** present.
- `/tmp` (overlay) has **2.3 TB free** — ample for a flash-attention clone + build.
- flash-attn installed is **wheel-only**: `flash_attn_2_cuda.cpython-312*.so` present, **no `.cu`/`csrc` source** in dist-packages. Version **2.7.3**.
- KIVI sources already on disk under `/work/effect-stage1/kivi_*` and `/work/VAR-Q/...` (reference material on the box).

**Source availability:** To get the CUDA source we must `git clone https://github.com/Dao-AILab/flash-attention` (the wheel ships no `.cu`). Standard, low-risk.

**Inject point (located via source, no build):**
- Ampere forward path lives in `csrc/flash_attn/src/flash_fwd_kernel.h`, function `compute_attn_1rowblock`.
- K/V tiles are loaded gmem→smem by `FLASH_NAMESPACE::copy<...>(gmem_tiled_copy_QKV, tKgK, tKsK, ...)` / `tVgV→tVsV`, with `cute::cp_async_fence()` + `cp_async_wait<0>()` forming the double-buffered cp.async pipeline. Copy engine = `Kernel_traits::GmemTiledCopyQKV`; element type `Kernel_traits::Element` (fp16/bf16). The `copy` helper is in `csrc/flash_attn/src/utils.h`; traits in `kernel_traits.h`.
- **Injection:** point `params.k_ptr`/`params.v_ptr` at our packed int32 cache, load packed int32 + per-(step,head,channel) scale into smem, **unpack 4×int8 + multiply per-channel scale → fp16 in smem before `FLASH_NAMESPACE::gemm`**. The per-channel scale aligns naturally with the head_dim (K-dim) tiling; the per-step piecewise scale needs a `step_ids`/boundary lookup per KV tile (our boundaries are tile-alignable). The two-segment (packed cached + fresh fp16) split maps to two ranges of `n_block`.

**Effort / feasibility estimate:**
- Build in-container is feasible (nvcc 12.9 + ninja + 2.3 TB scratch). flash-attn full build is heavy (many head-dim/causal template instantiations, tens of minutes to >1h). **Mitigation: build only the single config we need** (D=128, non-causal, fp16, sm80) by trimming the instantiation list — cuts build to minutes. No GPU needed to compile; only correctness/perf runs need GPU (defer, and only on a free GPU).
- Engineering effort: medium-high. CuTe/CUTLASS template surface is steep, but the change is localized (custom K/V `copy` + dequant before `gemm`, plus a packed-pointer + scale + step-boundary param plumb). The attention math, online softmax, and cp.async pipeline are reused unchanged — we inherit production-grade CUTLASS attention perf, which is exactly what the Triton PoC failed to deliver. Rough order: **~1–2 weeks** to a correct single-config fused kernel for an engineer comfortable with CUTLASS, less if scoped tightly to q8/D128/non-causal.
- Risk: CuTe debugging cost; smem budget for holding scale + unpacked tiles; matching FA's masking. All localized and bounded.

---

## RECOMMENDATION (ranked by effort-vs-payoff)

1. **PRIMARY — Fork flash-attn's CUDA Ampere kernel and inject int8-unpack + per-channel dequant before the GEMM.** This is the only path that both (a) supports our exact per-(step,head,channel) int8 / 4-per-int32 layout and (b) inherits CUTLASS attention perf. It directly addresses why the Triton PoC was 20x slow: the attention engine stays production-grade flash; we only add the dequant on the K/V smem load. Effort medium-high (~1–2 wk, single config), payoff high (only route to the ~1.35x ceiling). Inject point and toolchain are confirmed ready.

2. **SECONDARY / accelerator — port QServe(omniserve)'s KV8 fused attention as the starting skeleton** instead of flash-attn, *if* on inspection its KV8 SRAM-dequant structure is cleaner to retarget than flash-attn's CuTe. It's Apache-2.0, explicitly sm80/A100, already fuses int dequant in SRAM. We'd still rewrite the scale axis (per-token/per-head → per-channel) and packing, but inherit a fused-int-attention scaffold. Use KIVI (MIT, per-channel-K) and vLLM `scaled_convert` / TurboMind `ConvertKvCache` as register-dequant reference patterns. Verify QServe's KV8 packing/group in its CUDA source (unverified here) before committing.

3. **FALLBACK — keep iterating Triton only if a focused Triton-attention-perf spike closes the ~8x gap** to flash on this shape (correct block sizes, num_warps/stages, `tl.dot` fp16 accumulation, pipelining). Lowest engineering ceiling and the current PoC is far off; pursue only if owning CUTLASS is deemed too costly. The dequant-fusion logic the repo already has (`VAR_Q/fused/flash_dequant.py`) is reusable regardless of framework.

**Do NOT** plan to adopt FlashInfer/FA3-fp8 (Hopper-gated, no int8), Atom (RTX4090-only, no license), or SageAttention/vLLM/TurboMind as libraries — all are per-token/per-head scale and/or internal-quant, structurally incompatible with our per-channel int8; they are reference code only.

---

## Uncertainties flagged (not guessed)
- FlashInfer fp8-KV on sm80: could not find a definitive "runs on A100" confirmation; public evidence points to sm90 in practice. Needs a source-level dispatch check if ever pursued (it isn't, since no int8).
- QServe KV8 exact packing layout and KV group size live in unfetched CUDA source — verify before using as skeleton (option 2).
- QuaRot attention-kernel sm80 runtime is build-test-pending (its GEMM is Sm80-tagged but was benched on RTX3090).
- KIVI A100 support is inferred from build flags, not documented.
- flash-attn full-build wall-time in this container is an estimate (tens of min to >1h); the single-config trim is the mitigation and should be measured before scheduling.

## Sources
FlashInfer: https://github.com/flashinfer-ai/flashinfer · https://docs.flashinfer.ai/api/attention.html · https://arxiv.org/pdf/2501.01005
flash-attention: https://github.com/Dao-AILab/flash-attention · https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h · https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py
KIVI: https://github.com/jy-yuan/KIVI · https://arxiv.org/abs/2402.02750
QServe/omniserve: https://github.com/mit-han-lab/omniserve · https://arxiv.org/html/2405.04532v2
Atom: https://github.com/efeslab/Atom · https://arxiv.org/html/2310.19102v3
QuaRot: https://github.com/spcl/QuaRot · https://arxiv.org/html/2404.00456v2
SageAttention: https://github.com/thu-ml/SageAttention · https://arxiv.org/html/2410.02367v5
vLLM: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/ · https://github.com/vllm-project/vllm
LMDeploy/TurboMind: https://github.com/InternLM/lmdeploy/blob/main/docs/en/quantization/kv_quant.md

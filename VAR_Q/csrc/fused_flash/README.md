# VAR-Q Fused FlashAttention CUDA Backend

This directory contains the CUDA-only VAR-Q runtime and the optional
FlashAttention-2 Ampere forward specialization.

Runtime scope:

- standalone `quantize_pack`, `pack_int8`, `unpack_int8`, and
  `unpack_dequant` CUDA APIs for Q2/Q3/Q4/Q6/Q8
- Q3 uses 10 values per int32 word (30 payload bits)
- FP16/BF16/FP32 input, scale, and dequant output
- optional caller-provided `out` workspace for `unpack_dequant`
- BLHc/BHLc 4-D broadcast scales and optional token-to-scale group ids
- non-causal packed-KV attention for head_dim 64 and 128
- optimized sm80 fp16/head_dim=128 direct loader for Q2/Q3/Q4/Q8, with a
  generic packed CUDA path for Q6, head_dim=64, and BF16/FP32
- two-segment KV: cached packed KV followed by fresh floating-point KV

Upstream source:

- Project: Dao-AILab flash-attention
- Version targeted: v2.7.3
- License: BSD-3-Clause. Preserve upstream copyright/license headers in any
  vendored or patched source files.

Build strategy:

- Keep VAR-Q code changes local in this repository.
- Build with `scripts/bench/build_fused_flash.sh`. By default it reads an
  upstream checkout from `third_party/flash-attention`; set
  `FLASH_ATTN_SOURCE` to use another local checkout.
- `fwd` preserves the PR15 FP16/BF16 dense FlashAttention bridge and its
  explicit `softmax_scale` ABI where supported.
- `fwd_direct` dispatches the optimized sm80 Q2/Q3/Q4/Q8 loader for
  FP16/head_dim=128. Q6, head_dim=64, and BF16/FP32 use the generic packed-KV
  CUDA kernel without materializing a full dense historical cache.

Primary injection point:

- `csrc/flash_attn/src/flash_fwd_kernel.h`
- Function: `compute_attn_1rowblock`
- K/V gmem-to-smem load sites:
  `FLASH_NAMESPACE::copy(gmem_tiled_copy_QKV, tKgK, tKsK, ...)` and the
  corresponding V copy.

The optimized direct dequant strategy is unpack-to-smem before GEMM: load packed int32
plus per-(step, head, channel) scale, write an fp16 K/V tile in smem, and leave
FlashAttention's MMA and online softmax unchanged. Unsupported dtype/shape
combinations use the same explicit packed-format descriptor through the generic
CUDA path instead of a model-specific kernel.

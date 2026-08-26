# VAR-Q Fused FlashAttention CUDA Backend

This directory is the local home for Track B: a FlashAttention-2 Ampere forward
kernel modified to read VAR-Q packed q8 KV cache directly.

Scope for v1:

- A100 / sm80 only
- fp16, head_dim=128, non-causal forward inference
- VARQ q8 cache, packed 4 int8 values per int32 along head_dim
- BHLc tensors
- two-segment KV: cached packed KV followed by fresh fp16 last-scale KV

Upstream source:

- Project: Dao-AILab flash-attention
- Version targeted first: v2.7.3, matching the A100 `kivi_bench` wheel
- License: BSD-3-Clause. Preserve upstream copyright/license headers in any
  vendored or patched source files.

Build strategy:

- Keep VAR-Q code changes local in this repository.
- Build and validate on A100 by syncing this repository with
  `scripts/bench/remote_test.sh`.
- The build script may cache an upstream flash-attention checkout outside the
  repository on the remote host, then apply VAR-Q patch/source files from this
  directory.
- Current default implementation status: dense bridge. The extension dequants
  cached q8 K/V into temporary BLHc tensors, appends fresh fp16 K/V, and calls
  flash-attn's sm80 fp16 head_dim=128 forward.
- Experimental direct status: `fwd_direct` / `backend="cuda-direct"` patches
  the K/V gmem-to-smem load sites and passes correctness tests, but it uses
  scalar q8/fresh loads in the FlashAttention tile loop and is currently much
  slower than the dense bridge. Keep `backend="cuda"` as the performance path
  until the direct loader is vectorized and scale reuse is improved.

Primary injection point:

- `csrc/flash_attn/src/flash_fwd_kernel.h`
- Function: `compute_attn_1rowblock`
- K/V gmem-to-smem load sites:
  `FLASH_NAMESPACE::copy(gmem_tiled_copy_QKV, tKgK, tKsK, ...)` and the
  corresponding V copy.

The direct v1 dequant strategy is unpack-to-smem before GEMM: load packed int32
plus per-(step, head, channel) scale, write an fp16 K/V tile in smem, and leave
FlashAttention's MMA and online softmax unchanged. The current implementation
does not yet preserve the optimized cp.async/vectorized global copy path.

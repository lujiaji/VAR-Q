# Fused dequant-attention — known issues / limitations

The fused KV dequant+attention path (`enable_fused_kv_flashattn`, **off by
default**) is experimental. A high-effort code review (2026-06-25) surfaced the
issues below. The **softmax-scale bug was fixed** (it was active for Infinity);
the rest are documented here because they either don't trigger on the default
Infinity-8B path or are deliberate design trade-offs.

## Fixed
- **Softmax scale ignored `self.scale`.** The fused Triton + CUDA kernels
  hardcoded `1/sqrt(head_dim)`, but Infinity runs `cos_attn` (always True) where
  `self.scale = 1`. The fused path therefore mis-scaled the softmax ~11x and
  produced wrong attention for Infinity. Fixed by threading `softmax_scale`
  (= the module's `self.scale`) from `runtime.py` → `fused_dequant_attention` →
  both backends, overriding `params.scale_softmax*` in the CUDA kernels. Default
  (no scale passed) still falls back to `1/sqrt(head_dim)` for back-compat.

## Open (documented, not yet fixed)
- **Attention masking is dropped.** The non-fused path forwards
  `attn_bias_or_two_vector` (`VAR_visible_kvlen` / `VAR_invisible_qlen`) to
  `flash_attn_func`; the fused two-segment kernel attends to all cached+fresh KV
  unmasked. Only correct when the last AR step needs no partial visibility mask.
  Re-establishing this needs masking support in the fused kernel.
- **Fresh (current-step) K/V are used at full fp16**, not round-tripped through
  the quantizer like the non-fused `cache_current=False` path. Numerically this
  diverges from the quantized baseline on the last step (arguably *more* accurate,
  but not bit-matching the baseline an acceptance test may expect).
- **Cold cache.** `fused_dequant_attention` calls `_valid_cached_item()`, which
  raises if no prior scale step populated the cache. The gate only checks
  `is_last_scale`, which is also true for a length-1 scale schedule. Does not
  trigger on Infinity-8B (13 scales) but would crash a single-scale config.
- **`_scale_strides` axis heuristic.** Disambiguates the head axis by
  `shape[1]==H`; for a BLHc compact scale where `num_steps == H` it mis-identifies
  the step axis as the head axis. Edge case (steps rarely equal head count).
- **`step_ids` bound.** The dequant kernels index `step_ids` over the cached
  token range; if `_scale_L_counts` ever under/over-covers `cached_len` this can
  read out of bounds. Holds as long as the COMPACT scale-group counts sum to the
  cached length (the invariant the quantizer maintains).

## Efficiency (by design for v1, not bugs)
- The **dense-bridge** CUDA path (`backend="cuda"`, the non-direct default)
  materializes the full fp16 K/V cache every step before running dense
  FlashAttention — it does not stream tiles. Use `cuda-direct` to avoid this.
- `_valid_cached_item()` returns a non-contiguous prefix slice of a
  capacity-doubled buffer; `fused_dequant_attention` calls `.contiguous()` on it
  each step, copying the whole packed cache. Cheaper would be an exactly-sized /
  contiguous cache buffer.

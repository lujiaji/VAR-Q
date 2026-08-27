# Fused packed-KV attention limitations

- The optimized kernel is specialized for NVIDIA Ampere (`sm80`), FP16, and
  head dimension 128. Other supported dtypes, head dimension 64, and INT6 use
  the generic packed CUDA kernel.
- INT2/3/4/6/8 use one packed layout contract. Producers and consumers must
  preserve the exact bit width, token-to-scale group map, and K/V layout.
- Self-Forcing and LongLive launchers default to strict fused
  execution. A missing extension or unsupported runtime shape raises an error
  instead of silently restoring the full historical KV cache.

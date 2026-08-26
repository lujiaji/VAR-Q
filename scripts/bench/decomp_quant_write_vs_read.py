"""Decompose quant-on-write versus dequantization latency."""
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.bench.microbench_dequant_vs_fa2 import PATCH
LAST_PATCH = PATCH[-1]
from VAR_Q.quant import VAR_Q

BATCH, HEADS, HEAD_DIM = 1, 32, 128
QKV_FORMAT = "BHLc"
CACHE_PATCH = list(PATCH[:-1])
FRESH_TOKENS = LAST_PATCH * LAST_PATCH
CACHE_TOKENS = sum(p*p for p in CACHE_PATCH)
dev = torch.device("cuda")

def mk(role): return VAR_Q(quant_bits=8, qkv_format=QKV_FORMAT, quant_method="VARQ", kv_role=role, pack_to_int32=True, dequant_dtype="fp16")
def rnd(t): return torch.randn((BATCH, HEADS, t, HEAD_DIM), device=dev, dtype=torch.float16)

def timed(fn, iters=100, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)

kq = mk("k")
for p in CACHE_PATCH:
    kq.use_var_q(rnd(p*p), cache_current=True)
print(f"cached_len={kq.cached_len} fresh={FRESH_TOKENS}")

fresh = rnd(FRESH_TOKENS)
def quant_write():
    kq.quant(fresh)
quant_ms = timed(quant_write)

def deq():
    kq.dequant_all()
deq_ms = timed(deq)

print(f"[decomp] quant-on-write (quant+scale) = {quant_ms:.3f} ms")
print(f"[decomp] dequant_all (read path)      = {deq_ms:.3f} ms")
print(f"[decomp] ratio write/read = {quant_ms/deq_ms:.2f}")

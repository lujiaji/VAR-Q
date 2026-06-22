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

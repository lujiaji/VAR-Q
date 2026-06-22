import pytest
import torch

from tests.fused import oracle


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


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

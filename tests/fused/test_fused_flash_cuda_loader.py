import pytest
import torch

from VAR_Q.fused import flash_dequant_cuda
from tests.fused import oracle


def _build_case(device: str):
    B, H, D, bits, fmt = 1, 2, 128, 8, "BHLc"
    torch.manual_seed(0)
    kq, ref_k = oracle.build_varq_cache([1, 2], B, H, D, bits, fmt, device, "k")
    vq, ref_v = oracle.build_varq_cache([1, 2], B, H, D, bits, fmt, device, "v")
    fk = oracle.make_kv_tensor(B, H, 4, D, fmt, device)
    fv = oracle.make_kv_tensor(B, H, 4, D, fmt, device)
    q = oracle.make_kv_tensor(B, H, 4, D, fmt, device)
    kp = oracle.extract_packed(kq)
    vp = oracle.extract_packed(vq)
    step_ids = oracle.step_ids_from_groups(kp["group_lengths"], device)
    ref = oracle.ref_attention(q, torch.cat([ref_k, fk], dim=2), torch.cat([ref_v, fv], dim=2), fmt)
    return q, kp, vp, step_ids, fk, fv, ref


def test_fused_flash_cuda_backend_status_object():
    status = flash_dequant_cuda.availability()
    assert isinstance(status.available, bool)
    assert isinstance(status.message, str)


@pytest.mark.skipif(
    not flash_dequant_cuda.availability().available,
    reason="VAR-Q fused FlashAttention CUDA extension is not built",
)
def test_fused_flash_cuda_extension_exports_fwd():
    ext = flash_dequant_cuda.load_extension()
    assert callable(getattr(ext, "fwd", None))
    assert callable(getattr(ext, "fwd_direct", None))
    assert "Track B" in ext.backend_info()


@pytest.mark.skipif(
    not flash_dequant_cuda.availability().available or not torch.cuda.is_available(),
    reason="VAR-Q fused FlashAttention CUDA extension is not built on CUDA",
)
def test_fused_flash_cuda_dense_bridge_matches_oracle():
    device = "cuda"
    q, kp, vp, step_ids, fk, fv, ref = _build_case(device)

    out = flash_dequant_cuda.fused_flash_dequant_attention(
        q, kp["packed"], vp["packed"], kp["scale"], vp["scale"], step_ids, fk, fv
    )
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not flash_dequant_cuda.availability().available or not torch.cuda.is_available(),
    reason="VAR-Q fused FlashAttention CUDA extension is not built on CUDA",
)
def test_fused_flash_cuda_direct_matches_oracle():
    device = "cuda"
    q, kp, vp, step_ids, fk, fv, ref = _build_case(device)
    ext = flash_dequant_cuda.load_extension()

    out = ext.fwd_direct(
        q, kp["packed"], vp["packed"], kp["scale"], vp["scale"], step_ids, fk, fv,
        float(q.shape[-1]) ** -0.5,  # softmax_scale = 1/sqrt(head_dim), matching the oracle
    )
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)

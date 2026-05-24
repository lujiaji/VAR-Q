import pytest
import torch

from VAR_Q.pack_unpack import (
    _HAS_TRITON,
    pack_last_dim_to_int32_python,
    pack_last_dim_to_int32_triton,
    unpack_dequant_last_dim_from_int32_triton,
    unpack_last_dim_from_int32_python,
    unpack_last_dim_from_int32_triton,
)


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("last_dim", [7, 16, 33])
def test_pack_unpack_roundtrip_python(bits, last_dim):
    low = -(2 ** (bits - 1))
    high = 2 ** (bits - 1)
    x = torch.randint(low, high, (2, 3, last_dim), dtype=torch.int8)

    packed, meta = pack_last_dim_to_int32_python(x, bits)
    unpacked = unpack_last_dim_from_int32_python(packed, meta)

    assert packed.dtype == torch.int32
    assert unpacked.dtype == torch.int8
    assert unpacked.shape == x.shape
    assert torch.equal(unpacked, x)


def test_pack_unpack_preserves_leading_dimensions():
    x = torch.randint(-8, 8, (2, 4, 5, 9), dtype=torch.int8)
    packed, meta = pack_last_dim_to_int32_python(x, bits=4)
    unpacked = unpack_last_dim_from_int32_python(packed, meta)

    assert packed.shape[:-1] == x.shape[:-1]
    assert unpacked.shape == x.shape
    assert torch.equal(unpacked, x)


@pytest.mark.skipif(not torch.cuda.is_available() or not _HAS_TRITON, reason="CUDA/Triton not available")
@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_triton_pack_unpack_and_fused_dequant(bits):
    low = -(2 ** (bits - 1))
    high = 2 ** (bits - 1)
    x = torch.randint(low, high, (2, 5, 3, 17), dtype=torch.int8, device="cuda")

    packed, meta = pack_last_dim_to_int32_triton(x, bits)
    unpacked = unpack_last_dim_from_int32_triton(packed, meta)

    assert torch.equal(unpacked, x)

    scale = torch.rand((2, 1, 3, 17), device="cuda", dtype=torch.bfloat16) + 0.01
    out = torch.empty_like(x, dtype=torch.bfloat16)
    unpack_dequant_last_dim_from_int32_triton(packed, scale, meta, out, "BLHc")

    torch.testing.assert_close(out, (x.to(torch.bfloat16) * scale).to(torch.bfloat16), atol=0, rtol=0)

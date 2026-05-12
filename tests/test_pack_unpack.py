import pytest
import torch

from VAR_Q.pack_unpack import (
    pack_last_dim_to_int32_python,
    unpack_last_dim_from_int32_python,
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

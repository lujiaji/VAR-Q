import pytest
import torch

from VAR_Q.quant import VAR_Q


METHODS = (
    "VARQ",
    "G_TENSOR",
    "G_SCALE_HEAD_DIM",
    "G_HEAD_DIM",
    "G_SCALE",
    "G_TOKEN",
    "G_TOKEN_HEAD",
)
BITS = (2, 3, 4, 6, 8)
QKV_FORMATS = ("BLHc", "BHLc")
PACK_TO_INT32 = (True, False)


def _seq_dim(qkv_format: str) -> int:
    return 1 if qkv_format == "BLHc" else 2


def _shape(qkv_format: str, seq_len: int) -> tuple[int, int, int, int]:
    if qkv_format == "BLHc":
        return (2, seq_len, 4, 64)
    return (2, 4, seq_len, 64)


def _seed(method: str, bits: int, qkv_format: str, pack_to_int32: bool, offset: int) -> int:
    return (
        offset
        + METHODS.index(method) * 1_000
        + bits * 100
        + QKV_FORMATS.index(qkv_format) * 10
        + int(pack_to_int32)
    )


def _randn(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _quantizer(method: str, bits: int, qkv_format: str, pack_to_int32: bool) -> VAR_Q:
    return VAR_Q(
        quant_bits=bits,
        qkv_format=qkv_format,
        quant_method=method,
        pack_to_int32=pack_to_int32,
        dequant_dtype="fp32",
    )


def _assert_roundtrip_error(actual: torch.Tensor, expected: torch.Tensor, bits: int, tol: float) -> None:
    max_abs_err = (expected - actual).abs().max()
    quant_step = expected.abs().max() / (2 ** (bits - 1))
    assert max_abs_err <= quant_step * tol, (
        f"max_abs_err={max_abs_err.item()} quant_step={quant_step.item()} tol={tol}"
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("qkv_format", QKV_FORMATS)
@pytest.mark.parametrize("pack_to_int32", PACK_TO_INT32)
def test_quant_single_step_numerical(method, bits, qkv_format, pack_to_int32):
    x = _randn(_shape(qkv_format, seq_len=16), _seed(method, bits, qkv_format, pack_to_int32, 10_000))
    q = _quantizer(method, bits, qkv_format, pack_to_int32)

    out = q.use_var_q(x, cache_current=False)

    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    _assert_roundtrip_error(out, x, bits, tol=1.2)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("qkv_format", QKV_FORMATS)
@pytest.mark.parametrize("pack_to_int32", PACK_TO_INT32)
def test_quant_autoregressive_cache_numerical(method, bits, qkv_format, pack_to_int32):
    q = _quantizer(method, bits, qkv_format, pack_to_int32)
    seed = _seed(method, bits, qkv_format, pack_to_int32, 3_000)
    x_list = [
        _randn(_shape(qkv_format, seq_len=3), seed + step)
        for step in range(5)
    ]

    for x in x_list:
        q.quant_and_cache(x)
    out = q.dequant_all()
    truth = torch.cat(x_list, dim=_seq_dim(qkv_format))

    assert out.shape == truth.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    tol = 1.5 if method in ("G_HEAD_DIM", "G_TENSOR") else 1.2
    _assert_roundtrip_error(out, truth, bits, tol=tol)

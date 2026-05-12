import pytest
import torch

from VAR_Q.quant import dequantize_tensor, quantize_tensor


@pytest.mark.parametrize("qkv_format,shape", [("BLHc", (2, 5, 3, 4)), ("BHLc", (2, 3, 5, 4))])
@pytest.mark.parametrize("pack_to_int32", [False, True])
def test_quant_dequant_shape_roundtrip(qkv_format, shape, pack_to_int32):
    x = torch.randn(shape, dtype=torch.float32)
    quantized = quantize_tensor(
        x,
        quant_bits=4,
        qkv_format=qkv_format,
        quant_method="VARQ",
        pack_to_int32=pack_to_int32,
        dequant_dtype="fp32",
    )
    y = dequantize_tensor(
        quantized["packed"],
        quantized["scale"],
        quantized["pack_meta"],
        dequant_dtype="fp32",
        quant_meta=quantized["quant_meta"],
    )

    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_quantizer_cache_shape_with_compression_ratio():
    x1 = torch.randn(1, 4, 2, 3)
    x2 = torch.randn(1, 4, 2, 3)
    q1 = quantize_tensor(
        x1,
        quant_bits=4,
        qkv_format="BLHc",
        quant_method="VARQ",
        pack_to_int32=True,
        dequant_dtype="fp32",
        compression_ratio=0.5,
        max_scale_seq_len=4,
    )
    y1 = dequantize_tensor(
        q1["packed"],
        q1["scale"],
        q1["pack_meta"],
        dequant_dtype="fp32",
        quant_meta=q1["quant_meta"],
    )

    assert y1.shape == x1.shape
    assert torch.isfinite(y1).all()

    q2 = quantize_tensor(
        x2,
        quant_bits=3,
        qkv_format="BLHc",
        quant_method="G_HEAD_DIM",
        pack_to_int32=True,
        dequant_dtype="fp32",
    )
    y2 = dequantize_tensor(
        q2["packed"],
        q2["scale"],
        q2["pack_meta"],
        dequant_dtype="fp32",
        quant_meta=q2["quant_meta"],
    )
    assert y2.shape == x2.shape

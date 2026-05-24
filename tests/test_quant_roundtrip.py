import pytest
import torch

from VAR_Q.quant import InfinityStarVARQ, dequantize_tensor, quantize_tensor


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


def test_dequantize_tensor_writes_to_preallocated_output():
    x = torch.randn(2, 5, 3, 8, dtype=torch.bfloat16)
    q = quantize_tensor(
        x,
        quant_bits=4,
        qkv_format="BLHc",
        quant_method="VARQ",
        pack_to_int32=True,
        dequant_dtype="bf16",
    )
    out = torch.empty_like(x)
    y = dequantize_tensor(
        q["packed"],
        q["scale"],
        q["pack_meta"],
        dequant_dtype="bf16",
        quant_meta=q["quant_meta"],
        out=out,
    )

    assert y.data_ptr() == out.data_ptr()
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_infinitystar_selected_dequantizes_into_final_output():
    cache = InfinityStarVARQ(
        quant_bits=4,
        qkv_format="BHLc",
        quant_method="VARQ",
        dequant_dtype="bf16",
    )
    first = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
    second = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    current = torch.randn(1, 2, 2, 8, dtype=torch.bfloat16)
    cache.cache_scale(0, first, return_dequant=False)
    cache.cache_scale(1, second, return_dequant=False)

    selected = cache.get_selected([0, 1], current_kv=current, cat_dim=2)

    assert selected.shape == (1, 2, 9, 8)
    assert cache._scale_quantizers[0]._dequant_workspace is None
    assert cache._scale_quantizers[1]._dequant_workspace is None

    text_ref = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)
    mixed = cache.materialize_selected([0, text_ref, 1, current], cat_dim=2)
    assert mixed.shape == (1, 2, 10, 8)
    assert cache._scale_quantizers[0]._dequant_workspace is None
    assert cache._scale_quantizers[1]._dequant_workspace is None

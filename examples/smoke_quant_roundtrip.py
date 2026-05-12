#!/usr/bin/env python3
import torch

from VAR_Q.quant import dequantize_tensor, quantize_tensor


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 8, 4, 16, device=device)

    quantized = quantize_tensor(
        x,
        quant_bits=4,
        qkv_format="BLHc",
        quant_method="VARQ",
        pack_to_int32=True,
        dequant_dtype="fp32",
    )
    y = dequantize_tensor(
        quantized["packed"],
        quantized["scale"],
        quantized["pack_meta"],
        dequant_dtype="fp32",
        quant_meta=quantized["quant_meta"],
    )

    if y.shape != x.shape:
        raise RuntimeError(f"unexpected dequant shape: {tuple(y.shape)} != {tuple(x.shape)}")
    if not torch.isfinite(y).all():
        raise RuntimeError("dequantized tensor contains non-finite values")

    max_abs_err = (x - y).abs().max().item()
    print(f"quant roundtrip smoke passed on {device}: shape={tuple(y.shape)}, max_abs_err={max_abs_err:.4f}")


if __name__ == "__main__":
    main()

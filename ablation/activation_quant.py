"""Reference activation quantization used by the optional W4A8 ablation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch


@dataclass(frozen=True)
class ActivationQuantizationConfig:
    """Dynamic signed-A8 fake-quantization configuration.

    The output is dequantized back to the input floating dtype so an
    unmodified model can execute the numerical W4A8 reference path.
    """

    bits: int = 8
    granularity: str = "per_token"
    clip_ratio: float = 1.0
    clip_method: str = "absmax"
    clip_percentile: float = 1.0
    eps: float = 1e-12

    def validate(self) -> None:
        if self.bits != 8:
            raise ValueError(f"The W4A8 reference path requires A8, got A{self.bits}")
        if self.granularity not in {"per_token", "per_tensor"}:
            raise ValueError("granularity must be 'per_token' or 'per_tensor'")
        if not 0.0 < float(self.clip_ratio) <= 1.0:
            raise ValueError("clip_ratio must be in (0, 1]")
        if self.clip_method not in {"absmax", "percentile"}:
            raise ValueError("clip_method must be 'absmax' or 'percentile'")
        if not 0.0 < float(self.clip_percentile) <= 1.0:
            raise ValueError("clip_percentile must be in (0, 1]")
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be positive")


def _reduction_dims(tensor: torch.Tensor, granularity: str) -> Tuple[int, ...]:
    if granularity == "per_token":
        return (tensor.ndim - 1,)
    return tuple(range(tensor.ndim))


def _clip_threshold(
    tensor: torch.Tensor,
    config: ActivationQuantizationConfig,
    absmax: torch.Tensor,
) -> torch.Tensor:
    if config.clip_method == "absmax":
        return absmax * float(config.clip_ratio)
    magnitudes = tensor.abs()
    if config.granularity == "per_token":
        count = magnitudes.shape[-1]
        kth = max(1, min(count, int(math.ceil(config.clip_percentile * count))))
        threshold = torch.kthvalue(magnitudes, kth, dim=-1, keepdim=True).values
    else:
        flattened = magnitudes.reshape(-1)
        count = flattened.numel()
        kth = max(1, min(count, int(math.ceil(config.clip_percentile * count))))
        threshold = torch.kthvalue(flattened, kth).values.reshape((1,) * tensor.ndim)
    return threshold * float(config.clip_ratio)


def fake_quantize_activation(
    tensor: torch.Tensor,
    config: ActivationQuantizationConfig,
) -> torch.Tensor:
    """Quantize to dynamic signed INT8 and dequantize to the input dtype."""
    config.validate()
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise TypeError("Activation quantization requires a floating tensor")
    if tensor.ndim == 0:
        raise ValueError("Activation quantization requires at least one dimension")
    work = tensor.float()
    dims = _reduction_dims(work, config.granularity)
    absmax = work.abs().amax(dim=dims, keepdim=True)
    threshold = _clip_threshold(work, config, absmax)
    qmax = (1 << (config.bits - 1)) - 1
    scale = threshold / float(qmax)
    zero_units = scale <= float(config.eps)
    safe_scale = torch.where(zero_units, torch.ones_like(scale), scale)
    quantized = torch.clamp(torch.round(work / safe_scale), -qmax, qmax)
    dequantized = torch.where(
        zero_units,
        torch.zeros_like(work),
        quantized * safe_scale,
    )
    return dequantized.to(dtype=tensor.dtype)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import transformers  # type: ignore
except Exception:  # pragma: no cover - transformers is optional for some entrypoints.
    transformers = None


DEFAULT_WEIGHT_QUANT_METHOD = "FLexGen"
GPTQ_WEIGHT_QUANT_METHOD = "GPTQ"
DEFAULT_FLexGen_CACHE_DIR = "flexgen_cache"


def _cfg_get(raw_cfg: Any, key: str, default: Any) -> Any:
    if raw_cfg is None:
        return default
    if isinstance(raw_cfg, dict):
        return raw_cfg.get(key, default)
    return getattr(raw_cfg, key, default)


@dataclass(frozen=True)
class WeightQuantizationConfig:
    enable: bool = False
    method: str = DEFAULT_WEIGHT_QUANT_METHOD
    q_bits: int = 4
    group_size: int = 128
    sym: bool = True
    block_size: int = 128
    percdamp: float = 0.01
    act_order: bool = False
    static_groups: bool = False
    runtime_form: str = "fake"
    cache_dir: str = DEFAULT_FLexGen_CACHE_DIR

    def validate(self) -> None:
        if not self.enable:
            return
        if self.method not in {DEFAULT_WEIGHT_QUANT_METHOD, GPTQ_WEIGHT_QUANT_METHOD}:
            raise ValueError(
                f"Only method={DEFAULT_WEIGHT_QUANT_METHOD} or "
                f"{GPTQ_WEIGHT_QUANT_METHOD} is supported, got {self.method}"
            )
        if self.q_bits != 4:
            raise ValueError(f"Weight quantization is fixed to q_bits=4, got {self.q_bits}")
        if int(self.group_size) <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if int(self.block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not self.sym:
            raise ValueError("This FLexGen deployment only supports symmetric grouped quantization.")
        if float(self.percdamp) <= 0:
            raise ValueError(f"percdamp must be positive, got {self.percdamp}")
        if self.runtime_form != "fake":
            raise ValueError(f"Only runtime_form='fake' is supported, got {self.runtime_form}")


def build_weight_quantization_config(raw_cfg: Any) -> WeightQuantizationConfig:
    raw_method = str(_cfg_get(raw_cfg, "method", DEFAULT_WEIGHT_QUANT_METHOD))
    if raw_method.lower() == DEFAULT_WEIGHT_QUANT_METHOD.lower():
        method = DEFAULT_WEIGHT_QUANT_METHOD
    elif raw_method.upper() == GPTQ_WEIGHT_QUANT_METHOD:
        method = GPTQ_WEIGHT_QUANT_METHOD
    else:
        method = raw_method
    cfg = WeightQuantizationConfig(
        enable=bool(_cfg_get(raw_cfg, "enable", False)),
        method=method,
        q_bits=int(_cfg_get(raw_cfg, "q_bits", 4)),
        group_size=int(_cfg_get(raw_cfg, "group_size", 128)),
        sym=bool(_cfg_get(raw_cfg, "sym", True)),
        block_size=int(_cfg_get(raw_cfg, "block_size", 128)),
        percdamp=float(_cfg_get(raw_cfg, "percdamp", 0.01)),
        act_order=bool(_cfg_get(raw_cfg, "act_order", False)),
        static_groups=bool(_cfg_get(raw_cfg, "static_groups", False)),
        runtime_form=str(_cfg_get(raw_cfg, "runtime_form", "fake")),
        cache_dir=str(_cfg_get(raw_cfg, "cache_dir", DEFAULT_FLexGen_CACHE_DIR)),
    )
    cfg.validate()
    return cfg


def _is_transformers_conv1d(layer: nn.Module) -> bool:
    return transformers is not None and isinstance(layer, transformers.Conv1D)


def _is_linear_like(layer: nn.Module) -> bool:
    weight = getattr(layer, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.ndim == 2


def clone_layer_weight_matrix(layer: nn.Module) -> torch.Tensor:
    weight = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        return weight.flatten(1)
    if _is_transformers_conv1d(layer):
        return weight.t()
    if _is_linear_like(layer):
        return weight
    raise TypeError(f"Unsupported FLexGen layer type: {type(layer)}")


def assign_layer_weight_matrix(layer: nn.Module, weight_matrix: torch.Tensor) -> None:
    out = weight_matrix
    if _is_transformers_conv1d(layer):
        out = out.t()
    layer.weight.data.copy_(out.reshape_as(layer.weight).to(dtype=layer.weight.data.dtype, device=layer.weight.data.device))


def quantize_with_params(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: torch.Tensor) -> torch.Tensor:
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def _symmetric_group_params(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maxq = torch.tensor(2**bits - 1, device=x.device, dtype=x.dtype)
    xmin = torch.minimum(x.amin(dim=1, keepdim=True), torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype))
    xmax = torch.maximum(x.amax(dim=1, keepdim=True), torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype))
    xmax = torch.maximum(xmax, xmin.abs())
    has_negative = xmin < 0
    if torch.any(has_negative):
        xmin = torch.where(has_negative, -xmax, xmin)
    empty = (xmin == 0) & (xmax == 0)
    if torch.any(empty):
        xmin = torch.where(empty, torch.full_like(xmin, -1.0), xmin)
        xmax = torch.where(empty, torch.full_like(xmax, 1.0), xmax)
    scale = (xmax - xmin) / maxq
    zero = torch.full_like(scale, (float(maxq.item()) + 1.0) / 2.0)
    return scale, zero, maxq


class FLexGenLinearQuantizer:
    def __init__(self, layer: nn.Module, cfg: WeightQuantizationConfig):
        self.layer = layer
        self.cfg = cfg
        self.dev = layer.weight.device
        self.weight_matrix = clone_layer_weight_matrix(layer).float()
        self.rows = self.weight_matrix.shape[0]
        self.columns = self.weight_matrix.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if _is_linear_like(self.layer) or _is_transformers_conv1d(self.layer):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        elif isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2]).flatten(1)
        else:
            raise TypeError(f"Unsupported FLexGen layer type: {type(self.layer)}")

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2.0 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def _find_params(self, weight_slice: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _symmetric_group_params(weight_slice, self.cfg.q_bits)

    def quantize(self) -> Dict[str, Any]:
        W = self.weight_matrix.clone()
        H = self.H
        dead = torch.diag(H) == 0
        if torch.any(dead):
            H = H.clone()
            H[dead, dead] = 1
            W[:, dead] = 0

        groups: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
        if self.cfg.static_groups:
            groups = []
            for start in range(0, self.columns, self.cfg.group_size):
                groups.append(self._find_params(W[:, start:start + self.cfg.group_size]))

        perm = None
        invperm = None
        if self.cfg.act_order:
            perm = torch.argsort(torch.diag(H), descending=True)
            invperm = torch.argsort(perm)
            W = W[:, perm]
            H = H[perm][:, perm]

        damp = self.cfg.percdamp * torch.mean(torch.diag(H))
        if not torch.isfinite(damp) or float(damp.item()) <= 0.0:
            damp = torch.tensor(1e-6, device=self.dev, dtype=H.dtype)
        diag = torch.arange(self.columns, device=self.dev)
        H = H.clone()
        H = (H + H.t()) * 0.5
        last_info = None
        for retry in range(6):
            H_work = H.clone()
            H_work[diag, diag] += damp * (10 ** retry)
            Hchol, info = torch.linalg.cholesky_ex(H_work)
            if int(info.max().item()) == 0:
                H = Hchol
                last_info = None
                break
            last_info = int(info.max().item())
        if last_info is not None:
            raise RuntimeError(f"FLexGen Hessian Cholesky failed after damping retries; info={last_info}")
        H = torch.cholesky_inverse(H)
        H = (H + H.t()) * 0.5
        Hchol, info = torch.linalg.cholesky_ex(H, upper=True)
        if int(info.max().item()) != 0:
            H[diag, diag] += torch.finfo(H.dtype).eps * 1024
            Hchol = torch.linalg.cholesky(H, upper=True)
        H = Hchol
        Hinv = H

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        for i1 in range(0, self.columns, self.cfg.block_size):
            i2 = min(i1 + self.cfg.block_size, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            scale = zero = maxq = None
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if self.cfg.group_size != -1:
                    if not self.cfg.static_groups:
                        if (i1 + i) % self.cfg.group_size == 0 or scale is None:
                            scale, zero, maxq = self._find_params(W[:, (i1 + i):(i1 + i + self.cfg.group_size)])
                    else:
                        assert groups is not None
                        idx = i1 + i
                        if perm is not None:
                            idx = int(perm[idx].item())
                        scale, zero, maxq = groups[idx // self.cfg.group_size]
                else:
                    scale, zero, maxq = self._find_params(W)

                assert scale is not None and zero is not None and maxq is not None
                q = quantize_with_params(
                    w.unsqueeze(1),
                    scale.to(w.device, w.dtype),
                    zero.to(w.device, w.dtype),
                    maxq.to(w.device, w.dtype),
                ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / (d ** 2)
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            if i2 < self.columns:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if invperm is not None:
            Q = Q[:, invperm]

        assign_layer_weight_matrix(self.layer, Q)
        total_error = float(torch.sum(Losses).item())
        self.weight_matrix = Q.detach()
        self.H = None  # type: ignore[assignment]
        torch.cuda.empty_cache()
        return {
            "rows": self.rows,
            "columns": self.columns,
            "error": total_error,
            "group_size": self.cfg.group_size,
            "block_size": self.cfg.block_size,
            "act_order": self.cfg.act_order,
            "static_groups": self.cfg.static_groups,
        }


class GPTQLinearQuantizer(FLexGenLinearQuantizer):
    """Explicit GPTQ entrypoint using the existing second-order implementation."""

    def _find_params(
        self,
        weight_slice: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        maxq = torch.tensor(
            2**self.cfg.q_bits - 1,
            device=weight_slice.device,
            dtype=weight_slice.dtype,
        )
        absmax = weight_slice.abs().amax(dim=1, keepdim=True)
        safe_absmax = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
        scale = (2.0 * safe_absmax) / maxq
        zero = torch.full_like(scale, (float(maxq.item()) + 1.0) / 2.0)
        return scale, zero, maxq

    def quantize(self) -> Dict[str, Any]:
        if self.nsamples <= 0:
            raise RuntimeError(
                "GPTQ received no calibration activations for this layer."
            )
        if not torch.isfinite(self.H).all():
            raise RuntimeError("GPTQ Hessian contains non-finite values.")
        if int(torch.count_nonzero(torch.diag(self.H)).item()) == 0:
            raise RuntimeError("GPTQ calibration observed only zero-valued inputs.")
        return super().quantize()

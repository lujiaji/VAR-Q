"""GPTQ W4 plus dynamic-A8 reference runtime for next-frame video models.

Weights are calibrated once with GPTQ and stored as their dequantized W4
snapshot.  Selected linear inputs are dynamically fake-quantized to signed A8
at every forward.  This module intentionally provides no packed-weight claim
or packed runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_quant import (
    ActivationQuantizationConfig,
    fake_quantize_activation,
)
from .weight_flexgen_apply import maybe_apply_gptq


_NEXTFRAME_TARGET_ALIASES: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    "self_forcing": (
        ("self_attn.q", ("self_attn.q", "self_attn.q_proj", "attn.q", "attn.q_proj")),
        ("self_attn.k", ("self_attn.k", "self_attn.k_proj", "attn.k", "attn.k_proj")),
        ("self_attn.v", ("self_attn.v", "self_attn.v_proj", "attn.v", "attn.v_proj")),
        ("self_attn.o", ("self_attn.o", "self_attn.o_proj", "attn.o", "attn.o_proj", "attn.proj")),
        ("ffn.fc1", ("ffn.0", "ffn.fc1", "mlp.fc1", "mlp.0")),
        ("ffn.fc2", ("ffn.2", "ffn.fc2", "mlp.fc2", "mlp.2")),
    ),
    "longlive": (
        ("self_attn.q", ("self_attn.q", "attn.q", "attn.q_proj")),
        ("self_attn.k", ("self_attn.k", "attn.k", "attn.k_proj")),
        ("self_attn.v", ("self_attn.v", "attn.v", "attn.v_proj")),
        ("self_attn.o", ("self_attn.o", "attn.o", "attn.o_proj", "attn.proj")),
        ("ffn.fc1", ("ffn.0", "ffn.fc1", "mlp.fc1", "mlp.0")),
        ("ffn.fc2", ("ffn.2", "ffn.fc2", "mlp.fc2", "mlp.2")),
    ),
    "livetalk": (
        ("self_attn.q", ("self_attn.q", "attn.q", "attn.q_proj")),
        ("self_attn.k", ("self_attn.k", "attn.k", "attn.k_proj")),
        ("self_attn.v", ("self_attn.v", "attn.v", "attn.v_proj")),
        ("self_attn.o", ("self_attn.o", "attn.o", "attn.o_proj", "attn.proj")),
        ("ffn.fc1", ("ffn.0", "ffn.fc1", "mlp.fc1", "mlp.0")),
        ("ffn.fc2", ("ffn.2", "ffn.fc2", "mlp.fc2", "mlp.2")),
    ),
}


@dataclass(frozen=True)
class _ResolvedTarget:
    canonical_name: str
    module_path: str
    module: nn.Module


class GPTQFakeW4A8Linear(nn.Module):
    """Linear layer backed by one GPTQ-restored W4 snapshot and dynamic A8."""

    def __init__(
        self,
        *,
        runtime_name: str,
        restored_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        group_size: int,
        activation_config: ActivationQuantizationConfig,
    ) -> None:
        super().__init__()
        if restored_weight.ndim != 2:
            raise ValueError("GPTQ restored weight must be rank-2")
        if group_size <= 0:
            raise ValueError("GPTQ group_size must be positive")
        activation_config.validate()
        self.runtime_name = str(runtime_name)
        self.in_features = int(restored_weight.shape[1])
        self.out_features = int(restored_weight.shape[0])
        self.group_size = int(group_size)
        self.weight_bits = 4
        self.activation_bits = 8
        self.weight_runtime_form = "gptq_dequantized_snapshot"
        self.activation_config = activation_config
        self.register_buffer(
            "restored_weight",
            restored_weight.detach().clone(),
            persistent=True,
        )
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        *,
        runtime_name: str,
        group_size: int,
        activation_config: ActivationQuantizationConfig,
    ) -> "GPTQFakeW4A8Linear":
        weight = getattr(linear, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise RuntimeError(f"{runtime_name} is not a 2D-weight linear")
        bias = getattr(linear, "bias", None)
        return cls(
            runtime_name=runtime_name,
            restored_weight=weight,
            bias=bias if isinstance(bias, torch.Tensor) else None,
            group_size=group_size,
            activation_config=activation_config,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        activation = fake_quantize_activation(tensor, self.activation_config)
        weight = self.restored_weight
        if weight.device != activation.device or weight.dtype != activation.dtype:
            weight = weight.to(device=activation.device, dtype=activation.dtype)
        bias = self.bias
        if isinstance(bias, torch.Tensor) and (
            bias.device != activation.device or bias.dtype != activation.dtype
        ):
            bias = bias.to(device=activation.device, dtype=activation.dtype)
        return F.linear(activation, weight, bias)


@dataclass
class NextFrameW4A8Runtime:
    family: str
    modules: List[GPTQFakeW4A8Linear]

    def metadata(self) -> Dict[str, Any]:
        """Return static runtime properties, not execution counters."""
        return {
            "family": self.family,
            "weight_bits": 4,
            "activation_bits": 8,
            "packed_runtime": False,
            "weight_runtime_form": "gptq_dequantized_snapshot",
            "replaced_modules": len(self.modules),
            "replaced_module_names": [module.runtime_name for module in self.modules],
        }


def _resolve_target_blocks(model: nn.Module) -> List[nn.Module]:
    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, (list, tuple, nn.ModuleList)):
        return list(blocks)
    raise RuntimeError("Next-frame GPTQ model is missing a blocks collection")


def _resolve_alias_targets(
    block: nn.Module,
    aliases: Sequence[Tuple[str, Sequence[str]]],
    family: str,
) -> List[_ResolvedTarget]:
    by_name = dict(block.named_modules())
    resolved: List[_ResolvedTarget] = []
    missing: List[str] = []
    for canonical_name, candidates in aliases:
        target_path = None
        target_module = None
        for candidate in candidates:
            module = by_name.get(candidate)
            weight = getattr(module, "weight", None) if module is not None else None
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                target_path = candidate
                target_module = module
                break
        if target_path is None or target_module is None:
            missing.append(canonical_name)
        else:
            resolved.append(
                _ResolvedTarget(canonical_name, target_path, target_module)
            )
    if missing:
        raise RuntimeError(
            f"{family} GPTQ target mismatch in {block.__class__.__name__}: "
            f"missing={missing}"
        )
    return resolved


def _set_module_path(root: nn.Module, module_path: str, replacement: nn.Module) -> None:
    parent = root
    parts = module_path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def load_nextframe_runtime_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Next-frame runtime config must be a JSON object")
    config = dict(raw)
    config.setdefault("weight_quantization", {})
    config.setdefault("activation_quantization", {})
    return config


def _build_activation_config(
    config: Mapping[str, Any],
) -> ActivationQuantizationConfig:
    raw = dict(config.get("activation_quantization", {}))
    if not bool(raw.get("enable", False)):
        raise ValueError("W4A8 requires activation_quantization.enable=true")
    activation = ActivationQuantizationConfig(
        bits=int(raw.get("bits", 8)),
        granularity=str(raw.get("granularity", "per_token")),
        clip_ratio=float(raw.get("clip_ratio", 1.0)),
        clip_method=str(raw.get("clip_method", "absmax")),
        clip_percentile=float(raw.get("clip_percentile", 1.0)),
        eps=float(raw.get("eps", 1e-12)),
    )
    activation.validate()
    return activation


def _install_fake_runtime(
    model: nn.Module,
    family: str,
    config: Mapping[str, Any],
) -> NextFrameW4A8Runtime:
    weight_config = dict(config.get("weight_quantization", {}))
    group_size = int(weight_config.get("group_size", 128))
    activation_config = _build_activation_config(config)
    replacements: List[GPTQFakeW4A8Linear] = []
    for block_index, block in enumerate(_resolve_target_blocks(model)):
        targets = _resolve_alias_targets(
            block,
            _NEXTFRAME_TARGET_ALIASES[family],
            family,
        )
        for target in targets:
            runtime_name = f"block_{block_index}.{target.canonical_name}"
            replacement = GPTQFakeW4A8Linear.from_linear(
                target.module,
                runtime_name=runtime_name,
                group_size=group_size,
                activation_config=activation_config,
            )
            _set_module_path(block, target.module_path, replacement)
            replacements.append(replacement)
    return NextFrameW4A8Runtime(family=family, modules=replacements)


def _weight_family(family: str) -> str:
    return "longlive" if family == "livetalk" else family


def apply_nextframe_reference_quantization(
    model: nn.Module,
    family: str,
    config: Mapping[str, Any],
    *,
    text_encoder: Optional[nn.Module] = None,
    model_tag: Optional[str] = None,
    calibration_prompts: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Apply GPTQ W4 once, then install dynamic A8 on target linears."""
    family = str(family)
    if family not in _NEXTFRAME_TARGET_ALIASES:
        raise ValueError(f"Unsupported next-frame family: {family}")
    weight_config = dict(config.get("weight_quantization", {}))
    if not bool(weight_config.get("enable", False)):
        return {"applied": False, "reason": "disabled", "runtime": None}
    if str(weight_config.get("method", "GPTQ")).upper() != "GPTQ":
        raise ValueError("Next-frame W4A8 requires weight method=GPTQ")
    if int(weight_config.get("q_bits", 4)) != 4:
        raise ValueError("Next-frame W4A8 requires weight q_bits=4")
    if str(weight_config.get("runtime_form", "fake")) != "fake":
        raise ValueError(
            "The release GPTQ path supports runtime_form='fake' only; packed "
            "weights are not claimed."
        )
    _build_activation_config(config)

    kwargs: Dict[str, Any] = {}
    if text_encoder is not None:
        kwargs["text_encoder"] = text_encoder
    if calibration_prompts is not None:
        kwargs["calibration_prompts"] = calibration_prompts
    gptq = maybe_apply_gptq(
        model,
        weight_config,
        model_tag=model_tag or f"{family}-gptq-w4a8",
        family=_weight_family(family),
        **kwargs,
    )
    runtime = _install_fake_runtime(model, family, config)
    return {
        "applied": True,
        "gptq": gptq,
        "runtime": runtime,
        "runtime_metadata": runtime.metadata(),
    }


@dataclass
class DeferredNextFrameQuantization:
    pipeline: nn.Module
    family: str
    install: Callable[[str], Dict[str, Any]]
    original_generator_to: Any
    original_inference: Optional[Any]
    installed: bool = False
    result: Optional[Dict[str, Any]] = None


def defer_nextframe_quantization_until_generator_ready(
    pipeline: nn.Module,
    family: str,
    config: Mapping[str, Any],
    *,
    model_tag: str,
    calibration_prompts: Sequence[str],
    hook_installer: Optional[Callable[[nn.Module], Any]] = None,
) -> DeferredNextFrameQuantization:
    """Defer GPTQ until upstream checkpoint and optional LoRA loading finish."""
    generator = getattr(pipeline, "generator", None)
    if generator is None or getattr(generator, "model", None) is None:
        raise RuntimeError("Next-frame pipeline has no generator.model")
    if not calibration_prompts:
        raise ValueError("calibration_prompts must be non-empty")
    original_generator_to = generator.to
    original_inference = getattr(pipeline, "inference", None)
    holder: Dict[str, DeferredNextFrameQuantization] = {}

    def install(_trigger: str) -> Dict[str, Any]:
        handle = holder["handle"]
        if handle.installed:
            assert handle.result is not None
            return handle.result
        model = generator.model
        merge = getattr(model, "merge_and_unload", None)
        if callable(merge):
            merged = merge()
            if not isinstance(merged, nn.Module):
                raise RuntimeError("merge_and_unload did not return an nn.Module")
            generator.model = merged
            model = merged
        result = apply_nextframe_reference_quantization(
            model,
            family,
            config,
            text_encoder=getattr(pipeline, "text_encoder", None),
            model_tag=model_tag,
            calibration_prompts=calibration_prompts,
        )
        if hook_installer is not None:
            hook_installer(pipeline)
        handle.installed = True
        handle.result = result
        return result

    handle = DeferredNextFrameQuantization(
        pipeline=pipeline,
        family=family,
        install=install,
        original_generator_to=original_generator_to,
        original_inference=original_inference if callable(original_inference) else None,
    )
    holder["handle"] = handle

    def generator_to(*args, **kwargs):
        output = original_generator_to(*args, **kwargs)
        install("generator.to")
        return output

    generator.to = generator_to
    if callable(original_inference):

        def inference(*args, **kwargs):
            install("pipeline.inference")
            return original_inference(*args, **kwargs)

        pipeline.inference = inference
    return handle


def remove_deferred_nextframe_quantization(
    handle: DeferredNextFrameQuantization,
) -> None:
    """Restore only the temporary defer wrappers."""
    generator = getattr(handle.pipeline, "generator", None)
    if generator is not None:
        generator.to = handle.original_generator_to
    if handle.original_inference is not None:
        handle.pipeline.inference = handle.original_inference

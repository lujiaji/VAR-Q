from __future__ import annotations

import importlib
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from VAR_Q import build_infinitystar_cache_quantizer as build_varq_infinitystar_cache_quantizer
from VAR_Q import build_kv_cache_quantizer as build_varq_kv_cache_quantizer
from VAR_Q.config_loader import _normalize_quant_method_name


SUPPORTED_MODEL_TYPES = ("var", "infinity", "infinitystar", "self_forcing", "longlive", "generic_video")
_PACKABLE_BITS = (2, 3, 4, 6, 8)
_ABLATION_METHOD_NAMES = {
    "KIVI",
    "KIVI-CALI",
    "FLEXGEN",
    "KVQUANT",
    "ABL_KIVI",
    "ABL_KIVI_CALI",
    "ABL_KV_FLEXGEN",
    "ABL_KVQUANT",
}


@dataclass
class _PatchedAttribute:
    module: nn.Module
    name: str
    existed: bool
    original: Any


@dataclass
class HookHandle:
    model: nn.Module
    model_type: str
    modules: List[nn.Module] = field(default_factory=list)
    patched_attributes: List[_PatchedAttribute] = field(default_factory=list)
    hit_count: int = 0


def _get_ablation_api() -> Tuple[Callable[..., Any], Callable[..., Any], Callable[[str], bool]]:
    module = importlib.import_module("ablation")
    return (
        getattr(module, "build_kv_cache_quantizer"),
        getattr(module, "build_infinitystar_cache_quantizer"),
        getattr(module, "is_ablation_method"),
    )


def _is_ablation_method(quant_method: str) -> bool:
    normalized = _normalize_quant_method_name(str(quant_method))
    return normalized.upper() in _ABLATION_METHOD_NAMES


def _build_kv_cache_quantizer(**kwargs: Any) -> Any:
    quant_method = str(kwargs.get("quant_method", "VARQ"))
    if _is_ablation_method(quant_method):
        if str(kwargs.get("dequant_dtype", "bf16")) == "native":
            kwargs = dict(kwargs)
            kwargs["dequant_dtype"] = "bf16"
        build_ablation, _, _ = _get_ablation_api()
        return build_ablation(**kwargs)
    return build_varq_kv_cache_quantizer(**kwargs)


def _build_infinitystar_cache_quantizer(**kwargs: Any) -> Any:
    quant_method = str(kwargs.get("quant_method", "VARQ"))
    if _is_ablation_method(quant_method):
        if str(kwargs.get("dequant_dtype", "bf16")) == "native":
            kwargs = dict(kwargs)
            kwargs["dequant_dtype"] = "bf16"
        _, build_ablation, _ = _get_ablation_api()
        return build_ablation(**kwargs)
    return build_varq_infinitystar_cache_quantizer(**kwargs)


def _remember_attr(handle: HookHandle, module: nn.Module, name: str) -> None:
    handle.patched_attributes.append(
        _PatchedAttribute(
            module=module,
            name=name,
            existed=hasattr(module, name),
            original=getattr(module, name, None),
        )
    )


def _set_attr(handle: HookHandle, module: nn.Module, name: str, value: Any) -> None:
    _remember_attr(handle, module, name)
    setattr(module, name, value)


def _normalize_quant_config(raw_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(raw_config or {})
    if "enable" not in cfg and "enable_quantization" in cfg:
        cfg["enable"] = cfg["enable_quantization"]
    if "quant_method" in cfg:
        cfg["quant_method"] = _normalize_quant_method_name(cfg["quant_method"])
    if "q_bits" in cfg:
        cfg["q_bits"] = int(cfg["q_bits"])
    if "enable" in cfg:
        cfg["enable"] = bool(cfg["enable"])
    return cfg


def _role_bits(cfg: Dict[str, Any], role: str) -> int:
    key = f"q_bits_{role}"
    return int(cfg.get(key, cfg.get("q_bits", 8)))


def _role_pack(cfg: Dict[str, Any], role: str) -> bool:
    key = f"pack_to_int32_{role}"
    if key in cfg:
        return bool(cfg[key])
    if "pack_to_int32" in cfg:
        return bool(cfg["pack_to_int32"])
    return _role_bits(cfg, role) in _PACKABLE_BITS


def _iter_attention_modules(model: nn.Module, model_type: str) -> Iterable[Tuple[str, nn.Module]]:
    if model_type in ("self_forcing", "longlive", "generic_video"):
        # Generic video backends should use VideoKVCacheAdapter inside their
        # attention implementation. Automatic wrapping is intentionally limited
        # to known attention signatures to avoid silently changing semantics.
        return
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name != "SelfAttention":
            continue
        if model_type == "infinitystar":
            if all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")):
                yield name, module
        elif all(hasattr(module, attr) for attr in ("mat_qkv", "q_bias", "v_bias", "proj")):
            yield name, module


def _attention_discovery_requirements(model_type: str) -> str:
    if model_type == "infinitystar":
        return "class name SelfAttention with q_proj/k_proj/v_proj/o_proj attributes"
    return "class name SelfAttention with mat_qkv/q_bias/v_bias/proj attributes"


def _no_attention_hits_error(model_type: str, scanned_modules: int) -> RuntimeError:
    requirements = _attention_discovery_requirements(model_type)
    return RuntimeError(
        f"VAR-Q runtime hooks patched 0 attention modules for model_type={model_type}; "
        f"scanned {scanned_modules} modules. No module matched {requirements}. "
        "Check the target model version, class names, projection attribute names, and model_type; "
        "if the backend changed, update VAR_Q/hooks/runtime.py discovery before trusting hook results."
    )


def _patch_builder_globals(module: nn.Module, model_type: str) -> None:
    kv_caching = getattr(module, "kv_caching", None)
    func = getattr(kv_caching, "__func__", kv_caching)
    globals_dict = getattr(func, "__globals__", None)
    if not isinstance(globals_dict, dict):
        return
    globals_dict["build_varq_cache_quantizer"] = build_varq_kv_cache_quantizer
    globals_dict["_build_kv_cache_quantizer"] = _build_kv_cache_quantizer
    globals_dict["_build_infinitystar_cache_quantizer"] = _build_infinitystar_cache_quantizer
    globals_dict["build_varq_infinitystar_cache_quantizer"] = build_varq_infinitystar_cache_quantizer
    globals_dict["is_ablation_method"] = _is_ablation_method
    if model_type != "infinitystar":
        globals_dict["build_ablation_cache_quantizer"] = lambda **kwargs: _get_ablation_api()[0](**kwargs)
    else:
        globals_dict["build_ablation_cache_quantizer"] = lambda **kwargs: _get_ablation_api()[1](**kwargs)


def _max_scale_seq_len(module: nn.Module) -> Optional[int]:
    value = int(getattr(module, "max_scale_seq_len", 0) or 0)
    return value or None


def _kv_quantizer_kwargs(module: nn.Module, role: str, qkv_format: str) -> Dict[str, Any]:
    return {
        "quant_bits": int(getattr(module, f"q_bits_{role}", getattr(module, "q_bits", 8))),
        "qkv_format": qkv_format,
        "quant_method": str(getattr(module, "quant_method", "VARQ")),
        "kv_role": role,
        "blk_idx": int(getattr(module, "block_idx", 0)),
        "pack_to_int32": bool(getattr(module, f"pack_to_int32_{role}", True)),
        "kivi_group_size": int(getattr(module, "kivi_group_size", 128)),
        "kivi_cali_k_group_size": int(getattr(module, "kivi_cali_k_group_size", 128)),
        "kivi_cali_v_group_size": int(getattr(module, "kivi_cali_v_group_size", 128)),
        "compression_ratio": float(getattr(module, "compression_ratio", 1.0)),
        "max_scale_seq_len": _max_scale_seq_len(module),
        "rescale_qk": bool(getattr(module, "rescale_qk", False)),
        "debug": bool(getattr(module, "debug_memory", False)),
        "dequant_dtype": str(getattr(module, "dequant_dtype", "native")),
        "quant_compute_dtype": str(getattr(module, "quant_compute_dtype", "native")),
        "expected_total_seq_len": int(getattr(module, "expected_total_seq_len", 0) or 0) or None,
        "preallocate_kv_cache": bool(getattr(module, "preallocate_kv_cache", False)),
        "dequant_workspace_policy": str(getattr(module, "dequant_workspace_policy", "release")),
        "ablation_config": getattr(module, "ablation_config", None),
    }


def _infinitystar_quantizer_kwargs(module: nn.Module, role: str) -> Dict[str, Any]:
    return {
        "quant_bits": int(getattr(module, "q_bits", 8)),
        "qkv_format": "BHLc",
        "quant_method": str(getattr(module, "quant_method", "VARQ")),
        "kv_role": role,
        "kivi_group_size": int(getattr(module, "kivi_group_size", 128)),
        "kivi_cali_k_group_size": int(getattr(module, "kivi_cali_k_group_size", 128)),
        "kivi_cali_v_group_size": int(getattr(module, "kivi_cali_v_group_size", 128)),
        "pack_to_int32": bool(getattr(module, "pack_to_int32", True)),
        "compression_ratio": float(getattr(module, "compression_ratio", 1.0)),
        "max_scale_seq_len": _max_scale_seq_len(module),
        "rescale_qk": bool(getattr(module, "rescale_qk", False)),
        "debug": bool(getattr(module, "debug_memory", False)),
        "dequant_dtype": str(getattr(module, "dequant_dtype", "native")),
        "quant_compute_dtype": str(getattr(module, "quant_compute_dtype", "native")),
        "expected_total_seq_len": int(getattr(module, "expected_total_seq_len", 0) or 0) or None,
        "preallocate_kv_cache": bool(getattr(module, "preallocate_kv_cache", False)),
        "dequant_workspace_policy": str(getattr(module, "dequant_workspace_policy", "release")),
        "ablation_config": getattr(module, "ablation_config", None),
    }


def _ensure_kv_quantizers(module: nn.Module, qkv_format: str) -> None:
    if getattr(module, "k_quant", None) is not None and getattr(module, "v_quant", None) is not None:
        if hasattr(module.k_quant, "set_qkv_format"):
            module.k_quant.set_qkv_format(qkv_format)
        else:
            module.k_quant.qkv_format = qkv_format
        if hasattr(module.v_quant, "set_qkv_format"):
            module.v_quant.set_qkv_format(qkv_format)
        else:
            module.v_quant.qkv_format = qkv_format
        return
    module.k_quant = _build_kv_cache_quantizer(**_kv_quantizer_kwargs(module, "k", qkv_format))
    module.v_quant = _build_kv_cache_quantizer(**_kv_quantizer_kwargs(module, "v", qkv_format))


def _ensure_infinitystar_quantizers(module: nn.Module) -> None:
    if getattr(module, "k_varq", None) is not None and getattr(module, "v_varq", None) is not None:
        return
    module.k_varq = _build_infinitystar_cache_quantizer(**_infinitystar_quantizer_kwargs(module, "k"))
    module.v_varq = _build_infinitystar_cache_quantizer(**_infinitystar_quantizer_kwargs(module, "v"))


def _release_attention_dequant_workspaces(module: nn.Module) -> None:
    for attr in ("k_quant", "v_quant"):
        quantizer = getattr(module, attr, None)
        if quantizer is None:
            continue
        if hasattr(quantizer, "maybe_release_dequant_workspace"):
            quantizer.maybe_release_dequant_workspace()
        elif hasattr(quantizer, "release_dequant_workspace"):
            quantizer.release_dequant_workspace()


def _maybe_empty_cuda_cache_after_scale(module: nn.Module) -> None:
    policy = str(getattr(module, "empty_cache_policy", "after_generation"))
    if policy != "after_scale":
        if policy == "threshold" and torch.cuda.is_available():
            threshold = int(getattr(module, "empty_cache_threshold_bytes", 0) or 0)
            if threshold > 0 and torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > threshold:
                torch.cuda.empty_cache()
        return
    if int(getattr(module, "_varq_hook_order_idx", -1)) != int(getattr(module, "_varq_last_attention_order_idx", -2)):
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _use_quantizer(quantizer: Any, item: torch.Tensor, cache_current: bool) -> torch.Tensor:
    try:
        return quantizer.use_var_q(item, cache_current=cache_current)
    except TypeError:
        return quantizer.use_var_q(item)


def _as_attention_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor.contiguous()


def _var_is_last_scale(module: nn.Module, current_len: int) -> bool:
    return bool(getattr(module, "skip_cache_last_scale", True)) and int(
        getattr(module, "last_scale_seq_len", 0) or 0
    ) == int(current_len)


def _infinity_is_last_scale(module: nn.Module, scale_schedule: Any, scale_ind: int) -> bool:
    return (
        bool(getattr(module, "skip_cache_last_scale", True))
        and scale_schedule is not None
        and isinstance(scale_ind, int)
        and scale_ind == len(scale_schedule) - 1
    )


def _should_use_fused_kv_attn(enabled, is_last_scale, qkv_format, bits):
    """v1 fused path: only the last (two-segment) AR step, q8, known layout."""
    return bool(enabled) and bool(is_last_scale) and int(bits) == 8 \
        and qkv_format in ("BHLc", "BLHc")


def _attention_enabled(module: nn.Module) -> bool:
    return bool(getattr(module, "caching", False)) and bool(getattr(module, "enable_quantization", False))


def _shape_tuple(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _linear_out_features(proj: Any) -> Optional[int]:
    weight = getattr(proj, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
        return int(weight.shape[0])
    out_features = getattr(proj, "out_features", None)
    if out_features is None:
        return None
    return int(out_features)


def _infer_projected_heads(module: nn.Module, role: str) -> Optional[int]:
    head_dim = int(getattr(module, "head_dim", 0) or 0)
    if head_dim <= 0:
        return None
    proj_attr = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}.get(role)
    if proj_attr and hasattr(module, proj_attr):
        out_features = _linear_out_features(getattr(module, proj_attr))
        if out_features is not None and out_features % head_dim == 0:
            return int(out_features // head_dim)
    mat_qkv = getattr(module, "mat_qkv", None)
    out_features = _linear_out_features(mat_qkv)
    if out_features is not None and out_features % (3 * head_dim) == 0:
        return int(out_features // (3 * head_dim))
    return None


def _infer_expected_heads(module: nn.Module, role: str) -> Optional[int]:
    if role in ("k", "v") and hasattr(module, "num_key_value_heads"):
        value = getattr(module, "num_key_value_heads")
        if value is not None:
            return int(value)
    if hasattr(module, "num_heads"):
        value = getattr(module, "num_heads")
        if value is not None:
            return int(value)
    return _infer_projected_heads(module, role)


def _expected_heads_for_layout_check(module: nn.Module, role: str) -> Optional[int]:
    attr = f"_varq_expected_{role}_heads"
    if hasattr(module, attr):
        return int(getattr(module, attr))
    return _infer_expected_heads(module, role)


def _validate_hidden_states_3d(tensor: torch.Tensor, wrapper_name: str) -> None:
    if tensor.ndim != 3:
        raise RuntimeError(
            f"{wrapper_name} VAR-Q runtime hook expected hidden states with shape=(B, L, C); "
            f"got shape={_shape_tuple(tensor)}."
        )


def _format_qkv_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> str:
    return f"q.shape={_shape_tuple(q)}, k.shape={_shape_tuple(k)}, v.shape={_shape_tuple(v)}"


def _validate_qkv_layout(
    module: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qkv_format: str,
    expected_seq_len: int,
    wrapper_name: str,
) -> None:
    if qkv_format == "BLHc":
        seq_dim, head_dim = 1, 2
    elif qkv_format == "BHLc":
        seq_dim, head_dim = 2, 1
    else:
        raise RuntimeError(
            f"{wrapper_name} VAR-Q runtime hook expected layout=BLHc or BHLc; "
            f"got layout={qkv_format}, {_format_qkv_shapes(q, k, v)}."
        )

    for role, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.ndim != 4:
            raise RuntimeError(
                f"{wrapper_name} VAR-Q runtime hook expected layout={qkv_format} with 4D Q/K/V tensors; "
                f"got {role}.ndim={tensor.ndim}, {_format_qkv_shapes(q, k, v)}."
            )
        if int(tensor.size(seq_dim)) != int(expected_seq_len):
            raise RuntimeError(
                f"{wrapper_name} VAR-Q runtime hook expected layout={qkv_format} with seq_dim={seq_dim} "
                f"size={expected_seq_len}; got {role}.size({seq_dim})={int(tensor.size(seq_dim))}, "
                f"{_format_qkv_shapes(q, k, v)}."
            )
        expected_heads = _expected_heads_for_layout_check(module, role)
        # Some third-party modules do not expose stable head metadata or linear
        # projection widths. In that case the hook still checks ndim and the
        # sequence axis, but cannot cheaply prove the H axis.
        if expected_heads is None:
            continue
        if int(tensor.size(head_dim)) != int(expected_heads):
            raise RuntimeError(
                f"{wrapper_name} VAR-Q runtime hook expected layout={qkv_format} with head_dim={head_dim} "
                f"size={expected_heads}; got {role}.size({head_dim})={int(tensor.size(head_dim))}, "
                f"{_format_qkv_shapes(q, k, v)}."
            )


def _has_rope_cache(rope_cache: Any) -> bool:
    if rope_cache is None:
        return False
    if isinstance(rope_cache, (list, tuple, dict)):
        return len(rope_cache) > 0
    return True


def _call_slow_attn(globals_dict: Dict[str, Any], query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float, attn_bias: Any) -> torch.Tensor:
    slow_attn = globals_dict.get("slow_attn")
    if slow_attn is not None:
        try:
            return slow_attn(query, key, value, scale=scale, attn_mask=attn_bias)
        except TypeError:
            return slow_attn(query, key, value, scale=scale, attn_bias=attn_bias)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias, scale=scale)


def _wrap_var_forward(handle: HookHandle, module: nn.Module) -> None:
    original = module.forward
    globals_dict = getattr(getattr(original, "__func__", original), "__globals__", {})

    def forward_wrapper(self: nn.Module, x: torch.Tensor, attn_bias: Any):
        if not _attention_enabled(self):
            return original(x, attn_bias)

        _validate_hidden_states_3d(x, "VAR")
        B, L, C = x.shape
        cache_current = not _var_is_last_scale(self, L)
        qkv = F.linear(
            input=x,
            weight=self.mat_qkv.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        ).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = torch.float16 if globals_dict.get("AUTOCAST_FLASH_ATTN", False) else qkv.dtype

        if getattr(self, "using_flash", False):
            q, k, v = qkv.unbind(dim=2)
            qkv_format = "BLHc"
            _validate_qkv_layout(self, q, k, v, qkv_format, L, "VAR")
            if getattr(self, "attn_l2_norm", False):
                scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
                scale_mul = scale_mul.transpose(1, 2)
                q = F.normalize(q, dim=-1).mul(scale_mul)
                k = F.normalize(k, dim=-1)
            _ensure_kv_quantizers(self, qkv_format)
            if bool(getattr(self, "rescale_qk", False)):
                q, k = self.k_quant.rescale_qk(q, k)
            k = _as_attention_dtype(_use_quantizer(self.k_quant, k, cache_current), q.dtype)
            v = _as_attention_dtype(_use_quantizer(self.v_quant, v, cache_current), q.dtype)
            flash_attn_func = globals_dict.get("flash_attn_func")
            if flash_attn_func is None:
                raise RuntimeError("flash_attn_func is not available in the third-party VAR runtime.")
            oup = flash_attn_func(q.to(main_type), k.to(main_type), v.to(main_type), dropout_p=0, softmax_scale=self.scale)
            if oup.shape[1] != L:
                oup = oup[:, -L:]
            oup = oup.reshape(B, L, C)
        elif getattr(self, "using_xform", False):
            q, k, v = qkv.unbind(dim=2)
            qkv_format = "BLHc"
            _validate_qkv_layout(self, q, k, v, qkv_format, L, "VAR")
            if getattr(self, "attn_l2_norm", False):
                scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
                scale_mul = scale_mul.transpose(1, 2)
                q = F.normalize(q, dim=-1).mul(scale_mul)
                k = F.normalize(k, dim=-1)
            _ensure_kv_quantizers(self, qkv_format)
            if bool(getattr(self, "rescale_qk", False)):
                q, k = self.k_quant.rescale_qk(q, k)
            k = _as_attention_dtype(_use_quantizer(self.k_quant, k, cache_current), q.dtype)
            v = _as_attention_dtype(_use_quantizer(self.v_quant, v, cache_current), q.dtype)
            memory_efficient_attention = globals_dict.get("memory_efficient_attention")
            if memory_efficient_attention is None:
                raise RuntimeError("memory_efficient_attention is not available in the third-party VAR runtime.")
            oup = memory_efficient_attention(q.to(main_type), k.to(main_type), v.to(main_type), attn_bias=None if attn_bias is None else attn_bias.to(torch.float32).expand(B, self.num_heads, -1, -1))
            oup = oup.view(B, L, C)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            qkv_format = "BHLc"
            _validate_qkv_layout(self, q, k, v, qkv_format, L, "VAR")
            if getattr(self, "attn_l2_norm", False):
                scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
                q = F.normalize(q, dim=-1).mul(scale_mul)
                k = F.normalize(k, dim=-1)
            _ensure_kv_quantizers(self, qkv_format)
            if bool(getattr(self, "rescale_qk", False)):
                q, k = self.k_quant.rescale_qk(q, k)
            k = _as_attention_dtype(_use_quantizer(self.k_quant, k, cache_current), q.dtype)
            v = _as_attention_dtype(_use_quantizer(self.v_quant, v, cache_current), q.dtype)
            oup = _call_slow_attn(globals_dict, q, k, v, self.scale, attn_bias).transpose(1, 2).reshape(B, L, C)

        _release_attention_dequant_workspaces(self)
        del k, v
        _maybe_empty_cuda_cache_after_scale(self)
        return self.proj_drop(self.proj(oup))

    _remember_attr(handle, module, "forward")
    module.forward = types.MethodType(forward_wrapper, module)


def _wrap_infinity_forward(handle: HookHandle, module: nn.Module) -> None:
    original = module.forward
    globals_dict = getattr(getattr(original, "__func__", original), "__globals__", {})

    def forward_wrapper(
        self: nn.Module,
        x: torch.Tensor,
        attn_bias_or_two_vector: Any,
        attn_fn: Any = None,
        scale_schedule: Any = None,
        rope2d_freqs_grid: Any = None,
        scale_ind: int = 0,
    ):
        if not _attention_enabled(self):
            return original(x, attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=rope2d_freqs_grid, scale_ind=scale_ind)

        _validate_hidden_states_3d(x, "Infinity")
        B, L, C = x.shape
        qkv = F.linear(
            input=x,
            weight=self.mat_qkv.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        ).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = torch.float16 if globals_dict.get("AUTOCAST_FLASH_ATTN", False) else qkv.dtype

        if getattr(self, "using_flash", False):
            q, k, v = qkv.unbind(dim=2)
            qkv_format = "BLHc"
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            qkv_format = "BHLc"
        _validate_qkv_layout(self, q, k, v, qkv_format, L, "Infinity")

        if getattr(self, "cos_attn", False):
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if getattr(self, "using_flash", False):
                scale_mul = scale_mul.permute(0, 2, 1, 3)
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        if scale_schedule is not None:
            apply_rotary_emb = globals_dict.get("apply_rotary_emb")
            if apply_rotary_emb is None:
                raise RuntimeError("apply_rotary_emb is not available in the third-party Infinity runtime.")
            if getattr(self, "using_flash", False):
                q, k = q.transpose(1, 2), k.transpose(1, 2)
                q, k = apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
            else:
                q, k = apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind)

        _ensure_kv_quantizers(self, qkv_format)
        cache_current = not _infinity_is_last_scale(self, scale_schedule, scale_ind)
        if bool(getattr(self, "rescale_qk", False)):
            q, k = self.k_quant.rescale_qk(q, k)

        if _should_use_fused_kv_attn(
            getattr(self, "enable_fused_kv_flashattn", False),
            is_last_scale=not cache_current,
            qkv_format=self.k_quant.qkv_format,
            bits=self.k_quant.quant_bits,
        ):
            from VAR_Q.fused import fused_dequant_attention
            oup = fused_dequant_attention(
                q, self.k_quant, self.v_quant, k_fresh=k, v_fresh=v,
                qkv_format=self.k_quant.qkv_format,
                backend=getattr(self, "fused_kv_backend", "triton"),
                softmax_scale=getattr(self, "scale", None),
            )
            if self.k_quant.qkv_format == "BHLc":
                oup = oup.transpose(1, 2).reshape(B, L, C)
            else:
                oup = oup.reshape(B, L, C)
        else:
            k = _as_attention_dtype(_use_quantizer(self.k_quant, k, cache_current), q.dtype)
            v = _as_attention_dtype(_use_quantizer(self.v_quant, v, cache_current), q.dtype)

            if getattr(self, "using_flash", False):
                flash_attn_func = globals_dict.get("flash_attn_func")
                if flash_attn_func is None:
                    raise RuntimeError("flash_attn_func is not available in the third-party Infinity runtime.")
                if attn_bias_or_two_vector is not None:
                    oup = flash_attn_func(
                        q.to(main_type),
                        k.to(main_type),
                        v.to(main_type),
                        dropout_p=0,
                        softmax_scale=self.scale,
                        VAR_visible_kvlen=attn_bias_or_two_vector[0],
                        VAR_invisible_qlen=attn_bias_or_two_vector[1],
                    )
                else:
                    oup = flash_attn_func(q.to(main_type), k.to(main_type), v.to(main_type), dropout_p=0, softmax_scale=self.scale)
                if oup.shape[1] != L:
                    oup = oup[:, -L:]
                oup = oup.reshape(B, L, C)
            elif getattr(self, "use_flex_attn", False) and attn_fn is not None:
                oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
            else:
                attn_bias = None if attn_bias_or_two_vector is None else attn_bias_or_two_vector.to(x.device)
                oup = _call_slow_attn(globals_dict, q, k, v, self.scale, attn_bias).transpose(1, 2).reshape(B, L, C)

        _release_attention_dequant_workspaces(self)
        del k, v
        _maybe_empty_cuda_cache_after_scale(self)
        return self.proj_drop(self.proj(oup))

    _remember_attr(handle, module, "forward")
    module.forward = types.MethodType(forward_wrapper, module)


def _wrap_infinitystar_forward(handle: HookHandle, module: nn.Module) -> None:
    original = module.forward
    globals_dict = getattr(getattr(original, "__func__", original), "__globals__", {})

    def forward_wrapper(
        self: nn.Module,
        x: torch.Tensor,
        attn_bias_or_two_vector: Any,
        attn_fn: Any = None,
        rope2d_freqs_grid: Any = None,
        scale_schedule: Any = None,
        scale_ind: Any = 0,
        context_info: Any = None,
        last_repetition_step: bool = True,
        ref_text_scale_inds: Optional[List[Any]] = None,
    ):
        if not _attention_enabled(self) or getattr(self, "arch", "qwen") != "qwen":
            return original(
                x,
                attn_bias_or_two_vector,
                attn_fn=attn_fn,
                rope2d_freqs_grid=rope2d_freqs_grid,
                scale_schedule=scale_schedule,
                scale_ind=scale_ind,
                context_info=context_info,
                last_repetition_step=last_repetition_step,
                ref_text_scale_inds=[] if ref_text_scale_inds is None else ref_text_scale_inds,
            )

        _validate_hidden_states_3d(x, "InfinityStar")
        ref_text_scale_inds = [] if ref_text_scale_inds is None else ref_text_scale_inds
        bsz, q_len, _ = x.size()
        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        _validate_qkv_layout(self, query_states, key_states, value_states, "BHLc", q_len, "InfinityStar")

        apply_rotary_emb = globals_dict.get("apply_rotary_emb")
        if _has_rope_cache(rope2d_freqs_grid):
            if apply_rotary_emb is None:
                raise RuntimeError("apply_rotary_emb is not available in the third-party InfinityStar runtime.")
            try:
                query_states, key_states = apply_rotary_emb(query_states, key_states, scale_schedule, rope2d_freqs_grid, 0, getattr(self, "rope2d_normalized_by_hw", 0), scale_ind, context_info)
            except TypeError:
                query_states, key_states = apply_rotary_emb(query_states, key_states, rope2d_freqs_grid)

        _ensure_infinitystar_quantizers(self)
        key_states, value_states = _infinitystar_cache_select(self, key_states, value_states, scale_ind, context_info, last_repetition_step, ref_text_scale_inds)
        if key_states.dtype != query_states.dtype:
            key_states = key_states.to(query_states.dtype)
        if value_states.dtype != query_states.dtype:
            value_states = value_states.to(query_states.dtype)

        repeat_kv = globals_dict.get("repeat_kv")
        if repeat_kv is not None:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        scale = getattr(self, "scale", self.head_dim**-0.5)

        if getattr(self, "use_flex_attn", False) and attn_fn is not None:
            attn_output = attn_fn(query_states, key_states, value_states, scale=scale).transpose(1, 2)
        else:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            sp_manager = globals_dict.get("sp_manager")
            sp_all_to_all = globals_dict.get("sp_all_to_all")
            if bool(getattr(self, "use_sequence_parallel_attn", False)) and sp_all_to_all is not None and _sp_active(sp_manager):
                query_states = sp_all_to_all(query_states, scatter_dim=2, gather_dim=1)
                key_states = sp_all_to_all(key_states, scatter_dim=2, gather_dim=1)
                value_states = sp_all_to_all(value_states, scatter_dim=2, gather_dim=1)
            if bool(getattr(self, "using_flash", True)):
                from flash_attn import flash_attn_func

                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout_p=0.0,
                    softmax_scale=scale,
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_states.transpose(1, 2),
                    key_states.transpose(1, 2),
                    value_states.transpose(1, 2),
                    scale=scale,
                ).transpose(1, 2)
            if bool(getattr(self, "use_sequence_parallel_attn", False)) and sp_all_to_all is not None and _sp_active(sp_manager):
                attn_output = sp_all_to_all(attn_output, scatter_dim=1, gather_dim=2)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        return self.o_proj(attn_output)

    _remember_attr(handle, module, "forward")
    module.forward = types.MethodType(forward_wrapper, module)


def _infinitystar_cache_select(
    module: nn.Module,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scale_ind: Any,
    context_info: Any,
    last_repetition_step: bool,
    ref_text_scale_inds: List[Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(scale_ind, int):
        module.cached_k = module.cached_k or {}
        module.cached_v = module.cached_v or {}
        module.cached_k[scale_ind] = key_states
        module.cached_v[scale_ind] = value_states
        return key_states, value_states

    last_use_by_sid = [-1 for _ in range(len(context_info or {}))]
    for query_sid, query_info in (context_info or {}).items():
        if not isinstance(query_sid, int):
            continue
        for ref_sid in query_info.get("ref_sids", []):
            if isinstance(ref_sid, int) and 0 <= ref_sid < len(last_use_by_sid):
                last_use_by_sid[ref_sid] = max(last_use_by_sid[ref_sid], query_sid)

    cache_current = (
        not bool(getattr(module, "skip_cache_last_scale", True))
        or scale_ind < len(last_use_by_sid)
        and last_use_by_sid[scale_ind] > scale_ind
    )
    if last_repetition_step and cache_current:
        module.k_varq.cache_scale(scale_ind, key_states, overwrite=True, return_dequant=False)
        module.v_varq.cache_scale(scale_ind, value_states, overwrite=True, return_dequant=False)

    info = (context_info or {}).get(scale_ind, {})
    ref_sids = list(info.get("ref_sids", [])) + list(ref_text_scale_inds)
    k_parts: List[Any] = []
    v_parts: List[Any] = []
    has_quantized_ref = False
    for sid in ref_sids:
        if isinstance(sid, int):
            has_quantized_ref = True
            k_parts.append(sid)
            v_parts.append(sid)
        elif getattr(module, "cached_k", None) is not None and sid in module.cached_k:
            k_parts.append(module.cached_k[sid])
            v_parts.append(module.cached_v[sid])
    if has_quantized_ref:
        key_states = module.k_varq.materialize_selected(k_parts + [key_states], cat_dim=2)
        value_states = module.v_varq.materialize_selected(v_parts + [value_states], cat_dim=2)
    elif k_parts:
        key_states = torch.cat(k_parts + [key_states], dim=2)
        value_states = torch.cat(v_parts + [value_states], dim=2)

    # InfinityStar's schedule metadata only records ref_sids. Match the native
    # backend behavior: after each scale, drop any earlier scale whose last
    # reference is before the current scale. Looking for a non-existent
    # clear_sids field keeps every packed scale alive until kv_caching(False),
    # which can make VAR-Q reserve more memory than the dense baseline.
    stale_sids = [
        ref_sid
        for ref_sid in range(scale_ind)
        if ref_sid < len(last_use_by_sid) and last_use_by_sid[ref_sid] < scale_ind
    ]
    if stale_sids:
        module.k_varq.clear_scales(stale_sids)
        module.v_varq.clear_scales(stale_sids)
    return key_states, value_states


def _sp_active(sp_manager: Any) -> bool:
    if sp_manager is None:
        return False
    if hasattr(sp_manager, "is_active"):
        return bool(sp_manager.is_active())
    if hasattr(sp_manager, "sp_on"):
        return bool(sp_manager.sp_on())
    return False


def _wrap_forward(handle: HookHandle, module: nn.Module, model_type: str) -> bool:
    if not hasattr(module, "forward"):
        return False
    if model_type == "var":
        _wrap_var_forward(handle, module)
    elif model_type == "infinity":
        _wrap_infinity_forward(handle, module)
    elif model_type == "infinitystar":
        _wrap_infinitystar_forward(handle, module)
    else:
        return False
    return True


def _configure_attention(
    handle: HookHandle,
    module: nn.Module,
    model_type: str,
    cfg: Dict[str, Any],
    ablation_config: Optional[Dict[str, Any]],
    block_idx: int,
) -> None:
    quant_method = str(cfg.get("quant_method", "VARQ"))
    enable = bool(cfg.get("enable", True))
    q_bits = int(cfg.get("q_bits", 8))
    qkv_format = str(cfg.get("qkv_format", "BHLc" if model_type == "infinitystar" else "BLHc"))

    scalar_attrs = {
        "enable_quantization": enable,
        "q_bits": q_bits,
        "q_bits_k": _role_bits(cfg, "k"),
        "q_bits_v": _role_bits(cfg, "v"),
        "pack_to_int32": bool(cfg.get("pack_to_int32", q_bits in _PACKABLE_BITS)),
        "pack_to_int32_k": _role_pack(cfg, "k"),
        "pack_to_int32_v": _role_pack(cfg, "v"),
        "quant_method": quant_method,
        "qkv_format": qkv_format,
        "kivi_group_size": int(cfg.get("kivi_group_size", 128)),
        "kivi_cali_k_group_size": int(cfg.get("kivi_cali_k_group_size", 128)),
        "kivi_cali_v_group_size": int(cfg.get("kivi_cali_v_group_size", 128)),
        "compression_ratio": float(cfg.get("compression_ratio", cfg.get("ratio", 1.0))),
        "max_scale_seq_len": int(cfg.get("max_scale_seq_len", 0) or 0),
        "rescale_qk": bool(cfg.get("rescale_qk", False)),
        "skip_cache_last_scale": bool(cfg.get("skip_cache_last_scale", model_type in ("var", "infinity", "infinitystar"))),
        "empty_cache_policy": str(cfg.get("empty_cache_policy", "after_generation")),
        "empty_cache_threshold_bytes": int(cfg.get("empty_cache_threshold_bytes", 0) or 0),
        "enable_fused_kv_flashattn": bool(cfg.get("enable_fused_kv_flashattn", False)),
        "fused_kv_backend": str(cfg.get("fused_kv_backend", "triton")),
        "debug_memory": bool(cfg.get("debug_memory", cfg.get("profile_memory", False))),
        "dequant_dtype": str(cfg.get("dequant_dtype", "native")),
        "quant_compute_dtype": str(cfg.get("quant_compute_dtype", "native")),
        "expected_total_seq_len": int(cfg.get("expected_total_seq_len", 0) or 0),
        "preallocate_kv_cache": bool(cfg.get("preallocate_kv_cache", bool(cfg.get("expected_total_seq_len", 0)))),
        "dequant_workspace_policy": str(cfg.get("dequant_workspace_policy", "release")),
        "ablation_config": dict(ablation_config or {}),
        "block_idx": int(getattr(module, "block_idx", block_idx)),
        "_varq_runtime_hooked": True,
        "_varq_hook_model_type": model_type,
    }
    for role in ("q", "k", "v"):
        expected_heads = _infer_expected_heads(module, role)
        if expected_heads is not None:
            scalar_attrs[f"_varq_expected_{role}_heads"] = expected_heads
    if model_type == "infinitystar":
        scalar_attrs.update(
            {
                "_varq_available": enable,
                "force_text_prefill_flashattn": bool(cfg.get("force_text_prefill_flashattn", False)),
                "enable_sageattn": bool(cfg.get("enable_sageattn", getattr(module, "using_sageattn", False))),
                "sageattn_type": str(cfg.get("sageattn_type", getattr(module, "sageattn_type", "sageattn"))),
            }
        )
    for key, value in scalar_attrs.items():
        _set_attr(handle, module, key, value)


def _wrap_kv_caching(handle: HookHandle, module: nn.Module, model_type: str) -> None:
    original = module.kv_caching

    def _clear_quantizer(quantizer: Any) -> None:
        if quantizer is None:
            return
        if hasattr(quantizer, "clear_cache"):
            try:
                quantizer.clear_cache(free_buffers=True)
            except TypeError:
                quantizer.clear_cache()
        elif hasattr(quantizer, "clear_all"):
            quantizer.clear_all()

    def kv_caching_wrapper(self: nn.Module, enable: bool):
        if enable and hasattr(self, "_varq_last_memory_breakdown"):
            delattr(self, "_varq_last_memory_breakdown")
        if not enable:
            snapshot = {
                "packed_kv_bytes": 0,
                "scale_bytes": 0,
                "active_cache_bytes": 0,
                "dequant_workspace_bytes": 0,
                "dequant_workspace_peak_bytes": 0,
                "packed_cache_allocated_bytes": 0,
                "scale_cache_allocated_bytes": 0,
                "cache_buffer_bytes": 0,
                "current_quantized_bytes": 0,
                "current_scale_bytes": 0,
                "temporary_estimated_bytes": 0,
            }
            for attr in ("k_quant", "v_quant"):
                quantizer = getattr(self, attr, None)
                if quantizer is not None and hasattr(quantizer, "memory_breakdown"):
                    stats = quantizer.memory_breakdown()
                    for key in snapshot:
                        snapshot[key] += int(stats.get(key, 0))
            for attr in ("k_varq", "v_varq"):
                quantizer = getattr(self, attr, None)
                if quantizer is not None and hasattr(quantizer, "cache_bytes"):
                    stats = quantizer.cache_bytes()
                    snapshot["packed_kv_bytes"] += int(stats.get("packed_bytes", 0))
                    snapshot["scale_bytes"] += int(stats.get("scale_bytes", 0))
                    snapshot["active_cache_bytes"] += int(stats.get("packed_bytes", 0)) + int(stats.get("scale_bytes", 0))
                    snapshot["dequant_workspace_bytes"] += int(stats.get("dequant_workspace_bytes", 0))
                    snapshot["dequant_workspace_peak_bytes"] += int(stats.get("dequant_workspace_peak_bytes", 0))
            self._varq_last_memory_breakdown = snapshot
        result = original(enable)
        if not enable:
            for attr in ("k_quant", "v_quant", "k_varq", "v_varq"):
                _clear_quantizer(getattr(self, attr, None))
            policy = str(getattr(self, "empty_cache_policy", "after_generation"))
            if torch.cuda.is_available() and policy in ("after_generation", "threshold"):
                torch.cuda.empty_cache()
        if not enable or not bool(getattr(self, "enable_quantization", False)):
            return result
        if model_type == "infinitystar":
            if getattr(self, "k_varq", None) is None:
                self.k_varq = _build_infinitystar_cache_quantizer(**_infinitystar_quantizer_kwargs(self, "k"))
                self.v_varq = _build_infinitystar_cache_quantizer(**_infinitystar_quantizer_kwargs(self, "v"))
            return result
        if getattr(self, "k_quant", None) is None:
            qkv_format = str(getattr(self, "qkv_format", "BLHc"))
            self.k_quant = _build_kv_cache_quantizer(**_kv_quantizer_kwargs(self, "k", qkv_format))
            self.v_quant = _build_kv_cache_quantizer(**_kv_quantizer_kwargs(self, "v", qkv_format))
        return result

    _remember_attr(handle, module, "kv_caching")
    module.kv_caching = types.MethodType(kv_caching_wrapper, module)


def _last_scale_seq_len(model: nn.Module, model_type: str, cfg: Dict[str, Any]) -> int:
    if "last_scale_seq_len" in cfg:
        return int(cfg["last_scale_seq_len"])
    if model_type == "var" and hasattr(model, "patch_nums"):
        patch_nums = getattr(model, "patch_nums")
        if patch_nums:
            return int(patch_nums[-1]) ** 2
    return 0


def _expected_total_seq_len(model: nn.Module, model_type: str, cfg: Dict[str, Any]) -> int:
    if "expected_total_seq_len" in cfg:
        return int(cfg["expected_total_seq_len"])
    if model_type == "var" and hasattr(model, "patch_nums"):
        patch_nums = getattr(model, "patch_nums")
        if patch_nums:
            return int(sum(int(pn) ** 2 for pn in patch_nums))
    return 0


def install_varq_hooks(
    model: nn.Module,
    model_type: str,
    quant_config: Optional[Dict[str, Any]],
    ablation_config: Optional[Dict[str, Any]] = None,
    require_hits: bool = True,
) -> HookHandle:
    normalized_type = str(model_type).lower()
    if normalized_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type={model_type}. Expected one of {SUPPORTED_MODEL_TYPES}.")
    if normalized_type in ("self_forcing", "longlive", "generic_video"):
        raise ValueError(
            f"model_type={model_type} uses the backend-agnostic VideoKVCacheAdapter. "
            "Instantiate VAR_Q.hooks.VideoKVCacheAdapter from the backend attention code and call "
            "adapter.update(k, v, scale_idx=..., num_scales=...)."
        )
    cfg = _normalize_quant_config(quant_config)
    handle = HookHandle(model=model, model_type=normalized_type)
    last_scale_seq_len = _last_scale_seq_len(model, normalized_type, cfg)
    expected_total_seq_len = _expected_total_seq_len(model, normalized_type, cfg)
    scanned_modules = sum(1 for _ in model.named_modules())
    attention_modules = list(_iter_attention_modules(model, normalized_type))
    last_attention_order_idx = len(attention_modules) - 1
    for block_idx, (_name, module) in enumerate(attention_modules):
        _patch_builder_globals(module, normalized_type)
        _configure_attention(handle, module, normalized_type, cfg, ablation_config, block_idx)
        _set_attr(handle, module, "_varq_hook_order_idx", block_idx)
        _set_attr(handle, module, "_varq_last_attention_order_idx", last_attention_order_idx)
        if last_scale_seq_len:
            _set_attr(handle, module, "last_scale_seq_len", last_scale_seq_len)
        if expected_total_seq_len:
            _set_attr(handle, module, "expected_total_seq_len", expected_total_seq_len)
            _set_attr(handle, module, "preallocate_kv_cache", bool(cfg.get("preallocate_kv_cache", True)))
        if hasattr(module, "kv_caching"):
            _wrap_kv_caching(handle, module, normalized_type)
        if _wrap_forward(handle, module, normalized_type):
            handle.modules.append(module)
    handle.hit_count = len(handle.modules)
    if require_hits and handle.hit_count == 0:
        raise _no_attention_hits_error(normalized_type, scanned_modules)
    _set_attr(handle, model, "_varq_hook_handle", handle)
    return handle


def remove_varq_hooks(handle: HookHandle) -> None:
    for patched in reversed(handle.patched_attributes):
        if patched.existed:
            setattr(patched.module, patched.name, patched.original)
        elif hasattr(patched.module, patched.name):
            delattr(patched.module, patched.name)


def is_hooked(model: nn.Module) -> bool:
    return hasattr(model, "_varq_hook_handle")

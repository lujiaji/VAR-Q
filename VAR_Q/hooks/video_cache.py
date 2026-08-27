from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

from VAR_Q.quant import build_kv_cache_quantizer as build_varq_kv_cache_quantizer


def _build_video_kv_cache_quantizer(**kwargs: Any):
    """Build a core VAR-Q quantizer or an existing ablation quantizer."""
    quant_method = str(kwargs.get("quant_method", "VARQ"))
    from ablation import build_kv_cache_quantizer as build_ablation_kv_cache_quantizer
    from ablation import is_ablation_method

    if is_ablation_method(quant_method):
        if str(kwargs.get("dequant_dtype", "bf16")) == "native":
            kwargs = dict(kwargs)
            kwargs["dequant_dtype"] = "bf16"
        return build_ablation_kv_cache_quantizer(**kwargs)
    return build_varq_kv_cache_quantizer(**kwargs)


def _accepts_cache_current(quantizer: Any) -> bool:
    method = getattr(quantizer, "use_var_q", None)
    if method is None:
        return False
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return "cache_current" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _legacy_use_var_q(
    quantizer: Any,
    item: torch.Tensor,
    cache_current: bool,
    qkv_format: str,
) -> torch.Tensor:
    """Adapt the older KIVI/FlexGen cache API to the video-cache contract."""
    quantizer.quant(item)
    if cache_current:
        quantizer.cache()
        return quantizer.dequant_all()

    current = quantizer.dequant_current()
    if bool(getattr(quantizer, "_replace_cache_on_next_cache", False)):
        return current
    if getattr(quantizer, "cached_item", None) is None:
        return current
    seq_dim = 1 if qkv_format == "BLHc" else 2
    return torch.cat((quantizer.dequant_all(), current), dim=seq_dim)


def _materialize_cached_and_fresh(
    quantizer: Any,
    current: torch.Tensor,
    qkv_format: str,
) -> torch.Tensor:
    """Return the committed cache followed by full-precision current K/V."""
    explicit_length = getattr(quantizer, "cached_len", None)
    cached_item = getattr(quantizer, "cached_item", None)
    if explicit_length is not None:
        has_cache = int(explicit_length) > 0
    else:
        has_cache = torch.is_tensor(cached_item) and cached_item.numel() > 0
    if not has_cache:
        return current
    cached = quantizer.dequant_all()
    if cached.dtype != current.dtype:
        cached = cached.to(current.dtype)
    seq_dim = 1 if qkv_format == "BLHc" else 2
    return torch.cat((cached, current), dim=seq_dim)


def _tensor_nbytes(value: Any) -> int:
    if not torch.is_tensor(value):
        return 0
    return int(value.numel() * value.element_size())


def _normalize_video_quant_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(config or {})
    if "kv_quant" in cfg:
        cfg = dict(cfg["kv_quant"])
    if "enable_quantization" in cfg and "enable" not in cfg:
        cfg["enable"] = cfg["enable_quantization"]
    cfg.setdefault("enable", True)
    cfg.setdefault("quant_method", "VARQ")
    cfg.setdefault("qkv_format", "BLHc")
    cfg.setdefault("q_bits", 8)
    cfg.setdefault("pack_to_int32", True)
    cfg.setdefault("compression_ratio", 1.0)
    cfg.setdefault("max_scale_seq_len", 1560)
    cfg.setdefault("skip_cache_last_scale", True)
    cfg.setdefault("dequant_dtype", "native")
    cfg.setdefault("quant_compute_dtype", "native")
    cfg.setdefault("dequant_workspace_policy", "release")
    return cfg


@dataclass
class VideoKVCacheAdapter:
    """Backend-agnostic VAR-Q KV-cache adapter for next-frame/chunk video models.

    Use this when a backend attention implementation already has K/V tensors in
    hand. The adapter does not assume a specific module class or forward
    signature, which makes it suitable for Self-Forcing, LongLive, Wan-style
    transformers, and other video AR backends.
    """

    quant_config: Dict[str, Any]
    ablation_config: Optional[Dict[str, Any]] = None
    kv_role_prefix: str = "video"

    def __post_init__(self) -> None:
        raw_config = dict(self.quant_config or {})
        nested_config = raw_config.get("kv_quant", raw_config)
        if self.ablation_config is None and isinstance(nested_config, dict):
            nested_ablation = nested_config.get("ablation")
            if isinstance(nested_ablation, dict):
                self.ablation_config = dict(nested_ablation)
        self.quant_config = _normalize_video_quant_config(self.quant_config)
        self.enabled = bool(self.quant_config.get("enable", True))
        self.fused_enabled = bool(
            self.quant_config.get("enable_fused_kv_flashattn", False)
        )
        self.fused_backend = str(
            self.quant_config.get("fused_kv_backend", "cuda_direct")
        )
        self.fused_strict = bool(
            self.quant_config.get("fused_kv_strict", True)
        )
        self.fused_calls = 0
        self.fused_fallbacks = 0
        self.k_quant = self._build_quantizer("k")
        self.v_quant = self._build_quantizer("v")
        self._k_accepts_cache_current = _accepts_cache_current(self.k_quant)
        self._v_accepts_cache_current = _accepts_cache_current(self.v_quant)

    @classmethod
    def from_config(
        cls,
        quant_config: Dict[str, Any],
        ablation_config: Optional[Dict[str, Any]] = None,
        kv_role_prefix: str = "video",
    ) -> "VideoKVCacheAdapter":
        return cls(quant_config=quant_config, ablation_config=ablation_config, kv_role_prefix=kv_role_prefix)

    @classmethod
    def from_env(cls, env_var: str = "VARQ_CONFIG_FILE", kv_role_prefix: str = "video") -> "VideoKVCacheAdapter":
        config_path = os.environ.get(env_var)
        if not config_path:
            raise EnvironmentError(f"{env_var} is not set; pass a VAR-Q JSON config path through the launcher.")
        with open(config_path, "r") as f:
            raw_config = json.load(f)
        quant_config = raw_config.get("quantization", raw_config.get("kv_quant", raw_config))
        ablation_config = raw_config.get("ablation")
        if isinstance(quant_config, dict) and "ablation" in quant_config and ablation_config is None:
            ablation_config = quant_config.get("ablation")
        return cls(quant_config=quant_config, ablation_config=ablation_config, kv_role_prefix=kv_role_prefix)

    def _build_quantizer(self, role: str):
        cfg = self.quant_config
        role_bits = int(cfg.get(f"q_bits_{role}", cfg.get("q_bits", 8)))
        role_pack = bool(cfg.get(f"pack_to_int32_{role}", cfg.get("pack_to_int32", True)))
        return _build_video_kv_cache_quantizer(
            quant_bits=role_bits,
            qkv_format=str(cfg.get("qkv_format", "BLHc")),
            quant_method=str(cfg.get("quant_method", "VARQ")),
            kv_role=role,
            blk_idx=int(cfg.get("block_idx", 0)),
            pack_to_int32=role_pack,
            kivi_group_size=int(cfg.get("kivi_group_size", 128)),
            kivi_cali_k_group_size=int(cfg.get("kivi_cali_k_group_size", 128)),
            kivi_cali_v_group_size=int(cfg.get("kivi_cali_v_group_size", 128)),
            compression_ratio=float(cfg.get("compression_ratio", 1.0)),
            max_scale_seq_len=int(cfg.get("max_scale_seq_len", 1560)) or None,
            rescale_qk=bool(cfg.get("rescale_qk", False)),
            debug=bool(cfg.get("debug_memory", cfg.get("profile_memory", False))),
            ablation_config=self.ablation_config,
            dequant_dtype=str(cfg.get("dequant_dtype", "native")),
            quant_compute_dtype=str(cfg.get("quant_compute_dtype", "native")),
            expected_total_seq_len=int(cfg.get("expected_total_seq_len", 0) or 0) or None,
            preallocate_kv_cache=bool(cfg.get("preallocate_kv_cache", bool(cfg.get("expected_total_seq_len", 0)))),
            dequant_workspace_policy=str(cfg.get("dequant_workspace_policy", "release")),
        )

    def reset(self) -> None:
        for quantizer in (self.k_quant, self.v_quant):
            clear_cache = getattr(quantizer, "clear_cache")
            try:
                clear_cache(free_buffers=True)
            except TypeError as exc:
                if "free_buffers" not in str(exc):
                    raise
                clear_cache()
        self.release_workspaces()
        self.fused_calls = 0
        self.fused_fallbacks = 0

    def release_workspaces(self) -> None:
        for quantizer in (self.k_quant, self.v_quant):
            release = getattr(quantizer, "release_dequant_workspace", None)
            if release is not None:
                release()

    def cached_length(self, role: str = "k") -> int:
        if role not in ("k", "v"):
            raise ValueError(f"Unsupported KV role: {role}")
        quantizer = self.k_quant if role == "k" else self.v_quant
        explicit_length = getattr(quantizer, "cached_len", None)
        if explicit_length is not None:
            return int(explicit_length)
        cached_item = getattr(quantizer, "cached_item", None)
        if not torch.is_tensor(cached_item):
            return 0
        seq_dim = 1 if str(self.quant_config.get("qkv_format", "BLHc")) == "BLHc" else 2
        return int(cached_item.shape[seq_dim])

    def commit(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Quantize and retain a committed segment without dequantizing it."""
        if not self.enabled:
            return
        for quantizer, tensor in ((self.k_quant, key), (self.v_quant, value)):
            quant_and_cache = getattr(quantizer, "quant_and_cache", None)
            if quant_and_cache is not None:
                quant_and_cache(tensor.detach())
            else:
                quantizer.quant(tensor.detach())
                quantizer.cache()

    def has_fused_cache(self) -> bool:
        return bool(
            self.enabled
            and self.fused_enabled
            and self.cached_length("k") > 0
            and self.cached_length("v") > 0
            and str(self.quant_config.get("quant_method", "VARQ")).upper() == "VARQ"
        )

    def dequantized(self, role: str) -> torch.Tensor:
        if role not in ("k", "v"):
            raise ValueError(f"Unsupported KV role: {role}")
        quantizer = self.k_quant if role == "k" else self.v_quant
        return quantizer.dequant_all()

    def attend_fused(
        self,
        query: torch.Tensor,
        fresh_key: torch.Tensor,
        fresh_value: torch.Tensor,
        *,
        backend: Optional[str] = None,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Attend over a packed BLHc prefix and a full-precision suffix."""
        if str(self.quant_config.get("qkv_format", "BLHc")) != "BLHc":
            raise ValueError("The video fused adapter requires qkv_format='BLHc'.")
        if fresh_key.ndim != 4 or fresh_key.shape[-1] not in (64, 128):
            raise ValueError(
                "The fused video adapter supports BLHc head_dim 64 or 128, "
                f"got {tuple(fresh_key.shape)}"
            )
        if not self.has_fused_cache():
            raise RuntimeError("Fused attention requires a committed packed prefix.")
        from VAR_Q.fused import fused_dequant_attention

        output = fused_dequant_attention(
            query.transpose(1, 2).contiguous(),
            self.k_quant,
            self.v_quant,
            fresh_key.transpose(1, 2).contiguous(),
            fresh_value.transpose(1, 2).contiguous(),
            qkv_format="BHLc",
            backend=backend or self.fused_backend,
            softmax_scale=softmax_scale,
        )
        self.fused_calls += 1
        return output.transpose(1, 2).contiguous()

    def should_cache(self, scale_idx: Optional[int] = None, num_scales: Optional[int] = None) -> bool:
        if not bool(self.quant_config.get("skip_cache_last_scale", True)):
            return True
        if scale_idx is None or num_scales is None:
            return True
        return int(scale_idx) < int(num_scales) - 1

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scale_idx: Optional[int] = None,
        num_scales: Optional[int] = None,
        cache_current: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Commit cacheable K/V and return all K/V visible to attention.

        Repeated denoising forwards and the final chunk remain full precision;
        only committed history is retained in low-bit form.
        """
        if not self.enabled:
            return key, value
        cache = self.should_cache(scale_idx, num_scales) if cache_current is None else bool(cache_current)
        if bool(self.quant_config.get("rescale_qk", False)):
            # Q is not available in this generic adapter. Backends that need
            # rescale_qk should call the lower-level quantizer from their own
            # attention wrapper where Q is in scope.
            raise ValueError("rescale_qk=True requires Q and is not supported by VideoKVCacheAdapter.update().")
        qkv_format = str(self.quant_config.get("qkv_format", "BLHc"))
        if cache:
            if self._k_accepts_cache_current:
                key_out = self.k_quant.use_var_q(key, cache_current=True)
            else:
                key_out = _legacy_use_var_q(self.k_quant, key, True, qkv_format)
            if self._v_accepts_cache_current:
                value_out = self.v_quant.use_var_q(value, cache_current=True)
            else:
                value_out = _legacy_use_var_q(self.v_quant, value, True, qkv_format)
        else:
            key_out = _materialize_cached_and_fresh(self.k_quant, key, qkv_format)
            value_out = _materialize_cached_and_fresh(self.v_quant, value, qkv_format)
        if key_out.dtype != key.dtype:
            key_out = key_out.to(key.dtype)
        if value_out.dtype != value.dtype:
            value_out = value_out.to(value.dtype)
        key_out = key_out.contiguous()
        value_out = value_out.contiguous()
        for quantizer in (self.k_quant, self.v_quant):
            maybe_release = getattr(quantizer, "maybe_release_dequant_workspace", None)
            if maybe_release is not None:
                maybe_release()
        return key_out, value_out

    def memory_breakdown(self) -> Dict[str, int]:
        def stats(quantizer: Any) -> Dict[str, int]:
            get_stats = getattr(quantizer, "memory_breakdown", None)
            if get_stats is not None:
                return {key: int(value) for key, value in get_stats().items()}
            packed_bytes = _tensor_nbytes(getattr(quantizer, "cached_item", None))
            scale_bytes = _tensor_nbytes(getattr(quantizer, "cached_scale", None))
            return {
                "packed_kv_bytes": packed_bytes,
                "scale_bytes": scale_bytes,
                "active_cache_bytes": packed_bytes + scale_bytes,
            }

        k_stats = stats(self.k_quant)
        v_stats = stats(self.v_quant)
        keys = set(k_stats) | set(v_stats)
        return {key: int(k_stats.get(key, 0)) + int(v_stats.get(key, 0)) for key in keys}

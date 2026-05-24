from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

from VAR_Q.quant import build_kv_cache_quantizer


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
        self.quant_config = _normalize_video_quant_config(self.quant_config)
        self.enabled = bool(self.quant_config.get("enable", True))
        self.k_quant = self._build_quantizer("k")
        self.v_quant = self._build_quantizer("v")

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
        return build_kv_cache_quantizer(
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
        self.k_quant.clear_cache(free_buffers=True)
        self.v_quant.clear_cache(free_buffers=True)
        self.release_workspaces()

    def release_workspaces(self) -> None:
        self.k_quant.release_dequant_workspace()
        self.v_quant.release_dequant_workspace()

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
        """Quantize current K/V, return dequantized K/V visible to attention."""
        if not self.enabled:
            return key, value
        cache = self.should_cache(scale_idx, num_scales) if cache_current is None else bool(cache_current)
        if bool(self.quant_config.get("rescale_qk", False)):
            # Q is not available in this generic adapter. Backends that need
            # rescale_qk should call the lower-level quantizer from their own
            # attention wrapper where Q is in scope.
            raise ValueError("rescale_qk=True requires Q and is not supported by VideoKVCacheAdapter.update().")
        key_out = self.k_quant.use_var_q(key, cache_current=cache)
        value_out = self.v_quant.use_var_q(value, cache_current=cache)
        if key_out.dtype != key.dtype:
            key_out = key_out.to(key.dtype)
        if value_out.dtype != value.dtype:
            value_out = value_out.to(value.dtype)
        key_out = key_out.contiguous()
        value_out = value_out.contiguous()
        self.k_quant.maybe_release_dequant_workspace()
        self.v_quant.maybe_release_dequant_workspace()
        return key_out, value_out

    def memory_breakdown(self) -> Dict[str, int]:
        k_stats = self.k_quant.memory_breakdown()
        v_stats = self.v_quant.memory_breakdown()
        keys = set(k_stats) | set(v_stats)
        return {key: int(k_stats.get(key, 0)) + int(v_stats.get(key, 0)) for key in keys}

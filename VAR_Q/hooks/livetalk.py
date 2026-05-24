from __future__ import annotations

from dataclasses import dataclass, field
import math
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from VAR_Q.quant import build_kv_cache_quantizer


def _normalize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(config or {})
    if "quantization" in cfg:
        cfg = dict(cfg["quantization"])
    if "kv_quant" in cfg:
        cfg = dict(cfg["kv_quant"])
    if "enable_quantization" in cfg and "enable" not in cfg:
        cfg["enable"] = cfg["enable_quantization"]
    cfg.setdefault("enable", True)
    cfg.setdefault("quant_method", "VARQ")
    cfg.setdefault("q_bits", 4)
    cfg.setdefault("qkv_format", "BLHc")
    cfg.setdefault("pack_to_int32", True)
    cfg.setdefault("compression_ratio", 1.0)
    cfg.setdefault("max_scale_seq_len", 1024)
    cfg.setdefault("quant_compute_dtype", "native")
    cfg.setdefault("dequant_dtype", "native")
    cfg.setdefault("dequant_workspace_policy", "release")
    return cfg


@dataclass
class _PatchedAttribute:
    obj: Any
    name: str
    original: Any


@dataclass
class LiveTalkHookHandle:
    target: nn.Module
    quant_config: Dict[str, Any]
    patched_attributes: List[_PatchedAttribute] = field(default_factory=list)
    modules: List[nn.Module] = field(default_factory=list)


class LiveTalkLayerKVCache:
    """Chunk-addressable VAR-Q cache for LiveTalk/Wan causal self-attention.

    LiveTalk repeatedly overwrites the same token range during denoising and
    only commits the clean chunk at the end. A single append-only VAR_Q cache is
    therefore not enough; this wrapper stores one VAR_Q quantizer pair per token
    segment and replaces overlapping segments on overwrite.
    """

    def __init__(self, quant_config: Dict[str, Any], layer_idx: int, sink_tokens: int = 0) -> None:
        self.quant_config = dict(quant_config)
        self.layer_idx = int(layer_idx)
        self.sink_tokens = int(sink_tokens)
        self.segments: List[Dict[str, Any]] = []

    def _new_quantizer(self, role: str):
        cfg = self.quant_config
        bits = int(cfg.get(f"q_bits_{role}", cfg.get("q_bits", 4)))
        return build_kv_cache_quantizer(
            quant_bits=bits,
            qkv_format=str(cfg.get("qkv_format", "BLHc")),
            quant_method=str(cfg.get("quant_method", "VARQ")),
            kv_role=role,
            blk_idx=self.layer_idx,
            pack_to_int32=bool(cfg.get(f"pack_to_int32_{role}", cfg.get("pack_to_int32", True))),
            compression_ratio=float(cfg.get("compression_ratio", 1.0)),
            max_scale_seq_len=int(cfg.get("max_scale_seq_len", 1024) or 0) or None,
            rescale_qk=bool(cfg.get("rescale_qk", False)),
            debug=bool(cfg.get("debug_memory", cfg.get("profile_memory", False))),
            dequant_dtype=str(cfg.get("dequant_dtype", "native")),
            quant_compute_dtype=str(cfg.get("quant_compute_dtype", "native")),
            expected_total_seq_len=int(cfg.get("expected_total_seq_len", 0) or 0) or None,
            preallocate_kv_cache=bool(cfg.get("preallocate_kv_cache", False)),
            dequant_workspace_policy=str(cfg.get("dequant_workspace_policy", "release")),
        )

    def reset(self) -> None:
        for segment in self.segments:
            segment["k_quant"].clear_cache(free_buffers=True)
            segment["v_quant"].clear_cache(free_buffers=True)
        self.segments.clear()

    def store(self, start: int, end: int, key: torch.Tensor, value: torch.Tensor) -> None:
        start = int(start)
        end = int(end)
        if end <= start:
            return
        self.segments = [
            segment
            for segment in self.segments
            if int(segment["end"]) <= start or int(segment["start"]) >= end
        ]
        k_quant = self._new_quantizer("k")
        v_quant = self._new_quantizer("v")
        k_quant.quant_and_cache(key.detach())
        v_quant.quant_and_cache(value.detach())
        self.segments.append(
            {
                "start": start,
                "end": end,
                "k_quant": k_quant,
                "v_quant": v_quant,
            }
        )
        self.segments.sort(key=lambda segment: int(segment["start"]))

    def prune(self, current_end: int, max_attention_size: int) -> None:
        if max_attention_size <= 0:
            return
        cutoff = max(0, int(current_end) - int(max_attention_size))
        self.segments = [
            segment
            for segment in self.segments
            if int(segment["end"]) > cutoff or int(segment["start"]) < self.sink_tokens
        ]

    @staticmethod
    def _slice_tokens(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return tensor[:, int(start):int(end)]

    def _segment_tensor(self, segment: Dict[str, Any], role: str) -> torch.Tensor:
        quantizer = segment[f"{role}_quant"]
        tensor = quantizer.dequant_all()
        quantizer.maybe_release_dequant_workspace()
        return tensor

    def get_range(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(start)
        end = int(end)
        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        for segment in self.segments:
            seg_start = int(segment["start"])
            seg_end = int(segment["end"])
            ov_start = max(start, seg_start)
            ov_end = min(end, seg_end)
            if ov_start >= ov_end:
                continue
            rel_start = ov_start - seg_start
            rel_end = ov_end - seg_start
            k_parts.append(self._slice_tokens(self._segment_tensor(segment, "k"), rel_start, rel_end))
            v_parts.append(self._slice_tokens(self._segment_tensor(segment, "v"), rel_start, rel_end))
        if not k_parts:
            raise RuntimeError(f"No LiveTalk VAR-Q KV segment covers token range [{start}, {end}).")
        return torch.cat(k_parts, dim=1).contiguous(), torch.cat(v_parts, dim=1).contiguous()

    def memory_breakdown(self) -> Dict[str, int]:
        total: Dict[str, int] = {}
        for segment in self.segments:
            for role in ("k", "v"):
                stats = segment[f"{role}_quant"].memory_breakdown()
                for key, value in stats.items():
                    total[key] = int(total.get(key, 0)) + int(value)
        return total


def _remember(handle: LiveTalkHookHandle, obj: Any, name: str) -> None:
    handle.patched_attributes.append(_PatchedAttribute(obj=obj, name=name, original=getattr(obj, name)))


def _restore(handle: LiveTalkHookHandle) -> None:
    for patch in reversed(handle.patched_attributes):
        setattr(patch.obj, patch.name, patch.original)


def _find_livetalk_attention_modules(target: nn.Module) -> List[nn.Module]:
    modules: List[nn.Module] = []
    for module in target.modules():
        if module.__class__.__name__ != "CausalSelfAttention":
            continue
        if all(hasattr(module, attr) for attr in ("q", "k", "v", "o", "norm_q", "norm_k", "attn")):
            modules.append(module)
    return modules


def _make_livetalk_forward(original_forward):
    def forward_wrapper(
        self,
        x,
        freqs,
        block_mask=None,
        grid_sizes=None,
        kv_cache=None,
        current_start=0,
        cache_start=None,
    ):
        if kv_cache is None or not bool(getattr(self, "_varq_livetalk_enabled", False)):
            return original_forward(x, freqs, block_mask=block_mask, grid_sizes=grid_sizes, kv_cache=kv_cache, current_start=current_start, cache_start=cache_start)

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        roped_query = self._varq_rope_apply(q, freqs, self.num_heads).type_as(v)
        roped_key = self._varq_rope_apply(k, freqs, self.num_heads).type_as(v)

        bsz, num_new_tokens, _ = q.shape
        roped_key_4d = roped_key.view(bsz, num_new_tokens, self.num_heads, self.head_dim)
        value_4d = v.view(bsz, num_new_tokens, self.num_heads, self.head_dim)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item() if grid_sizes is not None else int(num_new_tokens)
        current_start = int(current_start or 0)
        current_end = current_start + int(num_new_tokens)
        sink_tokens = int(getattr(self, "sink_size", 0) or 0) * int(frame_seqlen)
        max_attention_size = int(getattr(self, "max_attention_size", 0) or 0)

        state = kv_cache.get("_varq_state")
        if state is None:
            state = LiveTalkLayerKVCache(
                quant_config=getattr(self, "_varq_livetalk_quant_config"),
                layer_idx=int(getattr(self, "_varq_livetalk_layer_idx", 0)),
                sink_tokens=sink_tokens,
            )
            kv_cache["_varq_state"] = state

        state.store(current_start, current_end, roped_key_4d, value_4d)
        state.prune(current_end, max_attention_size)

        if max_attention_size > 0 and current_end > max_attention_size:
            attn_start = max(0, current_end - max_attention_size)
            if sink_tokens > 0:
                # Keep sink tokens when they are still cached, then append the local window.
                recent_k, recent_v = state.get_range(attn_start, current_end)
                sink_k, sink_v = state.get_range(0, min(sink_tokens, current_end))
                key = torch.cat([sink_k, recent_k], dim=1).contiguous()
                value = torch.cat([sink_v, recent_v], dim=1).contiguous()
            else:
                key, value = state.get_range(attn_start, current_end)
        else:
            key, value = state.get_range(0, current_end)

        key = key.reshape(bsz, -1, self.dim)
        value = value.reshape(bsz, -1, self.dim)
        out = self.attn(roped_query, key.type_as(roped_query), value.type_as(roped_query))
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(current_end)
        return self.o(out)

    return forward_wrapper


def _patch_pipeline_cache(handle: LiveTalkHookHandle, pipeline: nn.Module, attention_modules: List[nn.Module]) -> None:
    if not hasattr(pipeline, "_initialize_kv_cache"):
        return

    original_init = pipeline._initialize_kv_cache
    original_reset = getattr(pipeline, "_reset_caches", None)

    def initialize_kv_cache_wrapper(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        kv_cache = []
        for layer_idx, module in enumerate(attention_modules):
            frame_seq_len = int(getattr(self, "frame_seq_length", 1024))
            local_attn_size = int(getattr(module, "local_attn_size", getattr(self, "local_attn_size", -1)))
            kv_capacity = 32760 if local_attn_size == -1 else local_attn_size * frame_seq_len
            kv_cache.append(
                {
                    "k": torch.empty((batch_size, 0, int(getattr(module, "dim", 0))), dtype=dtype, device=device),
                    "v": torch.empty((batch_size, 0, int(getattr(module, "dim", 0))), dtype=dtype, device=device),
                    "capacity": kv_capacity,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "_varq_state": LiveTalkLayerKVCache(
                        quant_config=handle.quant_config,
                        layer_idx=layer_idx,
                        sink_tokens=int(getattr(module, "sink_size", 0) or 0) * frame_seq_len,
                    ),
                }
            )
        self.kv_cache1 = kv_cache

    def reset_caches_wrapper(self, device: torch.device):
        if original_reset is not None:
            original_reset(device)
        for cache in getattr(self, "kv_cache1", []) or []:
            state = cache.get("_varq_state")
            if state is not None:
                state.reset()
            cache["global_end_index"] = torch.tensor([0], dtype=torch.long, device=device)
            cache["local_end_index"] = torch.tensor([0], dtype=torch.long, device=device)

    _remember(handle, pipeline, "_initialize_kv_cache")
    pipeline._initialize_kv_cache = types.MethodType(initialize_kv_cache_wrapper, pipeline)
    if original_reset is not None:
        _remember(handle, pipeline, "_reset_caches")
        pipeline._reset_caches = types.MethodType(reset_caches_wrapper, pipeline)


def install_livetalk_hooks(target: nn.Module, quant_config: Optional[Dict[str, Any]]) -> LiveTalkHookHandle:
    cfg = _normalize_config(quant_config)
    handle = LiveTalkHookHandle(target=target, quant_config=cfg)
    if not bool(cfg.get("enable", True)):
        return handle

    attention_modules = _find_livetalk_attention_modules(target)
    if not attention_modules:
        raise RuntimeError("No LiveTalk CausalSelfAttention modules found; check that the LiveTalk pipeline/model is initialized.")

    for layer_idx, module in enumerate(attention_modules):
        _remember(handle, module, "forward")
        module._varq_livetalk_enabled = True
        module._varq_livetalk_quant_config = cfg
        module._varq_livetalk_layer_idx = layer_idx
        func = getattr(module.forward, "__func__", module.forward)
        module._varq_rope_apply = func.__globals__["rope_apply"]
        module.forward = types.MethodType(_make_livetalk_forward(module.forward), module)
        handle.modules.append(module)

    _patch_pipeline_cache(handle, target, attention_modules)
    setattr(target, "_varq_livetalk_hook_handle", handle)
    return handle


def remove_livetalk_hooks(handle: LiveTalkHookHandle) -> None:
    _restore(handle)
    if hasattr(handle.target, "_varq_livetalk_hook_handle"):
        delattr(handle.target, "_varq_livetalk_hook_handle")


def collect_livetalk_memory_breakdown(target: nn.Module) -> Dict[str, int]:
    cuda_keys = {
        "cuda_memory_allocated",
        "cuda_max_memory_allocated",
        "cuda_memory_reserved",
        "cuda_max_memory_reserved",
    }
    totals: Dict[str, int] = {}
    for cache in getattr(target, "kv_cache1", []) or []:
        state = cache.get("_varq_state") if isinstance(cache, dict) else None
        if state is None:
            continue
        stats = state.memory_breakdown()
        for key, value in stats.items():
            if key in cuda_keys:
                continue
            totals[key] = int(totals.get(key, 0)) + int(value)
    if torch.cuda.is_available():
        totals.update(
            {
                "cuda_memory_allocated": int(torch.cuda.memory_allocated()),
                "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                "cuda_memory_reserved": int(torch.cuda.memory_reserved()),
                "cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved()),
            }
        )
    else:
        totals.update(
            {
                "cuda_memory_allocated": 0,
                "cuda_max_memory_allocated": 0,
                "cuda_memory_reserved": 0,
                "cuda_max_memory_reserved": 0,
            }
        )
    return totals

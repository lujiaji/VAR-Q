"""Runtime VAR-Q integration for an unmodified LongLive checkout.

Call :func:`install_longlive_hooks` after constructing the upstream pipeline.
The installer keeps committed history in compressed per-chunk segments while
the current denoising chunk remains in the model's native floating dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import math
import os
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from VAR_Q.hooks.video_cache import VideoKVCacheAdapter


@dataclass
class _Segment:
    start: int
    end: int
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    adapter: Optional[VideoKVCacheAdapter] = None

    def release(self) -> None:
        if self.adapter is not None:
            self.adapter.reset()
        self.key = None
        self.value = None


class _PackedQuantView:
    """Present multiple selected segment quantizers as one packed cache."""

    def __init__(self, quantizers: List[Any]) -> None:
        if not quantizers:
            raise RuntimeError("Packed LongLive attention requires cached segments.")
        template = quantizers[0]
        self.quant_bits = int(template.quant_bits)
        self.qkv_format = str(template.qkv_format)
        if self.qkv_format == "BLHc":
            seq_dim = 1
        elif self.qkv_format == "BHLc":
            seq_dim = 2
        else:
            raise ValueError(f"Unsupported LongLive packed layout: {self.qkv_format}")
        scale_dim = seq_dim
        packed_parts: List[torch.Tensor] = []
        scale_parts: List[torch.Tensor] = []
        self._scale_L_counts: List[int] = []
        for quantizer in quantizers:
            if int(quantizer.quant_bits) != self.quant_bits:
                raise ValueError("LongLive fused segments must use one KV precision.")
            if str(quantizer.qkv_format) != self.qkv_format:
                raise ValueError("LongLive fused segments must use one packed layout.")
            cached_item = quantizer._valid_cached_item()
            if not torch.is_tensor(cached_item):
                raise RuntimeError("LongLive fused segment is missing packed KV data.")
            valid_scale = quantizer._valid_cached_scale()
            if not torch.is_tensor(valid_scale):
                raise RuntimeError("LongLive fused segment is missing scale metadata.")
            packed_parts.append(cached_item)
            scale_parts.append(valid_scale)
            self._scale_L_counts.extend(int(v) for v in quantizer._scale_L_counts)
        self.cached_item = torch.cat(packed_parts, dim=seq_dim).contiguous()
        self.cached_scale = torch.cat(scale_parts, dim=scale_dim).contiguous()
        self.cached_len = int(self.cached_item.shape[seq_dim])

    def _valid_cached_item(self) -> torch.Tensor:
        return self.cached_item

    def _valid_cached_scale(self) -> torch.Tensor:
        return self.cached_scale



class LongLiveKVCacheState:
    """Compressed, segmented KV history for one LongLive attention layer."""

    def __init__(
        self,
        quant_config: Dict[str, Any],
        layer_idx: int,
        sink_tokens: int,
        max_attention_size: int,
    ) -> None:
        self.quant_config = dict(quant_config)
        self.layer_idx = int(layer_idx)
        self.sink_tokens = max(0, int(sink_tokens))
        self.max_attention_size = int(max_attention_size)
        self.segments: List[_Segment] = []
        self.fused_calls = 0
        self.fused_fallbacks = 0

    def _new_adapter(self, segment_index: int) -> VideoKVCacheAdapter:
        config = dict(self.quant_config)
        config["block_idx"] = self.layer_idx
        return VideoKVCacheAdapter.from_config(
            config,
            kv_role_prefix=f"longlive.block{self.layer_idx}.segment{segment_index}",
        )

    def reset(self) -> None:
        for segment in self.segments:
            segment.release()
        self.segments.clear()
        self.fused_calls = 0
        self.fused_fallbacks = 0

    def commit(
        self,
        start: int,
        end: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Replace an overlapping chunk and retain it as compressed history."""
        start, end = int(start), int(end)
        if end <= start:
            return

        survivors: List[_Segment] = []
        for segment in self.segments:
            if segment.end <= start or segment.start >= end:
                survivors.append(segment)
            else:
                segment.release()
        self.segments = survivors

        adapter = self._new_adapter(len(self.segments))
        adapter.commit(key, value)
        self.segments.append(_Segment(start=start, end=end, adapter=adapter))
        self.segments.sort(key=lambda segment: segment.start)
        self.prune(end)

    def prune(self, current_end: int) -> None:
        if self.max_attention_size <= 0:
            return
        local_budget = max(0, self.max_attention_size - self.sink_tokens)
        cutoff = max(self.sink_tokens, int(current_end) - local_budget)
        kept: List[_Segment] = []
        for segment in self.segments:
            if segment.start < self.sink_tokens or segment.end > cutoff:
                kept.append(segment)
            else:
                segment.release()
        self.segments = kept

    @staticmethod
    def _slice(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return tensor[:, int(start):int(end)]

    @staticmethod
    def _role_tensor(segment: _Segment, role: str) -> torch.Tensor:
        if segment.adapter is not None:
            return segment.adapter.dequantized(role)
        tensor = segment.key if role == "k" else segment.value
        if tensor is None:
            raise RuntimeError("LongLive cache segment has no tensor payload.")
        return tensor

    def get_range(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end = int(start), int(end)
        if end <= start:
            raise ValueError(f"Invalid LongLive cache range [{start}, {end}).")
        key_parts: List[torch.Tensor] = []
        value_parts: List[torch.Tensor] = []
        cursor = start
        for segment in self.segments:
            if segment.end <= cursor:
                continue
            if segment.start > cursor:
                break
            overlap_start = max(cursor, segment.start)
            overlap_end = min(end, segment.end)
            if overlap_end <= overlap_start:
                continue
            relative_start = overlap_start - segment.start
            relative_end = overlap_end - segment.start
            key_parts.append(
                self._slice(self._role_tensor(segment, "k"), relative_start, relative_end)
            )
            value_parts.append(
                self._slice(self._role_tensor(segment, "v"), relative_start, relative_end)
            )
            cursor = overlap_end
            if cursor >= end:
                break
        if cursor < end or not key_parts:
            ranges = [(segment.start, segment.end) for segment in self.segments]
            raise RuntimeError(
                f"LongLive cache has no contiguous coverage for [{start}, {end}); "
                f"segments={ranges}"
            )
        return (
            torch.cat(key_parts, dim=1).contiguous(),
            torch.cat(value_parts, dim=1).contiguous(),
        )

    def attention_kv(
        self,
        current_start: int,
        current_end: int,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        current_start, current_end = int(current_start), int(current_end)
        if current_start <= 0 or not self.segments:
            return current_key, current_value

        if self.max_attention_size > 0:
            local_budget = max(0, self.max_attention_size - self.sink_tokens)
            past_start = max(self.sink_tokens, current_end - local_budget)
        else:
            past_start = 0

        key_parts: List[torch.Tensor] = []
        value_parts: List[torch.Tensor] = []
        sink_end = min(self.sink_tokens, current_start)
        if sink_end > 0:
            sink_key, sink_value = self.get_range(0, sink_end)
            key_parts.append(sink_key)
            value_parts.append(sink_value)
        recent_start = max(past_start, sink_end)
        if recent_start < current_start:
            recent_key, recent_value = self.get_range(recent_start, current_start)
            key_parts.append(recent_key)
            value_parts.append(recent_value)
        key_parts.append(current_key)
        value_parts.append(current_value)
        return (
            torch.cat(key_parts, dim=1).contiguous(),
            torch.cat(value_parts, dim=1).contiguous(),
        )

    def _selected_segments(self, current_start: int, current_end: int) -> List[_Segment]:
        current_start, current_end = int(current_start), int(current_end)
        if current_start <= 0:
            return []
        if self.max_attention_size > 0:
            local_budget = max(0, self.max_attention_size - self.sink_tokens)
            past_start = max(self.sink_tokens, current_end - local_budget)
        else:
            past_start = 0
        sink_end = min(self.sink_tokens, current_start)
        selected: List[_Segment] = []
        for segment in self.segments:
            in_sink = segment.start < sink_end and segment.end <= sink_end
            in_recent = segment.start >= max(past_start, sink_end) and segment.end <= current_start
            if in_sink or in_recent:
                if segment.adapter is None:
                    raise RuntimeError("LongLive fused attention requires packed segments.")
                selected.append(segment)
        return selected

    def attend_fused(
        self,
        query: torch.Tensor,
        current_start: int,
        current_end: int,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        *,
        backend: str,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Attend to selected history without materializing dense historical KV."""
        selected = self._selected_segments(current_start, current_end)
        if not selected:
            raise RuntimeError("LongLive fused attention requires selected packed history.")
        key_quantizers = [segment.adapter.k_quant for segment in selected]
        value_quantizers = [segment.adapter.v_quant for segment in selected]
        key_view = _PackedQuantView(key_quantizers)
        value_view = _PackedQuantView(value_quantizers)
        if key_view.cached_len != value_view.cached_len:
            raise RuntimeError("LongLive packed K/V lengths do not match.")
        from VAR_Q.fused import fused_dequant_attention

        output = fused_dequant_attention(
            query.transpose(1, 2).contiguous(),
            key_view,
            value_view,
            current_key.transpose(1, 2).contiguous(),
            current_value.transpose(1, 2).contiguous(),
            qkv_format="BHLc",
            backend=backend,
            softmax_scale=softmax_scale,
        )
        self.fused_calls += 1
        return output.transpose(1, 2).contiguous()


@dataclass
class LongLiveHookHandle:
    pipeline: nn.Module
    enabled: bool
    modules: List[nn.Module]
    original_forwards: List[Tuple[nn.Module, Any]]
    original_initialize: Optional[Any] = None
    original_generator_forward: Optional[Any] = None
    original_inference: Optional[Any] = None


def _read_config(config_path: Optional[str]) -> Dict[str, Any]:
    path = config_path or os.environ.get("LONGLIVE_VARQ_CONFIG") or os.environ.get("VARQ_CONFIG_FILE")
    if not path:
        raise EnvironmentError(
            "A LongLive VAR-Q JSON config is required; pass config_path or set "
            "LONGLIVE_VARQ_CONFIG."
        )
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    config = raw.get("quantization", raw.get("kv_quant", raw))
    if not isinstance(config, dict):
        raise TypeError("LongLive VAR-Q configuration must be a JSON object.")
    return dict(config)


def _make_attention_forward(original_forward: Any):
    function = getattr(original_forward, "__func__", original_forward)
    module_globals = function.__globals__
    attention = module_globals["attention"]
    causal_rope_apply = module_globals["causal_rope_apply"]

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        *extra_args,
        **extra_kwargs,
    ):
        state = kv_cache.get("_varq_state") if isinstance(kv_cache, dict) else None
        if state is None:
            return original_forward(
                x,
                seq_lens,
                grid_sizes,
                freqs,
                block_mask,
                kv_cache,
                current_start,
                cache_start,
                *extra_args,
                **extra_kwargs,
            )

        batch, sequence = x.shape[:2]
        heads = int(self.num_heads)
        head_dim = int(self.head_dim)
        query = self.norm_q(self.q(x)).view(batch, sequence, heads, head_dim)
        key = self.norm_k(self.k(x)).view(batch, sequence, heads, head_dim)
        value = self.v(x).view(batch, sequence, heads, head_dim)
        frame_sequence = int(math.prod(grid_sizes[0][1:]).item())
        current_start = int(current_start or 0)
        current_end = current_start + int(sequence)
        start_frame = current_start // frame_sequence
        query = causal_rope_apply(
            query, grid_sizes, freqs, start_frame=start_frame
        ).type_as(value)
        key = causal_rope_apply(
            key, grid_sizes, freqs, start_frame=start_frame
        ).type_as(value)
        fused_enabled = bool(
            state.quant_config.get("enable_fused_kv_flashattn", False)
        )
        fused_backend = str(
            state.quant_config.get("fused_kv_backend", "cuda_direct")
        )
        fused_strict = bool(
            state.quant_config.get("fused_kv_strict", True)
        )
        use_fused = bool(
            fused_enabled
            and state.segments
            and current_start > 0
            and str(state.quant_config.get("quant_method", "VARQ")).upper() == "VARQ"
        )
        if use_fused:
            try:
                output = state.attend_fused(
                    query,
                    current_start,
                    current_end,
                    key,
                    value,
                    backend=fused_backend,
                    softmax_scale=getattr(self, "scale", None),
                )
            except Exception:
                state.fused_fallbacks += 1
                if fused_strict:
                    raise
                visible_key, visible_value = state.attention_kv(
                    current_start, current_end, key, value
                )
                output = attention(query, visible_key, visible_value)
        else:
            visible_key, visible_value = state.attention_kv(
                current_start, current_end, key, value
            )
            output = attention(query, visible_key, visible_value)

        commit = bool(kv_cache.get("_varq_commit", False))
        if commit:
            state.commit(current_start, current_end, key, value)
        update_info = {"action": "none", "is_recompute": not commit}
        return self.o(output.flatten(2)), (current_end, current_end, update_info)

    return forward


def _bound_arguments(method: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return dict(inspect.signature(method).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def install_longlive_hooks(
    pipeline: nn.Module,
    config_path: Optional[str] = None,
) -> LongLiveHookHandle:
    """Install VAR-Q without modifying files in the upstream LongLive tree."""
    existing = getattr(pipeline, "_varq_longlive_handle", None)
    if isinstance(existing, LongLiveHookHandle):
        return existing

    config = _read_config(config_path)
    if not bool(config.get("enable", True)):
        handle = LongLiveHookHandle(pipeline, False, [], [])
        setattr(pipeline, "_varq_longlive_handle", handle)
        return handle

    generator = getattr(pipeline, "generator", None)
    model = getattr(generator, "model", None)
    if model is None:
        raise RuntimeError("LongLive pipeline is missing generator.model.")
    modules = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "CausalWanSelfAttention"
    ]
    if not modules:
        raise RuntimeError("No LongLive CausalWanSelfAttention modules were found.")

    originals: List[Tuple[nn.Module, Any]] = []
    for module in modules:
        original = module.forward
        originals.append((module, original))
        module.forward = types.MethodType(_make_attention_forward(original), module)

    original_initialize = pipeline._initialize_kv_cache
    frame_sequence = int(getattr(pipeline, "frame_seq_length", 1560))

    def initialize_kv_cache(self, batch_size, dtype, device, *args, **kwargs):
        caches = []
        for layer_index, module in enumerate(modules):
            local_frames = int(getattr(module, "local_attn_size", -1))
            maximum = local_frames * frame_sequence if local_frames != -1 else -1
            sink = int(getattr(module, "sink_size", 0) or 0) * frame_sequence
            caches.append(
                {
                    "k": torch.empty(
                        (batch_size, 0, int(module.num_heads), int(module.head_dim)),
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.empty(
                        (batch_size, 0, int(module.num_heads), int(module.head_dim)),
                        dtype=dtype,
                        device=device,
                    ),
                    "_varq_state": LongLiveKVCacheState(
                        config,
                        layer_idx=layer_index,
                        sink_tokens=sink,
                        max_attention_size=maximum,
                    ),
                    "_varq_commit": False,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        self.kv_cache1 = caches

    pipeline._initialize_kv_cache = types.MethodType(initialize_kv_cache, pipeline)

    original_generator_forward = generator.forward
    generator_signature = original_generator_forward
    output_frames = int(getattr(getattr(pipeline, "args", None), "num_output_frames", 0) or 0)
    total_tokens = output_frames * frame_sequence
    skip_last = bool(config.get("skip_cache_last_scale", True))
    commit_timestep = int(config.get("commit_timestep", os.environ.get("LONGLIVE_VARQ_COMMIT_TIMESTEP", 0)))

    def generator_forward(self, *args, **kwargs):
        arguments = _bound_arguments(generator_signature, args, kwargs)
        cache = arguments.get("kv_cache")
        timestep = arguments.get("timestep")
        noisy = arguments.get("noisy_image_or_video")
        current_start = int(arguments.get("current_start") or 0)
        if isinstance(cache, list) and torch.is_tensor(timestep):
            is_context = bool(torch.all(timestep == commit_timestep).item())
            current_tokens = 0
            if torch.is_tensor(noisy) and noisy.ndim >= 2:
                current_tokens = int(noisy.shape[1]) * frame_sequence
            is_not_last = (
                not skip_last
                or total_tokens <= 0
                or current_start + current_tokens < total_tokens
            )
            for layer_cache in cache:
                if isinstance(layer_cache, dict) and "_varq_state" in layer_cache:
                    layer_cache["_varq_commit"] = is_context and is_not_last
        return original_generator_forward(*args, **kwargs)

    generator.forward = types.MethodType(generator_forward, generator)

    original_inference = getattr(pipeline, "inference", None)
    if callable(original_inference):

        def inference(self, *args, **kwargs):
            for layer_cache in getattr(self, "kv_cache1", None) or []:
                state = layer_cache.get("_varq_state") if isinstance(layer_cache, dict) else None
                if state is not None:
                    state.reset()
                    layer_cache["_varq_commit"] = False
                    layer_cache["global_end_index"].zero_()
                    layer_cache["local_end_index"].zero_()
            return original_inference(*args, **kwargs)

        pipeline.inference = types.MethodType(inference, pipeline)

    handle = LongLiveHookHandle(
        pipeline=pipeline,
        enabled=True,
        modules=modules,
        original_forwards=originals,
        original_initialize=original_initialize,
        original_generator_forward=original_generator_forward,
        original_inference=original_inference,
    )
    setattr(pipeline, "_varq_longlive_handle", handle)
    return handle


def remove_longlive_hooks(handle: LongLiveHookHandle) -> None:
    """Restore methods replaced by :func:`install_longlive_hooks`."""
    pipeline = handle.pipeline
    for module, original in handle.original_forwards:
        module.forward = original
    if handle.original_initialize is not None:
        pipeline._initialize_kv_cache = handle.original_initialize
    generator = getattr(pipeline, "generator", None)
    if generator is not None and handle.original_generator_forward is not None:
        generator.forward = handle.original_generator_forward
    if handle.original_inference is not None:
        pipeline.inference = handle.original_inference
    if getattr(pipeline, "_varq_longlive_handle", None) is handle:
        delattr(pipeline, "_varq_longlive_handle")

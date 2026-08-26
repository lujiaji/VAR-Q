"""Runtime VAR-Q integration for an unmodified Self-Forcing checkout.

The public installer patches a constructed upstream pipeline in memory.  It
does not require carrying a fork of Self-Forcing inside this repository.
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
class SelfForcingHookHandle:
    pipeline: nn.Module
    enabled: bool
    modules: List[nn.Module]
    original_forwards: List[Tuple[nn.Module, Any]]
    original_initialize: Optional[Any] = None
    original_generator_forward: Optional[Any] = None
    original_inference: Optional[Any] = None


def _read_config(config_path: Optional[str]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    path = (
        config_path
        or os.environ.get("SELF_FORCING_VARQ_CONFIG")
        or os.environ.get("VARQ_CONFIG_FILE")
    )
    if not path:
        raise EnvironmentError(
            "A Self-Forcing VAR-Q JSON config is required; pass config_path or "
            "set SELF_FORCING_VARQ_CONFIG."
        )
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    quant = raw.get("quantization", raw.get("kv_quant", raw))
    if not isinstance(quant, dict):
        raise TypeError("Self-Forcing VAR-Q configuration must be a JSON object.")
    ablation = raw.get("ablation")
    if ablation is None and isinstance(quant.get("ablation"), dict):
        ablation = quant.get("ablation")
    return dict(quant), dict(ablation) if isinstance(ablation, dict) else None


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
        adapter = kv_cache.get("_varq_adapter") if isinstance(kv_cache, dict) else None
        if adapter is None:
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
        if int(getattr(self, "local_attn_size", -1)) != -1:
            raise NotImplementedError(
                "Self-Forcing VAR-Q requires local_attn_size=-1; combine sparse "
                "attention separately through the ablation API."
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

        commit = bool(kv_cache.get("_varq_commit", False))
        use_fused = adapter.has_fused_cache()
        committed_by_fallback = False
        if use_fused:
            try:
                output = adapter.attend_fused(
                    query,
                    key,
                    value,
                    softmax_scale=getattr(self, "scale", None),
                )
            except Exception:
                adapter.fused_fallbacks += 1
                if adapter.fused_strict:
                    raise
                visible_key, visible_value = adapter.update(
                    key,
                    value,
                    cache_current=commit,
                )
                committed_by_fallback = commit
                output = attention(query, visible_key, visible_value)
        else:
            visible_key, visible_value = adapter.update(
                key,
                value,
                cache_current=commit,
            )
            committed_by_fallback = commit
            output = attention(query, visible_key, visible_value)
        if commit:
            if not committed_by_fallback:
                adapter.commit(key, value)
            committed = adapter.cached_length("k")
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(committed)
        return self.o(output.flatten(2))

    return forward


def _bound_arguments(method: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return dict(inspect.signature(method).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def install_self_forcing_hooks(
    pipeline: nn.Module,
    config_path: Optional[str] = None,
) -> SelfForcingHookHandle:
    """Install compressed KV caching on a constructed Self-Forcing pipeline."""
    existing = getattr(pipeline, "_varq_self_forcing_handle", None)
    if isinstance(existing, SelfForcingHookHandle):
        return existing

    config, ablation_config = _read_config(config_path)
    if not bool(config.get("enable", True)):
        handle = SelfForcingHookHandle(pipeline, False, [], [])
        setattr(pipeline, "_varq_self_forcing_handle", handle)
        return handle

    generator = getattr(pipeline, "generator", None)
    model = getattr(generator, "model", None)
    if model is None:
        raise RuntimeError("Self-Forcing pipeline is missing generator.model.")
    modules = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "CausalWanSelfAttention"
    ]
    if not modules:
        raise RuntimeError("No Self-Forcing CausalWanSelfAttention modules were found.")
    if any(int(getattr(module, "local_attn_size", -1)) != -1 for module in modules):
        raise NotImplementedError(
            "Self-Forcing VAR-Q currently requires local_attn_size=-1."
        )

    originals: List[Tuple[nn.Module, Any]] = []
    for module in modules:
        original = module.forward
        originals.append((module, original))
        module.forward = types.MethodType(_make_attention_forward(original), module)

    original_initialize = pipeline._initialize_kv_cache

    def initialize_kv_cache(self, batch_size, dtype, device, *args, **kwargs):
        caches = []
        for layer_index, module in enumerate(modules):
            layer_config = dict(config)
            layer_config["block_idx"] = layer_index
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
                    "_varq_adapter": VideoKVCacheAdapter.from_config(
                        layer_config,
                        ablation_config=ablation_config,
                        kv_role_prefix=f"self_forcing.block{layer_index}",
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
    frame_sequence = int(getattr(pipeline, "frame_seq_length", 1560))
    output_frames = int(getattr(getattr(pipeline, "args", None), "num_output_frames", 0) or 0)
    total_tokens = output_frames * frame_sequence
    skip_last = bool(config.get("skip_cache_last_scale", True))
    commit_timestep = int(
        config.get(
            "commit_timestep",
            os.environ.get("SELF_FORCING_VARQ_COMMIT_TIMESTEP", 0),
        )
    )

    def generator_forward(self, *args, **kwargs):
        arguments = _bound_arguments(generator_signature, args, kwargs)
        cache = arguments.get("kv_cache")
        timestep = arguments.get("timestep")
        noisy = arguments.get("noisy_image_or_video")
        current_start = int(arguments.get("current_start") or 0)
        if isinstance(cache, list) and torch.is_tensor(timestep):
            commit = bool(torch.all(timestep == commit_timestep).item())
            current_tokens = 0
            if torch.is_tensor(noisy) and noisy.ndim >= 2:
                current_tokens = int(noisy.shape[1]) * frame_sequence
            if skip_last and total_tokens > 0 and current_start + current_tokens >= total_tokens:
                commit = False
            for layer_cache in cache:
                if isinstance(layer_cache, dict) and "_varq_adapter" in layer_cache:
                    layer_cache["_varq_commit"] = commit
        return original_generator_forward(*args, **kwargs)

    generator.forward = types.MethodType(generator_forward, generator)

    original_inference = getattr(pipeline, "inference", None)
    if callable(original_inference):

        def inference(self, *args, **kwargs):
            for layer_cache in getattr(self, "kv_cache1", None) or []:
                adapter = layer_cache.get("_varq_adapter") if isinstance(layer_cache, dict) else None
                if adapter is not None:
                    adapter.reset()
                    layer_cache["_varq_commit"] = False
                    layer_cache["global_end_index"].zero_()
                    layer_cache["local_end_index"].zero_()
            return original_inference(*args, **kwargs)

        pipeline.inference = types.MethodType(inference, pipeline)

    handle = SelfForcingHookHandle(
        pipeline=pipeline,
        enabled=True,
        modules=modules,
        original_forwards=originals,
        original_initialize=original_initialize,
        original_generator_forward=original_generator_forward,
        original_inference=original_inference,
    )
    setattr(pipeline, "_varq_self_forcing_handle", handle)
    return handle


def remove_self_forcing_hooks(handle: SelfForcingHookHandle) -> None:
    """Restore methods replaced by :func:`install_self_forcing_hooks`."""
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
    if getattr(pipeline, "_varq_self_forcing_handle", None) is handle:
        delattr(pipeline, "_varq_self_forcing_handle")

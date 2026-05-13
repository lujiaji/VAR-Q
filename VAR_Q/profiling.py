from __future__ import annotations

from typing import Any, Dict

import torch


MEMORY_KEYS = (
    "packed_kv_bytes",
    "scale_bytes",
    "dequant_workspace_bytes",
    "dequant_workspace_peak_bytes",
    "packed_cache_allocated_bytes",
    "scale_cache_allocated_bytes",
)


def reset_cuda_memory_stats() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def cuda_memory_stats() -> Dict[str, int]:
    if not torch.cuda.is_available():
        return {
            "cuda_memory_allocated": 0,
            "cuda_max_memory_allocated": 0,
            "cuda_memory_reserved": 0,
            "cuda_max_memory_reserved": 0,
        }
    torch.cuda.synchronize()
    return {
        "cuda_memory_allocated": int(torch.cuda.memory_allocated()),
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "cuda_memory_reserved": int(torch.cuda.memory_reserved()),
        "cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved()),
    }


def _add_quantizer_stats(total: Dict[str, int], quantizer: Any) -> None:
    if quantizer is None or not hasattr(quantizer, "memory_breakdown"):
        return
    stats = quantizer.memory_breakdown()
    for key in MEMORY_KEYS:
        total[key] += int(stats.get(key, 0))


def collect_varq_memory_breakdown(model: torch.nn.Module) -> Dict[str, int]:
    total = {key: 0 for key in MEMORY_KEYS}
    handle = getattr(model, "_varq_hook_handle", None)
    modules = getattr(handle, "modules", None)
    if modules is None:
        modules = [module for module in model.modules() if getattr(module, "_varq_runtime_hooked", False)]

    for module in modules:
        snapshot = getattr(module, "_varq_last_memory_breakdown", None)
        if snapshot:
            for key in MEMORY_KEYS:
                total[key] += int(snapshot.get(key, 0))
        _add_quantizer_stats(total, getattr(module, "k_quant", None))
        _add_quantizer_stats(total, getattr(module, "v_quant", None))
        for attr in ("k_varq", "v_varq"):
            quantizer = getattr(module, attr, None)
            if quantizer is None or not hasattr(quantizer, "cache_bytes"):
                continue
            stats = quantizer.cache_bytes()
            total["packed_kv_bytes"] += int(stats.get("packed_bytes", 0))
            total["scale_bytes"] += int(stats.get("scale_bytes", 0))
            total["dequant_workspace_bytes"] += int(stats.get("dequant_workspace_bytes", 0))
            total["dequant_workspace_peak_bytes"] += int(stats.get("dequant_workspace_peak_bytes", 0))

    total.update(cuda_memory_stats())
    return total


def format_memory_breakdown(stats: Dict[str, int]) -> str:
    return (
        f"packed_kv_bytes={stats.get('packed_kv_bytes', 0)} "
        f"scale_bytes={stats.get('scale_bytes', 0)} "
        f"dequant_workspace_bytes={stats.get('dequant_workspace_bytes', 0)} "
        f"dequant_workspace_peak_bytes={stats.get('dequant_workspace_peak_bytes', 0)} "
        f"cuda_memory_allocated={stats.get('cuda_memory_allocated', 0)} "
        f"cuda_max_memory_allocated={stats.get('cuda_max_memory_allocated', 0)} "
        f"cuda_memory_reserved={stats.get('cuda_memory_reserved', 0)} "
        f"cuda_max_memory_reserved={stats.get('cuda_max_memory_reserved', 0)}"
    )

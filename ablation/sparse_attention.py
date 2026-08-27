"""Optional sparse-attention algorithms for VAR-Q ablation studies.

The module exposes executable FlexAttention policies only.  It contains no
benchmark harness, tracing counters, or model-specific evaluation code.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


_COMPILED_FLEX_ATTENTION = None


def _run_flex_attention(*args, **kwargs):
    from torch.nn.attention.flex_attention import flex_attention

    global _COMPILED_FLEX_ATTENTION
    compile_enabled = (
        torch.cuda.is_available()
        and os.environ.get("VARQ_FLEXATTN_COMPILE", "1") != "0"
    )
    if compile_enabled:
        if _COMPILED_FLEX_ATTENTION is None:
            _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)
        return _COMPILED_FLEX_ATTENTION(*args, **kwargs)
    return flex_attention(*args, **kwargs)


def _to_bhld(tensor: torch.Tensor, qkv_format: str) -> torch.Tensor:
    if qkv_format == "BHLc":
        return tensor
    if qkv_format == "BLHc":
        return tensor.transpose(1, 2)
    raise ValueError(f"Unsupported qkv_format={qkv_format!r}")


def _from_bhld(tensor: torch.Tensor, qkv_format: str) -> torch.Tensor:
    return tensor if qkv_format == "BHLc" else tensor.transpose(1, 2)


def _validate_scale_lengths(scale_lengths: Iterable[int], key_len: int) -> list[int]:
    lengths = [int(length) for length in scale_lengths]
    if any(length <= 0 for length in lengths) or sum(lengths) != key_len:
        raise ValueError(
            f"scale_lengths={lengths} do not partition key_len={key_len}"
        )
    return lengths


def streaming_keep_mask(
    query_len: int,
    key_len: int,
    *,
    scale_lengths: Sequence[int],
    local_window: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the exact token mask for per-scale sinks plus a local window."""
    if query_len < 1 or key_len < 1 or local_window < 1:
        raise ValueError("query_len, key_len, and local_window must be positive")
    lengths = _validate_scale_lengths(scale_lengths, key_len)
    sinks = torch.zeros(key_len, dtype=torch.bool, device=device)
    offset = 0
    for length in lengths:
        sinks[offset] = True
        offset += length
    query_position = torch.arange(key_len - query_len, key_len, device=device)[:, None]
    key_position = torch.arange(key_len, device=device)[None, :]
    left = local_window // 2
    right = local_window - left
    local = (key_position >= query_position - left) & (
        key_position < query_position + right
    )
    return local | sinks[None, :]


def _block_means(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    batch, heads, length, dim = tensor.shape
    blocks = math.ceil(length / block_size)
    padded = blocks * block_size
    if padded != length:
        tensor = F.pad(tensor, (0, 0, 0, padded - length))
    values = tensor.reshape(batch, heads, blocks, block_size, dim)
    counts = torch.full(
        (blocks,), block_size, dtype=tensor.dtype, device=tensor.device
    )
    counts[-1] = length - (blocks - 1) * block_size
    return values.sum(dim=3) / counts[None, None, :, None]


def row_topk_block_selection(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    fraction: float,
    block_size: int,
) -> torch.Tensor:
    """Return row-wise content top-k blocks as ``[B,H,Qb,Kb]``."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    query_blocks = _block_means(query.float(), block_size)
    key_blocks = _block_means(key.float(), block_size)
    scores = torch.matmul(query_blocks, key_blocks.transpose(-1, -2))
    keep = max(1, math.ceil(scores.shape[-1] * fraction))
    indices = scores.topk(keep, dim=-1, sorted=False).indices
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected.scatter_(-1, indices, True)
    return selected


def _selected_block_mask(
    selected: torch.Tensor,
    *,
    query_len: int,
    key_len: int,
    block_size: int,
    token_mask_mod=None,
):
    from torch.nn.attention.flex_attention import BlockMask

    if selected.ndim != 4:
        raise ValueError(f"selected must be rank-4, got {tuple(selected.shape)}")
    _, _, query_blocks, key_blocks = selected.shape
    if query_blocks != math.ceil(query_len / block_size):
        raise ValueError("selected query-block dimension does not match query_len")
    if key_blocks != math.ceil(key_len / block_size):
        raise ValueError("selected KV-block dimension does not match key_len")
    block_counts = selected.sum(dim=-1, dtype=torch.int32)
    if int(block_counts.min().item()) <= 0:
        raise ValueError("a sparse attention row selected no KV blocks")
    block_ids = torch.arange(
        key_blocks, device=selected.device, dtype=torch.int32
    ).view(1, 1, 1, key_blocks)
    block_ids = block_ids.expand_as(selected)
    compact = torch.where(
        selected, block_ids, torch.full_like(block_ids, key_blocks)
    ).sort(dim=-1).values.contiguous()

    def mask_mod(batch_index, head_index, query_index, key_index):
        valid = (query_index < query_len) & (key_index < key_len)
        query_block = torch.clamp(
            query_index // block_size, max=query_blocks - 1
        )
        key_block = torch.clamp(key_index // block_size, max=key_blocks - 1)
        keep = valid & selected[
            batch_index, head_index, query_block, key_block
        ]
        if token_mask_mod is not None:
            keep = keep & token_mask_mod(
                batch_index, head_index, query_index, key_index
            )
        return keep

    return BlockMask.from_kv_blocks(
        block_counts,
        compact,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(query_len, key_len),
    )


def _sliding_selection(
    *,
    query_len: int,
    key_len: int,
    local_window: int,
    block_size: int,
    batch: int,
    heads: int,
    device: torch.device,
) -> torch.Tensor:
    if local_window < 1:
        raise ValueError("local_window must be positive")
    query_blocks = math.ceil(query_len / block_size)
    key_blocks = math.ceil(key_len / block_size)
    selected = torch.zeros(
        (batch, heads, query_blocks, key_blocks),
        dtype=torch.bool,
        device=device,
    )
    left = local_window // 2
    right = local_window - left
    query_offset = key_len - query_len
    for query_block in range(query_blocks):
        query_start = query_offset + query_block * block_size
        query_end = min(query_offset + query_len, query_start + block_size) - 1
        local_start = max(0, query_start - left)
        local_end = min(key_len - 1, query_end + right - 1)
        selected[
            :, :, query_block, local_start // block_size : local_end // block_size + 1
        ] = True
    return selected


def _streaming_selection(
    *,
    query_len: int,
    key_len: int,
    scale_lengths: Sequence[int],
    local_window: int,
    block_size: int,
    batch: int,
    heads: int,
    device: torch.device,
) -> torch.Tensor:
    lengths = _validate_scale_lengths(scale_lengths, key_len)
    selected = _sliding_selection(
        query_len=query_len,
        key_len=key_len,
        local_window=local_window,
        block_size=block_size,
        batch=batch,
        heads=heads,
        device=device,
    )
    sink_blocks = []
    offset = 0
    for length in lengths:
        sink_blocks.append(offset // block_size)
        offset += length
    selected[:, :, :, sink_blocks] = True
    return selected


def _current_scale_selection(
    *,
    query_len: int,
    key_len: int,
    cached_window: int,
    global_sink: bool,
    block_size: int,
    batch: int,
    heads: int,
    device: torch.device,
) -> torch.Tensor:
    if cached_window < 1:
        raise ValueError("cached_window must be positive")
    if query_len < 1 or query_len > key_len:
        raise ValueError("query_len must be in [1, key_len]")
    query_blocks = math.ceil(query_len / block_size)
    key_blocks = math.ceil(key_len / block_size)
    selected = torch.zeros(
        (batch, heads, query_blocks, key_blocks),
        dtype=torch.bool,
        device=device,
    )
    cached_len = key_len - query_len
    cached_start = max(0, cached_len - cached_window)
    selected[:, :, :, cached_start // block_size :] = True
    if global_sink and cached_len > 0:
        selected[:, :, :, 0] = True
    return selected


def _evenly_spaced_indices(first: int, last: int, count: int) -> list[int]:
    available = last - first + 1
    count = min(max(1, int(count)), available)
    if count == 1:
        return [first]
    return sorted(
        {
            first
            + (index * (available - 1) + (count - 1) // 2) // (count - 1)
            for index in range(count)
        }
    )


def regular_block_selection(
    *,
    query_len: int,
    key_len: int,
    scale_lengths: Sequence[int],
    fraction: float,
    local_window: int,
    block_size: int,
    per_scale: bool,
    batch: int,
    heads: int,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic global blocks plus a local window."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    lengths = _validate_scale_lengths(scale_lengths, key_len)
    selected = _sliding_selection(
        query_len=query_len,
        key_len=key_len,
        local_window=local_window,
        block_size=block_size,
        batch=batch,
        heads=heads,
        device=device,
    )
    key_blocks = selected.shape[-1]
    global_blocks: set[int] = set()
    if per_scale:
        offset = 0
        for length in lengths:
            first = offset // block_size
            last = (offset + length - 1) // block_size
            keep = max(1, math.ceil((last - first + 1) * fraction))
            global_blocks.update(_evenly_spaced_indices(first, last, keep))
            offset += length
    else:
        keep = max(1, math.ceil(key_blocks * fraction))
        global_blocks.update(_evenly_spaced_indices(0, key_blocks - 1, keep))
    selected[:, :, :, sorted(global_blocks)] = True
    return selected


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    qkv_format: str,
    mode: str,
    scale: float | None = None,
    scale_lengths: Iterable[int] = (),
    local_window: int = 128,
    topk_fraction: float = 0.25,
    block_size: int = 128,
) -> torch.Tensor:
    """Run one supported sparse-attention policy with PyTorch FlexAttention."""
    query_bhld = _to_bhld(query, qkv_format).contiguous()
    key_bhld = _to_bhld(key, qkv_format).contiguous()
    value_bhld = _to_bhld(value, qkv_format).contiguous()
    if query_bhld.ndim != 4 or key_bhld.ndim != 4 or value_bhld.ndim != 4:
        raise ValueError("query, key, and value must all be rank-4 tensors")
    if key_bhld.shape != value_bhld.shape:
        raise ValueError("key and value must have identical shapes")
    if (
        query_bhld.shape[:2] != key_bhld.shape[:2]
        or query_bhld.shape[-1] != key_bhld.shape[-1]
    ):
        raise ValueError("query/key batch, head, and head dimensions must match")
    if query_bhld.shape[-2] > key_bhld.shape[-2]:
        raise ValueError("query length cannot exceed KV length")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    batch, heads, query_len, head_dim = query_bhld.shape
    key_len = key_bhld.shape[-2]
    value_dim = value_bhld.shape[-1]
    query_flex = query_bhld.reshape(batch * heads, 1, query_len, head_dim)
    key_flex = key_bhld.reshape(batch * heads, 1, key_len, head_dim)
    value_flex = value_bhld.reshape(batch * heads, 1, key_len, value_dim)
    normalized = mode.lower().replace("-", "_")

    if normalized in {"global_sink_streaming", "streaming_global_sink"}:
        selected = _sliding_selection(
            query_len=query_len,
            key_len=key_len,
            local_window=local_window,
            block_size=block_size,
            batch=batch * heads,
            heads=1,
            device=query.device,
        )
        selected[:, :, :, 0] = True
        offset = key_len - query_len
        left = local_window // 2
        right = local_window - left

        def token_mask(_batch, _head, query_index, key_index):
            global_query = offset + query_index
            local = (key_index >= global_query - left) & (
                key_index < global_query + right
            )
            return local | (key_index == 0)

    elif normalized == "streaming":
        lengths = _validate_scale_lengths(scale_lengths, key_len)
        selected = _streaming_selection(
            query_len=query_len,
            key_len=key_len,
            scale_lengths=lengths,
            local_window=local_window,
            block_size=block_size,
            batch=batch * heads,
            heads=1,
            device=query.device,
        )
        sinks = torch.zeros(key_len, dtype=torch.bool, device=query.device)
        sink_offset = 0
        for length in lengths:
            sinks[sink_offset] = True
            sink_offset += length
        offset = key_len - query_len
        left = local_window // 2
        right = local_window - left

        def token_mask(_batch, _head, query_index, key_index):
            global_query = offset + query_index
            local = (key_index >= global_query - left) & (
                key_index < global_query + right
            )
            return local | sinks[torch.clamp(key_index, max=key_len - 1)]

    elif normalized in {"current_scale_sliding", "current_full_sliding"}:
        cached_len = key_len - query_len
        cached_start = max(0, cached_len - local_window)
        selected = _current_scale_selection(
            query_len=query_len,
            key_len=key_len,
            cached_window=local_window,
            global_sink=False,
            block_size=block_size,
            batch=batch * heads,
            heads=1,
            device=query.device,
        )

        def token_mask(_batch, _head, _query_index, key_index):
            cached = (key_index >= cached_start) & (key_index < cached_len)
            current = (key_index >= cached_len) & (key_index < key_len)
            return cached | current

    elif normalized in {"current_scale_streaming", "current_full_streaming"}:
        cached_len = key_len - query_len
        cached_start = max(0, cached_len - local_window)
        selected = _current_scale_selection(
            query_len=query_len,
            key_len=key_len,
            cached_window=local_window,
            global_sink=True,
            block_size=block_size,
            batch=batch * heads,
            heads=1,
            device=query.device,
        )

        def token_mask(_batch, _head, _query_index, key_index):
            cached = (key_index >= cached_start) & (key_index < cached_len)
            current = (key_index >= cached_len) & (key_index < key_len)
            sink = (key_index == 0) & (cached_len > 0)
            return cached | current | sink

    elif normalized in {"sliding", "sliding_window", "local", "local_window"}:
        selected = _sliding_selection(
            query_len=query_len,
            key_len=key_len,
            local_window=local_window,
            block_size=block_size,
            batch=batch * heads,
            heads=1,
            device=query.device,
        )
        offset = key_len - query_len
        left = local_window // 2
        right = local_window - left

        def token_mask(_batch, _head, query_index, key_index):
            global_query = offset + query_index
            return (key_index >= global_query - left) & (
                key_index < global_query + right
            )

    elif normalized in {"regular_block", "block_stride", "strided_block"}:
        selected = regular_block_selection(
            query_len=query_len,
            key_len=key_len,
            scale_lengths=list(scale_lengths),
            fraction=topk_fraction,
            local_window=local_window,
            block_size=block_size,
            per_scale=False,
            batch=batch * heads,
            heads=1,
            device=query.device,
        )
        token_mask = None
    elif normalized in {
        "scale_blocks",
        "regular_scale",
        "scale_stratified_blocks",
    }:
        selected = regular_block_selection(
            query_len=query_len,
            key_len=key_len,
            scale_lengths=list(scale_lengths),
            fraction=topk_fraction,
            local_window=local_window,
            block_size=block_size,
            per_scale=True,
            batch=batch * heads,
            heads=1,
            device=query.device,
        )
        token_mask = None
    elif normalized in {"row_topk", "row_topk_25", "block_topk"}:
        selected = row_topk_block_selection(
            query_bhld,
            key_bhld,
            fraction=topk_fraction,
            block_size=block_size,
        ).reshape(
            batch * heads,
            1,
            math.ceil(query_len / block_size),
            math.ceil(key_len / block_size),
        )
        token_mask = None
    else:
        raise ValueError(f"Unsupported sparse attention mode={mode!r}")

    block_mask = _selected_block_mask(
        selected,
        query_len=query_len,
        key_len=key_len,
        block_size=block_size,
        token_mask_mod=token_mask,
    )
    output = _run_flex_attention(
        query_flex,
        key_flex,
        value_flex,
        block_mask=block_mask,
        scale=scale,
    )
    output = output.reshape(batch, heads, query_len, value_dim)
    return _from_bhld(output, qkv_format)

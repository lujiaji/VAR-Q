from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from .pack_unpack import (
    CUDA_PACK_BITS,
    pack_last_dim_to_int32_cuda,
    pack_last_dim_to_int32_python,
    unpack_last_dim_from_int32_cuda,
    unpack_last_dim_from_int32_python,
)
from .legacy_quant import (
    DEFAULT_KIVI_CALI_K_GROUP_SIZE,
    DEFAULT_KIVI_CALI_V_GROUP_SIZE,
    DEFAULT_KIVI_GROUP_SIZE,
    SUPPORTED_QKV_FORMATS,
    _clone_quant_meta,
    _dequant_dtype_name,
    _expand_kivi_scale_for_shape,
    _from_bhld_layout,
    _group_lengths_for_axis,
    _merge_kivi_quant_meta,
    _normalize_kv_role,
    _to_bhld_layout,
    dequantize_tensor as legacy_dequantize_tensor,
    resolve_dequant_dtype,
)

ABLATION_METHODS = ("ABL_KIVI", "ABL_KIVI_CALI", "ABL_KV_FLexGen", "ABL_KVQUANT")
ABLATION_METHOD_ALIASES = {
    "KIVI": "ABL_KIVI",
    "KIVI-CALI": "ABL_KIVI_CALI",
    "FLEXGEN": "ABL_KV_FLexGen",
    "FLEX-GEN": "ABL_KV_FLexGen",
    "KVQUANT": "ABL_KVQUANT",
}
SUPPORTED_ABLATION_Q_BITS = (2, 3, 4, 6, 8)
DEFAULT_ABLATION_CACHE_DIR = "ablation_cache"


def normalize_ablation_method(quant_method: str) -> str:
    raw = str(quant_method)
    key = raw.replace("_", "-").upper()
    return ABLATION_METHOD_ALIASES.get(key, raw)


def is_ablation_method(quant_method: str) -> bool:
    return normalize_ablation_method(quant_method) in ABLATION_METHODS


def _cfg_get(raw_cfg: Any, key: str, default: Any) -> Any:
    if raw_cfg is None:
        return default
    if isinstance(raw_cfg, dict):
        return raw_cfg.get(key, default)
    return getattr(raw_cfg, key, default)


@dataclass(frozen=True)
class AblationConfig:
    method: str = "ABL_KIVI"
    q_bits: int = 4
    sym: bool = True
    group_size_k: int = 128
    group_size_v: int = 128
    block_size: int = 128
    percdamp: float = 0.01
    act_order: bool = False
    static_groups: bool = False
    runtime_form: str = "fake"
    cache_dir: str = DEFAULT_ABLATION_CACHE_DIR
    calibration_cache: Optional[str] = None
    use_nuq: bool = False
    use_pre_rope_k: bool = False
    outlier_ratio: float = 0.01
    outlier_cap_per_token: int = 0
    sink_fp16_tokens: int = 0
    nuq_codebook_path: Optional[str] = None

    def validate(self) -> None:
        if self.method not in ABLATION_METHODS:
            raise ValueError(f"Unsupported ablation method: {self.method}")
        if self.q_bits not in SUPPORTED_ABLATION_Q_BITS:
            raise ValueError(
                f"Ablation quantization only supports q_bits in {SUPPORTED_ABLATION_Q_BITS}, got {self.q_bits}"
            )
        if not self.sym:
            raise ValueError("Ablation quantization only supports symmetric quantization.")
        if int(self.group_size_k) <= 0 or int(self.group_size_v) <= 0:
            raise ValueError(
                f"group_size_k/group_size_v must be positive, got {self.group_size_k}/{self.group_size_v}"
            )
        if int(self.block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if float(self.percdamp) <= 0:
            raise ValueError(f"percdamp must be positive, got {self.percdamp}")
        if self.runtime_form != "fake":
            raise ValueError(f"Only runtime_form='fake' is supported, got {self.runtime_form}")
        if float(self.outlier_ratio) < 0:
            raise ValueError(f"outlier_ratio must be >= 0, got {self.outlier_ratio}")
        if int(self.outlier_cap_per_token) < 0:
            raise ValueError(
                f"outlier_cap_per_token must be >= 0, got {self.outlier_cap_per_token}"
            )
        if int(self.sink_fp16_tokens) < 0:
            raise ValueError(f"sink_fp16_tokens must be >= 0, got {self.sink_fp16_tokens}")


def build_ablation_config(raw_cfg: Any, quant_method: Optional[str] = None) -> Optional[AblationConfig]:
    cfg = None
    if raw_cfg:
        cfg = AblationConfig(
            method=normalize_ablation_method(str(_cfg_get(raw_cfg, "method", quant_method or "ABL_KIVI"))),
            q_bits=int(_cfg_get(raw_cfg, "q_bits", 4)),
            sym=bool(_cfg_get(raw_cfg, "sym", True)),
            group_size_k=int(_cfg_get(raw_cfg, "group_size_k", 128)),
            group_size_v=int(_cfg_get(raw_cfg, "group_size_v", 128)),
            block_size=int(_cfg_get(raw_cfg, "block_size", 128)),
            percdamp=float(_cfg_get(raw_cfg, "percdamp", 0.01)),
            act_order=bool(_cfg_get(raw_cfg, "act_order", False)),
            static_groups=bool(_cfg_get(raw_cfg, "static_groups", False)),
            runtime_form=str(_cfg_get(raw_cfg, "runtime_form", "fake")),
            cache_dir=str(_cfg_get(raw_cfg, "cache_dir", DEFAULT_ABLATION_CACHE_DIR)),
            calibration_cache=_cfg_get(raw_cfg, "calibration_cache", None),
            use_nuq=bool(_cfg_get(raw_cfg, "use_nuq", False)),
            use_pre_rope_k=bool(_cfg_get(raw_cfg, "use_pre_rope_k", False)),
            outlier_ratio=float(_cfg_get(raw_cfg, "outlier_ratio", 0.01)),
            outlier_cap_per_token=int(_cfg_get(raw_cfg, "outlier_cap_per_token", 0)),
            sink_fp16_tokens=int(_cfg_get(raw_cfg, "sink_fp16_tokens", 0)),
            nuq_codebook_path=_cfg_get(raw_cfg, "nuq_codebook_path", None),
        )
        cfg.validate()
        return cfg
    if quant_method and is_ablation_method(quant_method):
        cfg = AblationConfig(method=normalize_ablation_method(str(quant_method)))
        cfg.validate()
        return cfg
    return None


def _is_flexgen_meta(quant_meta: Optional[Dict[str, Any]]) -> bool:
    return bool(quant_meta) and str(quant_meta.get("scheme", "")) == "ABL_KV_FLexGen"


def _is_kvquant_meta(quant_meta: Optional[Dict[str, Any]]) -> bool:
    return bool(quant_meta) and str(quant_meta.get("scheme", "")) == "ABL_KVQUANT"


def _qkv_seq_dim(qkv_format: str) -> int:
    return 1 if str(qkv_format) == "BLHc" else 2


def _append_sparse_meta(
    base_meta: Dict[str, Any],
    next_meta: Dict[str, Any],
    seq_offset: int,
) -> None:
    base_coords = base_meta.get("sparse_coords")
    next_coords = next_meta.get("sparse_coords")
    base_vals = base_meta.get("sparse_values")
    next_vals = next_meta.get("sparse_values")
    if next_coords is None or next_vals is None:
        return
    seq_dim = _qkv_seq_dim(str(base_meta.get("qkv_format", "BLHc")))
    next_coords = next_coords.clone()
    if next_coords.numel() > 0:
        next_coords[:, seq_dim] += int(seq_offset)
    if base_coords is None or base_vals is None:
        base_meta["sparse_coords"] = next_coords
        base_meta["sparse_values"] = next_vals.clone()
        return
    if base_coords.numel() == 0:
        base_meta["sparse_coords"] = next_coords
        base_meta["sparse_values"] = next_vals.clone()
        return
    if next_coords.numel() == 0:
        return
    base_meta["sparse_coords"] = torch.cat([base_coords, next_coords], dim=0)
    base_meta["sparse_values"] = torch.cat([base_vals, next_vals], dim=0)


def _merge_ablation_quant_meta(
    base_meta: Optional[Dict[str, Any]],
    next_meta: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if next_meta is None:
        return _clone_quant_meta(base_meta)
    if base_meta is None:
        return _clone_quant_meta(next_meta)
    if _is_flexgen_meta(base_meta) and _is_flexgen_meta(next_meta):
        base = _clone_quant_meta(base_meta)
        nxt = _clone_quant_meta(next_meta)
        if str(base.get("group_axis")) != str(nxt.get("group_axis")):
            raise ValueError("Cannot merge FLexGen quant meta with mismatched group_axis")
        if _normalize_kv_role(base.get("kv_role", "k")) != _normalize_kv_role(nxt.get("kv_role", "k")):
            raise ValueError("Cannot merge FLexGen quant meta with mismatched kv_role")
        if base.get("qkv_format") != nxt.get("qkv_format"):
            raise ValueError("Cannot merge FLexGen quant meta with mismatched qkv_format")
        if base.get("group_axis") == "BLOCK2D":
            base["token_group_lengths"] = list(base.get("token_group_lengths", [])) + list(
                nxt.get("token_group_lengths", [])
            )
            if list(base.get("dim_group_lengths", [])) != list(nxt.get("dim_group_lengths", [])):
                raise ValueError("Cannot merge FLexGen block quant meta with mismatched dim group lengths")
        elif base.get("group_axis") == "TOKEN":
            base["group_lengths"] = list(base.get("group_lengths", [])) + list(nxt.get("group_lengths", []))
        else:
            if list(base.get("group_lengths", [])) != list(nxt.get("group_lengths", [])):
                raise ValueError("Cannot merge FLexGen-D quant meta with mismatched group lengths")
        base_shape = list(base.get("orig_shape", ()))
        next_shape = list(nxt.get("orig_shape", ()))
        if base_shape and next_shape and len(base_shape) == len(next_shape):
            seq_dim = _qkv_seq_dim(str(base.get("qkv_format")))
            base_shape[seq_dim] = int(base_shape[seq_dim]) + int(next_shape[seq_dim])
            base["orig_shape"] = tuple(base_shape)
        return base
    if _is_kvquant_meta(base_meta) and _is_kvquant_meta(next_meta):
        base = _clone_quant_meta(base_meta)
        nxt = _clone_quant_meta(next_meta)
        if _normalize_kv_role(base.get("kv_role", "k")) != _normalize_kv_role(nxt.get("kv_role", "k")):
            raise ValueError("Cannot merge KVQuant quant meta with mismatched kv_role")
        if base.get("qkv_format") != nxt.get("qkv_format"):
            raise ValueError("Cannot merge KVQuant quant meta with mismatched qkv_format")
        if str(base.get("group_axis")) != str(nxt.get("group_axis")):
            raise ValueError("Cannot merge KVQuant quant meta with mismatched group_axis")
        if str(base.get("group_axis")) == "TOKEN":
            base["group_lengths"] = list(base.get("group_lengths", [])) + list(nxt.get("group_lengths", []))
        else:
            if list(base.get("group_lengths", [])) != list(nxt.get("group_lengths", [])):
                raise ValueError("Cannot merge KVQuant quant meta with mismatched group lengths")
        base_shape = list(base.get("orig_shape", ()))
        next_shape = list(nxt.get("orig_shape", ()))
        if base_shape and next_shape and len(base_shape) == len(next_shape):
            seq_dim = _qkv_seq_dim(str(base.get("qkv_format")))
            seq_offset = int(base_shape[seq_dim])
            base_shape[seq_dim] = seq_offset + int(next_shape[seq_dim])
            base["orig_shape"] = tuple(base_shape)
            _append_sparse_meta(base, nxt, seq_offset)
        return base
    return _merge_kivi_quant_meta(base_meta, next_meta)


def _expand_ablation_scale_for_shape(
    scale: torch.Tensor,
    quant_meta: Optional[Dict[str, Any]],
    target_shape: Sequence[int],
) -> torch.Tensor:
    if not _is_flexgen_meta(quant_meta):
        if not _is_kvquant_meta(quant_meta):
            return scale

    qkv_format = str(quant_meta.get("qkv_format", "BLHc"))
    group_axis = str(quant_meta.get("group_axis", ""))
    if group_axis == "BLOCK2D":
        token_lengths = [int(v) for v in quant_meta.get("token_group_lengths", ())]
        dim_lengths = [int(v) for v in quant_meta.get("dim_group_lengths", ())]
        if not token_lengths or not dim_lengths:
            return scale
        token_counts = torch.tensor(token_lengths, device=scale.device, dtype=torch.long)
        dim_counts = torch.tensor(dim_lengths, device=scale.device, dtype=torch.long)
        if qkv_format == "BLHc":
            bhgd = scale.permute(0, 2, 1, 3).contiguous()
        else:
            bhgd = scale
        expanded = bhgd.repeat_interleave(token_counts, dim=2).repeat_interleave(dim_counts, dim=3)
        target_len = int(target_shape[1] if qkv_format == "BLHc" else target_shape[2])
        target_dim = int(target_shape[-1])
        if expanded.size(2) > target_len:
            expanded = expanded[:, :, :target_len, :]
        if expanded.size(3) > target_dim:
            expanded = expanded[:, :, :, :target_dim]
        if qkv_format == "BLHc":
            return expanded.permute(0, 2, 1, 3).contiguous()
        return expanded
    group_lengths = [int(v) for v in quant_meta.get("group_lengths", ())]
    if not group_lengths:
        return scale
    counts = torch.tensor(group_lengths, device=scale.device, dtype=torch.long)
    if group_axis == "D":
        bhlg = scale if qkv_format == "BHLc" else scale.permute(0, 2, 1, 3).contiguous()
        expanded = bhlg.repeat_interleave(counts, dim=3)
        target_dim = int(target_shape[-1])
        if expanded.size(3) > target_dim:
            expanded = expanded[:, :, :, :target_dim]
        if qkv_format == "BLHc":
            return expanded.permute(0, 2, 1, 3).contiguous()
        return expanded
    if group_axis == "TOKEN":
        if qkv_format == "BLHc" and scale.dim() == 4 and scale.size(1) == len(group_lengths):
            bhgd = scale.permute(0, 2, 1, 3).contiguous()
        else:
            bhgd = scale
        expanded = bhgd.repeat_interleave(counts, dim=2)
        target_len = int(target_shape[1] if qkv_format == "BLHc" else target_shape[2])
        if expanded.size(2) > target_len:
            expanded = expanded[:, :, :target_len, :]
        if qkv_format == "BLHc":
            return expanded.permute(0, 2, 1, 3).contiguous()
        return expanded
    raise ValueError(f"Unsupported FLexGen group_axis={group_axis}")


def _apply_sparse_exact_values(tensor: torch.Tensor, quant_meta: Optional[Dict[str, Any]]) -> torch.Tensor:
    if not _is_kvquant_meta(quant_meta):
        return tensor
    sparse_coords = quant_meta.get("sparse_coords")
    sparse_values = quant_meta.get("sparse_values")
    if sparse_coords is None or sparse_values is None or sparse_coords.numel() == 0:
        return tensor
    coords = sparse_coords.to(device=tensor.device, dtype=torch.long)
    values = sparse_values.to(device=tensor.device, dtype=tensor.dtype)
    out = tensor.clone()
    out[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = values
    return out


def _dequantize_block2d_int8(
    q_int8: torch.Tensor,
    scale: torch.Tensor,
    quant_meta: Dict[str, Any],
    target_dtype: torch.dtype,
) -> torch.Tensor:
    qkv_format = str(quant_meta.get("qkv_format", "BLHc"))
    token_lengths = [int(v) for v in quant_meta.get("token_group_lengths", ())]
    dim_lengths = [int(v) for v in quant_meta.get("dim_group_lengths", ())]
    if not token_lengths or not dim_lengths:
        expanded = _expand_ablation_scale_for_shape(scale, quant_meta, q_int8.shape)
        return (q_int8.to(torch.float32) * expanded).to(target_dtype)

    token_group_size = int(quant_meta.get("block_size", token_lengths[0]))
    dim_group_size = int(quant_meta.get("dim_group_size", dim_lengths[0]))
    if token_group_size <= 0 or dim_group_size <= 0:
        expanded = _expand_ablation_scale_for_shape(scale, quant_meta, q_int8.shape)
        return (q_int8.to(torch.float32) * expanded).to(target_dtype)

    if qkv_format == "BLHc":
        q_bhld = q_int8.permute(0, 2, 1, 3).contiguous()
        scale_bhgd = scale.permute(0, 2, 1, 3).contiguous()
    else:
        q_bhld = q_int8
        scale_bhgd = scale

    batch_size, num_heads, seq_len, head_dim = q_bhld.shape
    token_group_count = len(token_lengths)
    dim_group_count = len(dim_lengths)
    padded_seq_len = token_group_count * token_group_size
    padded_head_dim = dim_group_count * dim_group_size
    padded = q_bhld
    if padded_seq_len != seq_len or padded_head_dim != head_dim:
        padded = torch.nn.functional.pad(
            q_bhld,
            (0, padded_head_dim - head_dim, 0, padded_seq_len - seq_len),
        )
    grouped = padded.view(
        batch_size,
        num_heads,
        token_group_count,
        token_group_size,
        dim_group_count,
        dim_group_size,
    )
    dequant_grouped = grouped.to(torch.float32) * scale_bhgd[:, :, :, None, :, None]
    dequant_bhld = dequant_grouped.view(batch_size, num_heads, padded_seq_len, padded_head_dim)[
        :, :, :seq_len, :head_dim
    ].to(target_dtype)
    if qkv_format == "BLHc":
        return dequant_bhld.permute(0, 2, 1, 3).contiguous()
    return dequant_bhld.contiguous()


class FLexGenStatsCache:
    def __init__(self, root: Optional[str]):
        self.root = Path(root).expanduser().resolve() if root else None

    def load(self, blk_idx: int, kv_role: str, head_idx: int, chunk_idx: int) -> Optional[torch.Tensor]:
        if self.root is None:
            return None
        path = (
            self.root
            / f"block_{int(blk_idx):02d}"
            / f"{str(kv_role)}_head{int(head_idx):02d}_chunk{int(chunk_idx):04d}.pt"
        )
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        H = payload["H"] if isinstance(payload, dict) and "H" in payload else payload
        if not torch.is_tensor(H) or H.ndim != 2:
            raise ValueError(f"Invalid Hessian payload at {path}")
        return H.float()


def _compute_group_params_signed(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    bound_max = (2 ** (bits - 1)) - 1
    max_abs = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = (max_abs / float(bound_max)).clamp_min(1e-12)
    zero = torch.full_like(scale, float(2 ** (bits - 1)))
    return scale, zero


def _flexgen_quantize_matrix(
    weight: torch.Tensor,
    H: Optional[torch.Tensor],
    bits: int,
    group_size: int,
    block_size: int,
    percdamp: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    W = weight.float().clone()
    rows, columns = W.shape
    if H is None:
        H = W.t().matmul(W)
    H = H.to(device=W.device, dtype=torch.float32)
    if H.shape != (columns, columns):
        raise ValueError(f"Expected H shape {(columns, columns)}, got {tuple(H.shape)}")
    H = torch.nan_to_num((H + H.t()) * 0.5, nan=0.0, posinf=0.0, neginf=0.0)

    dead = torch.diag(H) == 0
    if torch.any(dead):
        H = H.clone()
        H[dead, dead] = 1
        W[:, dead] = 0

    diag_vals = torch.diag(H)
    positive_diag = diag_vals[diag_vals > 0]
    diag_mean = positive_diag.mean() if positive_diag.numel() else torch.tensor(1.0, device=W.device)
    damp = float(percdamp) * float(diag_mean.item())
    if not math.isfinite(damp) or damp <= 0:
        damp = 1e-4
    diag = torch.arange(columns, device=W.device)
    last_info = None
    for retry in range(10):
        H_work = H.clone()
        H_work[diag, diag] += damp * (10 ** retry)
        Hchol, info = torch.linalg.cholesky_ex(H_work)
        if int(info.max().item()) == 0:
            H = Hchol
            last_info = None
            break
        last_info = int(info.max().item())
    if last_info is not None:
        # KV calibration statistics can be rank-deficient for early chunks. Preserve the
        # FLexGen inverse path by projecting the PSD estimate onto a strictly positive
        # spectrum instead of falling back to plain RTN or aborting the benchmark.
        evals, evecs = torch.linalg.eigh(H)
        floor = max(float(damp), 1e-4)
        H_work = (evecs * evals.clamp_min(floor).unsqueeze(0)).matmul(evecs.t())
        H_work = torch.nan_to_num((H_work + H_work.t()) * 0.5, nan=0.0, posinf=0.0, neginf=0.0)
        H_work[diag, diag] += floor
        Hchol, info = torch.linalg.cholesky_ex(H_work)
        if int(info.max().item()) != 0:
            H_work = torch.diag(torch.diag(H).clamp_min(floor))
            Hchol = torch.linalg.cholesky(H_work)
        H = Hchol
    H = torch.cholesky_inverse(H)
    H = (H + H.t()) * 0.5
    Hchol, info = torch.linalg.cholesky_ex(H, upper=True)
    if int(info.max().item()) != 0:
        H[diag, diag] += torch.finfo(H.dtype).eps * 1024
        Hchol = torch.linalg.cholesky(H, upper=True)
    Hinv = Hchol

    q_signed = torch.zeros_like(W, dtype=torch.int8)
    group_scales: List[torch.Tensor] = []
    cur_scale: Optional[torch.Tensor] = None
    cur_zero: Optional[torch.Tensor] = None

    for group_start in range(0, columns, group_size):
        group_end = min(group_start + group_size, columns)
        cur_scale, cur_zero = _compute_group_params_signed(W[:, group_start:group_end], bits)
        group_scales.append(cur_scale)
        for global_col in range(group_start, group_end):
            assert cur_scale is not None and cur_zero is not None
            w = W[:, global_col]
            d = Hinv[global_col, global_col]
            q_codes = torch.clamp(
                torch.round(w.unsqueeze(1) / cur_scale) + cur_zero,
                0,
                (2 ** bits) - 1,
            )
            q = (q_codes - cur_zero).flatten()
            q_float = q.float() * cur_scale.flatten()
            q_signed[:, global_col] = q.to(torch.int8)
            err = (w - q_float) / d
            if global_col + 1 < columns:
                W[:, global_col + 1:] -= err.unsqueeze(1) @ Hinv[global_col, global_col + 1:].unsqueeze(0)

    scales = torch.cat(group_scales, dim=1)
    return q_signed, scales


class AblationKVQuantizer:
    _COMPACT_SCALE_METHODS = frozenset(("ABL_KV_FLexGen",))

    def __init__(
        self,
        quant_bits: int = 4,
        qkv_format: str = "BLHc",
        quant_method: str = "ABL_KIVI",
        kv_role: str = "k",
        blk_idx: int = 0,
        pack_to_int32: bool = True,
        eps: float = 1e-12,
        debug: bool = False,
        rescale_qk: bool = False,
        dequant_dtype: str = "bf16",
        kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
        kivi_cali_k_group_size: int = DEFAULT_KIVI_CALI_K_GROUP_SIZE,
        kivi_cali_v_group_size: int = DEFAULT_KIVI_CALI_V_GROUP_SIZE,
        ablation_config: Optional[Dict[str, Any]] = None,
    ):
        if qkv_format not in SUPPORTED_QKV_FORMATS:
            raise ValueError(f"Unsupported qkv_format={qkv_format}")
        if not is_ablation_method(quant_method):
            raise ValueError(f"Unsupported ablation quant_method={quant_method}")
        self.quant_bits = int(quant_bits)
        self.qkv_format = qkv_format
        self.quant_method = normalize_ablation_method(str(quant_method))
        self.kv_role = _normalize_kv_role(kv_role)
        self.pack_to_int32 = bool(pack_to_int32)
        self.eps = float(eps)
        self.debug = bool(debug)
        self.rescale_qk_enabled = bool(rescale_qk)
        self.dequant_dtype = resolve_dequant_dtype(dequant_dtype)
        self.bound_min = -(2 ** (self.quant_bits - 1))
        self.bound_max = (2 ** (self.quant_bits - 1)) - 1
        self.cur_blk_idx = int(blk_idx)

        self.kivi_group_size = int(kivi_group_size)
        self.kivi_cali_k_group_size = int(kivi_cali_k_group_size)
        self.kivi_cali_v_group_size = int(kivi_cali_v_group_size)
        self.ablation_config = build_ablation_config(ablation_config, quant_method=quant_method)
        if self.ablation_config is None:
            self.ablation_config = AblationConfig(method=self.quant_method)
        self._stats_cache = FLexGenStatsCache(self.ablation_config.calibration_cache)
        self._chunk_index = 0
        self._nuq_codebook: Optional[torch.Tensor] = None

        self.quantized_item: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        self.quant_meta: Optional[Dict[str, Any]] = None
        self.cached_item: Optional[torch.Tensor] = None
        self.cached_scale: Optional[torch.Tensor] = None
        self.cached_quant_meta: Optional[Dict[str, Any]] = None
        self._pack_meta: Optional[Dict[str, int]] = None
        self._replace_cache_on_next_cache = False

    def _to_bhld(self, tensor: torch.Tensor) -> torch.Tensor:
        return _to_bhld_layout(tensor, self.qkv_format)

    def _from_bhld(self, tensor: torch.Tensor) -> torch.Tensor:
        return _from_bhld_layout(tensor, self.qkv_format)

    def _compute_scale(self, x: torch.Tensor, reduce_dims: Tuple[int, ...], keepdim: bool = True) -> torch.Tensor:
        max_abs = x.to(torch.float32).abs().amax(dim=reduce_dims, keepdim=keepdim)
        return (max_abs / float(self.bound_max)).clamp_min(self.eps)

    def _quantize_to_int8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return torch.round(x.to(torch.float32) / scale).clamp(self.bound_min, self.bound_max).to(torch.int8)

    def _dequantize_from_int8(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (q.to(torch.float32) * scale).to(self.dequant_dtype)

    def _load_nuq_codebook(self) -> torch.Tensor:
        if self._nuq_codebook is not None:
            return self._nuq_codebook
        path = self.ablation_config.nuq_codebook_path
        if not path:
            raise ValueError("ABL_KVQUANT use_nuq=true requires nuq_codebook_path.")
        payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
        role_key = f"{self.kv_role}_codebook"
        if isinstance(payload, dict):
            codebook = payload.get(role_key, payload.get("codebook"))
        else:
            codebook = payload
        if not torch.is_tensor(codebook):
            raise ValueError(f"Invalid NUQ codebook payload at {path}")
        codebook = codebook.float().flatten()
        if codebook.numel() != 2 ** self.quant_bits:
            raise ValueError(
                f"Expected {2 ** self.quant_bits} codebook values, got {codebook.numel()} from {path}"
            )
        self._nuq_codebook = codebook
        return self._nuq_codebook

    def _pack_last_dim_to_int32(self, q_int8: torch.Tensor, bits: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        if q_int8.is_cuda and bits in CUDA_PACK_BITS:
            return pack_last_dim_to_int32_cuda(q_int8, bits)
        return pack_last_dim_to_int32_python(q_int8, bits)

    def _unpack_last_dim_from_int32(self, packed: torch.Tensor, meta: Dict[str, int]) -> torch.Tensor:
        bits = int(meta["bits"])
        if packed.is_cuda and bits in CUDA_PACK_BITS:
            return unpack_last_dim_from_int32_cuda(packed, meta)
        return unpack_last_dim_from_int32_python(packed, meta)

    def _quantize_kivi(self, item: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        bhld = self._to_bhld(item)
        _, _, seq_len, head_dim = bhld.shape
        if self.kv_role == "k":
            group_lengths = _group_lengths_for_axis(seq_len, self.kivi_group_size)
            q_chunks: List[torch.Tensor] = []
            scale_chunks: List[torch.Tensor] = []
            start = 0
            for group_len in group_lengths:
                chunk = bhld[:, :, start:start + group_len, :]
                scale_chunk = self._compute_scale(chunk, reduce_dims=(2,), keepdim=True)
                q_chunks.append(self._quantize_to_int8(chunk, scale_chunk))
                scale_chunks.append(scale_chunk.squeeze(2))
                start += group_len
            q_bhld = torch.cat(q_chunks, dim=2)
            scale = self._from_bhld(torch.stack(scale_chunks, dim=2))
            quant_meta = {
                "scheme": "KIVI",
                "kv_role": self.kv_role,
                "group_size": int(self.kivi_group_size),
                "group_lengths": list(group_lengths),
                "qkv_format": self.qkv_format,
                "orig_shape": tuple(int(v) for v in item.shape),
            }
            return self._from_bhld(q_bhld), scale, quant_meta
        group_lengths = _group_lengths_for_axis(head_dim, self.kivi_group_size)
        q_chunks = []
        scale_chunks = []
        start = 0
        for group_len in group_lengths:
            chunk = bhld[:, :, :, start:start + group_len]
            scale_chunk = self._compute_scale(chunk, reduce_dims=(3,), keepdim=True)
            q_chunks.append(self._quantize_to_int8(chunk, scale_chunk))
            scale_chunks.append(scale_chunk)
            start += group_len
        q_bhld = torch.cat(q_chunks, dim=3)
        scale = self._from_bhld(torch.cat(scale_chunks, dim=3))
        quant_meta = {
            "scheme": "KIVI",
            "kv_role": self.kv_role,
            "group_size": int(self.kivi_group_size),
            "group_lengths": list(group_lengths),
            "qkv_format": self.qkv_format,
            "orig_shape": tuple(int(v) for v in item.shape),
        }
        return self._from_bhld(q_bhld), scale, quant_meta

    def _quantize_kivi_cali(self, item: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        bhld = self._to_bhld(item)
        batch_size, num_heads, seq_len, head_dim = bhld.shape
        if self.kv_role == "k":
            group_lengths = _group_lengths_for_axis(seq_len, self.kivi_cali_k_group_size)
            q_chunks: List[torch.Tensor] = []
            scale_chunks: List[torch.Tensor] = []
            start = 0
            for group_len in group_lengths:
                chunk = bhld[:, :, start:start + group_len, :]
                scale_chunk = self._compute_scale(chunk, reduce_dims=(2,), keepdim=True)
                q_chunks.append(self._quantize_to_int8(chunk, scale_chunk))
                scale_chunks.append(scale_chunk.squeeze(2))
                start += group_len
            q_bhld = torch.cat(q_chunks, dim=2)
            scale = self._from_bhld(torch.stack(scale_chunks, dim=2))
            return self._from_bhld(q_bhld), scale, {
                "scheme": "KIVI-CALI",
                "kv_role": self.kv_role,
                "group_size": int(self.kivi_cali_k_group_size),
                "group_lengths": list(group_lengths),
                "group_axis": "TOKEN",
                "qkv_format": self.qkv_format,
                "orig_shape": tuple(int(v) for v in item.shape),
            }

        flat = bhld.permute(0, 2, 1, 3).contiguous().reshape(batch_size, seq_len, num_heads * head_dim)
        group_lengths = _group_lengths_for_axis(num_heads * head_dim, self.kivi_cali_v_group_size)
        q_chunks = []
        scale_chunks = []
        start = 0
        for group_len in group_lengths:
            chunk = flat[:, :, start:start + group_len]
            scale_chunk = self._compute_scale(chunk, reduce_dims=(2,), keepdim=True)
            q_chunks.append(self._quantize_to_int8(chunk, scale_chunk))
            scale_chunks.append(scale_chunk)
            start += group_len
        q_flat = torch.cat(q_chunks, dim=2)
        q_bhld = q_flat.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        scale_blg = torch.cat(scale_chunks, dim=2)
        scale = scale_blg.unsqueeze(2) if self.qkv_format == "BLHc" else scale_blg.unsqueeze(1)
        return self._from_bhld(q_bhld), scale, {
            "scheme": "KIVI-CALI",
            "kv_role": self.kv_role,
            "group_size": int(self.kivi_cali_v_group_size),
            "group_lengths": list(group_lengths),
            "group_axis": "HD_FLAT",
            "qkv_format": self.qkv_format,
            "orig_shape": tuple(int(v) for v in item.shape),
            "num_heads": int(num_heads),
            "head_dim": int(head_dim),
        }

    def _uniform_codes_from_scale(self, dense_qkv: torch.Tensor, scale_qkv: torch.Tensor) -> torch.Tensor:
        return self._quantize_to_int8(dense_qkv, scale_qkv)

    def _nuq_codes_from_scale(self, dense_qkv: torch.Tensor, scale_qkv: torch.Tensor) -> torch.Tensor:
        codebook = self._load_nuq_codebook().to(device=dense_qkv.device, dtype=torch.float32)
        normalized = (dense_qkv.float() / scale_qkv.float()).clamp(-1.0, 1.0)
        cb = codebook.view(*([1] * normalized.ndim), -1)
        distances = (normalized.unsqueeze(-1) - cb).abs()
        indices = torch.argmin(distances, dim=-1)
        values = codebook[indices]
        signed = torch.round(values * float(self.bound_max)).clamp(self.bound_min, self.bound_max)
        return signed.to(torch.int8)

    def _extract_sparse_outliers(
        self,
        bhld: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, num_heads, seq_len, head_dim = bhld.shape
        sink_tokens = min(int(self.ablation_config.sink_fp16_tokens), seq_len)
        keep_count = int(self.ablation_config.outlier_cap_per_token)
        if keep_count <= 0:
            keep_count = int(math.ceil(float(self.ablation_config.outlier_ratio) * head_dim))
        keep_count = max(0, min(head_dim, keep_count))
        if sink_tokens == 0 and keep_count == 0:
            return bhld, None, None

        working = bhld.clone()
        coords_list: List[torch.Tensor] = []
        values_list: List[torch.Tensor] = []

        if sink_tokens > 0:
            sink_values = working[:, :, :sink_tokens, :].reshape(-1, head_dim)
            if sink_values.numel() > 0:
                b_ids = torch.arange(batch_size, device=bhld.device).view(batch_size, 1, 1, 1)
                h_ids = torch.arange(num_heads, device=bhld.device).view(1, num_heads, 1, 1)
                l_ids = torch.arange(sink_tokens, device=bhld.device).view(1, 1, sink_tokens, 1)
                d_ids = torch.arange(head_dim, device=bhld.device).view(1, 1, 1, head_dim)
                sink_coords = torch.stack(
                    (
                        b_ids.expand(batch_size, num_heads, sink_tokens, head_dim),
                        h_ids.expand(batch_size, num_heads, sink_tokens, head_dim),
                        l_ids.expand(batch_size, num_heads, sink_tokens, head_dim),
                        d_ids.expand(batch_size, num_heads, sink_tokens, head_dim),
                    ),
                    dim=-1,
                ).reshape(-1, 4)
                coords_list.append(sink_coords)
                values_list.append(sink_values.reshape(-1))
            working[:, :, :sink_tokens, :] = 0

        if keep_count > 0 and seq_len > sink_tokens:
            token_count = seq_len - sink_tokens
            tail = working[:, :, sink_tokens:, :].clone()
            flat = tail.reshape(-1, head_dim)
            if flat.numel() > 0:
                topk_idx = flat.abs().topk(k=keep_count, dim=1, largest=True, sorted=False).indices
                topk_vals = torch.gather(flat, 1, topk_idx)
                vector_count = flat.size(0)
                b_base = torch.arange(batch_size, device=bhld.device).repeat_interleave(num_heads * token_count)
                h_base = torch.arange(num_heads, device=bhld.device).repeat_interleave(token_count).repeat(batch_size)
                l_base = torch.arange(sink_tokens, seq_len, device=bhld.device).repeat(batch_size * num_heads)
                coords = torch.stack(
                    (
                        b_base.unsqueeze(1).expand(vector_count, keep_count),
                        h_base.unsqueeze(1).expand(vector_count, keep_count),
                        l_base.unsqueeze(1).expand(vector_count, keep_count),
                        topk_idx,
                    ),
                    dim=-1,
                ).reshape(-1, 4)
                coords_list.append(coords)
                values_list.append(topk_vals.reshape(-1))
                flat.scatter_(1, topk_idx, 0.0)
                working[:, :, sink_tokens:, :] = tail

        if not coords_list:
            return working, None, None
        coords_bhld = torch.cat(coords_list, dim=0).to(torch.int32)
        values = torch.cat(values_list, dim=0).to(torch.float32)
        if self.qkv_format == "BLHc":
            coords_qkv = coords_bhld[:, [0, 2, 1, 3]].contiguous()
        else:
            coords_qkv = coords_bhld
        return working, coords_qkv, values

    def _quantize_dense_with_group_axis(
        self,
        dense_bhld: torch.Tensor,
        group_axis: str,
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        if group_axis == "TOKEN":
            seq_len = dense_bhld.shape[2]
            group_lengths = _group_lengths_for_axis(seq_len, group_size)
            q_chunks: List[torch.Tensor] = []
            scale_chunks: List[torch.Tensor] = []
            start = 0
            for group_len in group_lengths:
                chunk = dense_bhld[:, :, start:start + group_len, :]
                scale_chunk = self._compute_scale(chunk, reduce_dims=(2,), keepdim=True)
                if self.ablation_config.use_nuq:
                    q_chunk = self._nuq_codes_from_scale(self._from_bhld(chunk), self._from_bhld(scale_chunk.expand_as(chunk)))
                    q_chunk = self._to_bhld(q_chunk)
                else:
                    q_chunk = self._quantize_to_int8(chunk, scale_chunk)
                q_chunks.append(q_chunk)
                scale_chunks.append(scale_chunk.squeeze(2))
                start += group_len
            q_bhld = torch.cat(q_chunks, dim=2)
            scale = self._from_bhld(torch.stack(scale_chunks, dim=2))
            return q_bhld, scale, group_lengths
        head_dim = dense_bhld.shape[3]
        group_lengths = _group_lengths_for_axis(head_dim, group_size)
        q_chunks = []
        scale_chunks = []
        start = 0
        for group_len in group_lengths:
            chunk = dense_bhld[:, :, :, start:start + group_len]
            scale_chunk = self._compute_scale(chunk, reduce_dims=(3,), keepdim=True)
            if self.ablation_config.use_nuq:
                q_chunk = self._nuq_codes_from_scale(self._from_bhld(chunk), self._from_bhld(scale_chunk.expand_as(chunk)))
                q_chunk = self._to_bhld(q_chunk)
            else:
                q_chunk = self._quantize_to_int8(chunk, scale_chunk)
            q_chunks.append(q_chunk)
            scale_chunks.append(scale_chunk)
            start += group_len
        q_bhld = torch.cat(q_chunks, dim=3)
        scale = self._from_bhld(torch.cat(scale_chunks, dim=3))
        return q_bhld, scale, group_lengths

    def _quantize_dense_2d_blocks(
        self,
        dense_bhld: torch.Tensor,
        token_group_size: int,
        dim_group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        batch_size, num_heads, seq_len, head_dim = dense_bhld.shape
        token_lengths = _group_lengths_for_axis(seq_len, token_group_size)
        dim_lengths = _group_lengths_for_axis(head_dim, dim_group_size)
        token_group_count = len(token_lengths)
        dim_group_count = len(dim_lengths)
        padded_seq_len = token_group_count * int(token_group_size)
        padded_head_dim = dim_group_count * int(dim_group_size)

        padded = dense_bhld
        if padded_seq_len != seq_len or padded_head_dim != head_dim:
            padded = torch.nn.functional.pad(
                dense_bhld,
                (0, padded_head_dim - head_dim, 0, padded_seq_len - seq_len),
            )
        grouped = padded.view(
            batch_size,
            num_heads,
            token_group_count,
            int(token_group_size),
            dim_group_count,
            int(dim_group_size),
        )
        scale_grouped = self._compute_scale(grouped, reduce_dims=(3, 5), keepdim=True)
        q_grouped = self._quantize_to_int8(grouped, scale_grouped)
        q_bhld = q_grouped.view(batch_size, num_heads, padded_seq_len, padded_head_dim)[
            :, :, :seq_len, :head_dim
        ].contiguous()
        scale_bhgd = scale_grouped.squeeze(5).squeeze(3).contiguous()

        scale = scale_bhgd.permute(0, 2, 1, 3).contiguous() if self.qkv_format == "BLHc" else scale_bhgd
        return q_bhld, scale, token_lengths, dim_lengths

    def _quantize_kvquant(self, item: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        bhld = self._to_bhld(item)
        dense_bhld, sparse_coords, sparse_values = self._extract_sparse_outliers(bhld)
        if self.kv_role == "k":
            q_bhld, scale, group_lengths = self._quantize_dense_with_group_axis(
                dense_bhld,
                group_axis="TOKEN",
                group_size=self.ablation_config.group_size_k,
            )
            meta = {
                "scheme": "ABL_KVQUANT",
                "kv_role": self.kv_role,
                "group_axis": "TOKEN",
                "group_lengths": list(group_lengths),
                "qkv_format": self.qkv_format,
                "orig_shape": tuple(int(v) for v in item.shape),
                "use_nuq": bool(self.ablation_config.use_nuq),
                "pre_rope": bool(self.ablation_config.use_pre_rope_k),
                "sink_fp16_tokens": int(self.ablation_config.sink_fp16_tokens),
            }
        else:
            q_bhld, scale, group_lengths = self._quantize_dense_with_group_axis(
                dense_bhld,
                group_axis="D",
                group_size=self.ablation_config.group_size_v,
            )
            meta = {
                "scheme": "ABL_KVQUANT",
                "kv_role": self.kv_role,
                "group_axis": "D",
                "group_lengths": list(group_lengths),
                "qkv_format": self.qkv_format,
                "orig_shape": tuple(int(v) for v in item.shape),
                "use_nuq": bool(self.ablation_config.use_nuq),
                "pre_rope": False,
                "sink_fp16_tokens": int(self.ablation_config.sink_fp16_tokens),
            }
        if sparse_coords is not None and sparse_values is not None:
            meta["sparse_coords"] = sparse_coords.cpu()
            meta["sparse_values"] = sparse_values.cpu()
        else:
            meta["sparse_coords"] = torch.empty((0, 4), dtype=torch.int32)
            meta["sparse_values"] = torch.empty((0,), dtype=torch.float32)
        return self._from_bhld(q_bhld), scale, meta

    def _flexgen_hessian_for_head(self, matrix: torch.Tensor, head_idx: int) -> torch.Tensor:
        cached = self._stats_cache.load(self.cur_blk_idx, self.kv_role, head_idx, self._chunk_index)
        if cached is not None:
            return cached.to(device=matrix.device, dtype=torch.float32)
        return matrix.float().t().matmul(matrix.float())

    def _quantize_kv_flexgen(self, item: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        bhld = self._to_bhld(item)
        dim_group_size = (
            self.ablation_config.group_size_k if self.kv_role == "k" else self.ablation_config.group_size_v
        )
        q_bhld, scale, token_lengths, dim_lengths = self._quantize_dense_2d_blocks(
            bhld,
            token_group_size=self.ablation_config.block_size,
            dim_group_size=dim_group_size,
        )
        return self._from_bhld(q_bhld), scale, {
            "scheme": "ABL_KV_FLexGen",
            "kv_role": self.kv_role,
            "group_axis": "BLOCK2D",
            "token_group_lengths": list(token_lengths),
            "dim_group_lengths": list(dim_lengths),
            "block_size": int(self.ablation_config.block_size),
            "dim_group_size": int(dim_group_size),
            "qkv_format": self.qkv_format,
            "orig_shape": tuple(int(v) for v in item.shape),
        }

    def _expand_scale_for_tensor(
        self,
        scale: torch.Tensor,
        quant_meta: Optional[Dict[str, Any]],
        target_shape: Sequence[int],
    ) -> torch.Tensor:
        if _is_flexgen_meta(quant_meta) or _is_kvquant_meta(quant_meta):
            return _expand_ablation_scale_for_shape(scale, quant_meta, target_shape)
        if quant_meta and str(quant_meta.get("scheme", "")) in ("KIVI", "KIVI-CALI"):
            return _expand_kivi_scale_for_shape(scale, quant_meta, target_shape)
        return scale

    def quant(self, item: torch.Tensor):
        self.quant_meta = None
        self._replace_cache_on_next_cache = False
        if self.quant_method == "ABL_KIVI":
            q, scale, meta = self._quantize_kivi(item)
        elif self.quant_method == "ABL_KIVI_CALI":
            if self.kv_role == "v" and self.cached_item is not None and self.cached_scale is not None:
                item = torch.cat([self.dequant_all(), item], dim=1 if self.qkv_format == "BLHc" else 2)
                self._replace_cache_on_next_cache = True
            q, scale, meta = self._quantize_kivi_cali(item)
        elif self.quant_method == "ABL_KVQUANT":
            q, scale, meta = self._quantize_kvquant(item)
        else:
            q, scale, meta = self._quantize_kv_flexgen(item)
        self.quant_meta = meta
        if self.pack_to_int32:
            q_packed, meta_pack = self._pack_last_dim_to_int32(q, self.quant_bits)
            self._pack_meta = meta_pack
            self.quantized_item = q_packed
        else:
            self.quantized_item = q
        self.scale = scale

    def cache(self):
        q_cur = self.quantized_item
        s_cur = self.scale
        meta_cur = _clone_quant_meta(self.quant_meta)
        if self.cached_item is None or self._replace_cache_on_next_cache:
            self.cached_item, self.cached_scale = q_cur, s_cur
            self.cached_quant_meta = meta_cur
        else:
            cat_dim = 1 if self.qkv_format == "BLHc" else 2
            self.cached_item = torch.cat([self.cached_item, q_cur], dim=cat_dim)
            if _is_flexgen_meta(meta_cur):
                if str(meta_cur.get("group_axis")) == "TOKEN":
                    self.cached_scale = torch.cat([self.cached_scale, s_cur], dim=2)
                else:
                    self.cached_scale = torch.cat([self.cached_scale, s_cur], dim=cat_dim)
            else:
                self.cached_scale = torch.cat([self.cached_scale, s_cur], dim=cat_dim)
            self.cached_quant_meta = _merge_ablation_quant_meta(self.cached_quant_meta, meta_cur)
        self._replace_cache_on_next_cache = False
        self._chunk_index += 1

    def dequant_current(self) -> torch.Tensor:
        if self.pack_to_int32:
            q_int8 = self._unpack_last_dim_from_int32(self.quantized_item, self._pack_meta)
        else:
            q_int8 = self.quantized_item
        scale = self._expand_scale_for_tensor(self.scale, self.quant_meta, q_int8.shape)
        dequant = self._dequantize_from_int8(q_int8, scale)
        return _apply_sparse_exact_values(dequant, self.quant_meta)

    def dequant_all(self) -> torch.Tensor:
        if self.cached_item is None:
            return self.dequant_current()
        if self.pack_to_int32:
            q_int8 = self._unpack_last_dim_from_int32(self.cached_item, self._pack_meta)
        else:
            q_int8 = self.cached_item
        scale = self._expand_scale_for_tensor(self.cached_scale, self.cached_quant_meta, q_int8.shape)
        dequant = self._dequantize_from_int8(q_int8, scale)
        return _apply_sparse_exact_values(dequant, self.cached_quant_meta)

    def quant_and_cache(self, item: torch.Tensor):
        self.quant(item)
        self.cache()

    def clear_cache(self):
        self.quantized_item = None
        self.scale = None
        self.quant_meta = None
        self.cached_item = None
        self.cached_scale = None
        self.cached_quant_meta = None
        self._pack_meta = None
        self._replace_cache_on_next_cache = False
        self._chunk_index = 0

    def use_var_q(self, item: torch.Tensor) -> torch.Tensor:
        self.quant_and_cache(item)
        return self.dequant_all()

    def rescale_qk(self, q: torch.Tensor, k: torch.Tensor, return_theta: bool = False):
        theta = None
        if self.rescale_qk_enabled:
            range_q = q.abs().max()
            range_k = k.abs().max()
            theta = torch.sqrt((range_k + 1e-6) / (range_q + 1e-6))
            q = q * theta
            k = k / theta
        if return_theta:
            return q, k, theta
        return q, k


class InfinityStarAblationCache:
    def __init__(
        self,
        quant_bits: int = 4,
        qkv_format: str = "BHLc",
        quant_method: str = "ABL_KIVI",
        kv_role: str = "k",
        kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
        kivi_cali_k_group_size: int = DEFAULT_KIVI_CALI_K_GROUP_SIZE,
        kivi_cali_v_group_size: int = DEFAULT_KIVI_CALI_V_GROUP_SIZE,
        pack_to_int32: bool = True,
        eps: float = 1e-12,
        debug: bool = False,
        rescale_qk: bool = False,
        dequant_dtype: str = "bf16",
        ablation_config: Optional[Dict[str, Any]] = None,
    ):
        self.quant_bits = quant_bits
        self.qkv_format = qkv_format
        self.quant_method = quant_method
        self.kv_role = _normalize_kv_role(kv_role)
        self.kivi_group_size = kivi_group_size
        self.kivi_cali_k_group_size = kivi_cali_k_group_size
        self.kivi_cali_v_group_size = kivi_cali_v_group_size
        self.pack_to_int32 = pack_to_int32
        self.eps = eps
        self.debug = debug
        self.rescale_qk = rescale_qk
        self.dequant_dtype = dequant_dtype
        self.ablation_config = ablation_config
        self._scale_quantizers: Dict[Union[int, str], AblationKVQuantizer] = {}

    def _new_quantizer(self, scale_id: Union[int, str]) -> AblationKVQuantizer:
        blk_idx = int(scale_id) if isinstance(scale_id, int) else 0
        return AblationKVQuantizer(
            quant_bits=self.quant_bits,
            qkv_format=self.qkv_format,
            quant_method=self.quant_method,
            kv_role=self.kv_role,
            blk_idx=blk_idx,
            kivi_group_size=self.kivi_group_size,
            kivi_cali_k_group_size=self.kivi_cali_k_group_size,
            kivi_cali_v_group_size=self.kivi_cali_v_group_size,
            pack_to_int32=self.pack_to_int32,
            eps=self.eps,
            debug=self.debug,
            rescale_qk=self.rescale_qk,
            dequant_dtype=self.dequant_dtype,
            ablation_config=self.ablation_config,
        )

    def _ensure_quantizer(self, scale_id: Union[int, str]) -> AblationKVQuantizer:
        if scale_id not in self._scale_quantizers:
            self._scale_quantizers[scale_id] = self._new_quantizer(scale_id)
        return self._scale_quantizers[scale_id]

    def cache_scale(self, scale_id: Union[int, str], kv_tensor: torch.Tensor, overwrite: bool = True, return_dequant: bool = True):
        q = self._ensure_quantizer(scale_id)
        if overwrite:
            q.cached_item = None
            q.cached_scale = None
            q.cached_quant_meta = None
            q.quantized_item = None
            q.scale = None
            q.quant_meta = None
            q._pack_meta = None
        if return_dequant:
            return q.use_var_q(kv_tensor)
        q.quant_and_cache(kv_tensor)
        return None

    def get_scale(self, scale_id: Union[int, str]) -> torch.Tensor:
        return self._ensure_quantizer(scale_id).dequant_all()

    def get_scale_quantized(self, scale_id: Union[int, str]):
        q = self._ensure_quantizer(scale_id)
        return q.cached_item, q.cached_scale, q._pack_meta

    def clear_scales(self, scale_ids: Iterable[Union[int, str]]) -> None:
        for sid in scale_ids:
            self._scale_quantizers.pop(sid, None)

    def cache_bytes(self) -> Dict[str, int]:
        packed_bytes = 0
        scale_bytes = 0
        for q in self._scale_quantizers.values():
            if q.cached_item is not None:
                packed_bytes += q.cached_item.numel() * q.cached_item.element_size()
            if q.cached_scale is not None:
                scale_bytes += q.cached_scale.numel() * q.cached_scale.element_size()
        return {"packed_bytes": packed_bytes, "scale_bytes": scale_bytes, "total_bytes": packed_bytes + scale_bytes}

    def live_scale_ids(self) -> List[Union[int, str]]:
        return sorted(list(self._scale_quantizers.keys()), key=lambda x: str(x))


def build_kv_cache_quantizer(
    quant_bits: int,
    qkv_format: str,
    quant_method: str,
    kv_role: str = "k",
    blk_idx: int = 0,
    pack_to_int32: bool = True,
    eps: float = 1e-12,
    debug: bool = False,
    rescale_qk: bool = False,
    dequant_dtype: str = "bf16",
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_cali_k_group_size: int = DEFAULT_KIVI_CALI_K_GROUP_SIZE,
    kivi_cali_v_group_size: int = DEFAULT_KIVI_CALI_V_GROUP_SIZE,
    ablation_config: Optional[Dict[str, Any]] = None,
    compression_ratio: float = 1.0,
    max_scale_seq_len: Optional[int] = None,
    **_ignored: Any,
):
    quant_method = normalize_ablation_method(quant_method)
    if not is_ablation_method(quant_method):
        raise ValueError(f"build_kv_cache_quantizer only supports ablation methods, got {quant_method}")
    return AblationKVQuantizer(
        quant_bits=quant_bits,
        qkv_format=qkv_format,
        quant_method=quant_method,
        kv_role=kv_role,
        blk_idx=blk_idx,
        pack_to_int32=pack_to_int32,
        eps=eps,
        debug=debug,
        rescale_qk=rescale_qk,
        dequant_dtype=dequant_dtype,
        kivi_group_size=kivi_group_size,
        kivi_cali_k_group_size=kivi_cali_k_group_size,
        kivi_cali_v_group_size=kivi_cali_v_group_size,
        ablation_config=ablation_config,
    )


def build_infinitystar_cache_quantizer(
    quant_bits: int,
    qkv_format: str,
    quant_method: str,
    kv_role: str = "k",
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_cali_k_group_size: int = DEFAULT_KIVI_CALI_K_GROUP_SIZE,
    kivi_cali_v_group_size: int = DEFAULT_KIVI_CALI_V_GROUP_SIZE,
    pack_to_int32: bool = True,
    eps: float = 1e-12,
    debug: bool = False,
    rescale_qk: bool = False,
    dequant_dtype: str = "bf16",
    ablation_config: Optional[Dict[str, Any]] = None,
    compression_ratio: float = 1.0,
    max_scale_seq_len: Optional[int] = None,
    **_ignored: Any,
):
    quant_method = normalize_ablation_method(quant_method)
    if not is_ablation_method(quant_method):
        raise ValueError(
            f"build_infinitystar_cache_quantizer only supports ablation methods, got {quant_method}"
        )
    return InfinityStarAblationCache(
        quant_bits=quant_bits,
        qkv_format=qkv_format,
        quant_method=quant_method,
        kv_role=kv_role,
        kivi_group_size=kivi_group_size,
        kivi_cali_k_group_size=kivi_cali_k_group_size,
        kivi_cali_v_group_size=kivi_cali_v_group_size,
        pack_to_int32=pack_to_int32,
        eps=eps,
        debug=debug,
        rescale_qk=rescale_qk,
        dequant_dtype=dequant_dtype,
        ablation_config=ablation_config,
    )


def quantize_tensor(
    item: torch.Tensor,
    quant_bits: int,
    qkv_format: str,
    quant_method: str,
    pack_to_int32: bool,
    dequant_dtype: str | torch.dtype,
    kv_role: str = "k",
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_cali_k_group_size: int = DEFAULT_KIVI_CALI_K_GROUP_SIZE,
    kivi_cali_v_group_size: int = DEFAULT_KIVI_CALI_V_GROUP_SIZE,
    ablation_config: Optional[Dict[str, Any]] = None,
    blk_idx: int = 0,
    compression_ratio: float = 1.0,
    max_scale_seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    if not is_ablation_method(quant_method):
        raise ValueError(f"quantize_tensor only supports ablation methods, got {quant_method}")
    quantizer = build_kv_cache_quantizer(
        quant_bits=quant_bits,
        qkv_format=qkv_format,
        quant_method=quant_method,
        kv_role=kv_role,
        blk_idx=blk_idx,
        pack_to_int32=pack_to_int32,
        dequant_dtype=_dequant_dtype_name(dequant_dtype),
        kivi_group_size=kivi_group_size,
        kivi_cali_k_group_size=kivi_cali_k_group_size,
        kivi_cali_v_group_size=kivi_cali_v_group_size,
        ablation_config=ablation_config,
        compression_ratio=compression_ratio,
        max_scale_seq_len=max_scale_seq_len,
    )
    quantizer.quant(item)
    return {
        "packed": quantizer.quantized_item,
        "scale": quantizer.scale,
        "pack_meta": quantizer._pack_meta if pack_to_int32 else None,
        "quant_meta": _clone_quant_meta(quantizer.quant_meta),
    }


def dequantize_tensor(
    packed: torch.Tensor,
    scale: torch.Tensor,
    pack_meta: Optional[Dict[str, int]],
    dequant_dtype: str | torch.dtype,
    quant_meta: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    if not (
        _is_flexgen_meta(quant_meta)
        or _is_kvquant_meta(quant_meta)
        or (quant_meta and str(quant_meta.get("scheme", "")) in ("KIVI", "KIVI-CALI"))
    ):
        return legacy_dequantize_tensor(
            packed=packed,
            scale=scale,
            pack_meta=pack_meta,
            dequant_dtype=dequant_dtype,
            quant_meta=quant_meta,
        )
    target_dtype = resolve_dequant_dtype(dequant_dtype)
    if pack_meta is not None:
        bits = int(pack_meta["bits"])
        if packed.is_cuda and bits in CUDA_PACK_BITS:
            q_int8 = unpack_last_dim_from_int32_cuda(packed, pack_meta)
        else:
            q_int8 = unpack_last_dim_from_int32_python(packed, pack_meta)
    else:
        q_int8 = packed
    if _is_flexgen_meta(quant_meta) and str(quant_meta.get("group_axis", "")) == "BLOCK2D":
        dequant = _dequantize_block2d_int8(q_int8, scale, quant_meta, target_dtype)
        return _apply_sparse_exact_values(dequant, quant_meta)
    if _is_flexgen_meta(quant_meta) or _is_kvquant_meta(quant_meta):
        scale = _expand_ablation_scale_for_shape(scale, quant_meta, q_int8.shape)
    else:
        scale = _expand_kivi_scale_for_shape(scale, quant_meta, q_int8.shape)
    dequant = (q_int8.to(torch.float32) * scale).to(target_dtype)
    return _apply_sparse_exact_values(dequant, quant_meta)


def _compute_attention_weights(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    return torch.softmax(scores, dim=-1)


def save_ablation_stats_from_qkv_dump(
    dump_dir: str,
    out_dir: str,
    chunk_lengths: Sequence[int],
) -> None:
    dump_root = Path(dump_dir).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    chunk_lengths = [int(v) for v in chunk_lengths]
    qkv_files = sorted(dump_root.glob("*.pt"))
    if not qkv_files:
        raise FileNotFoundError(f"No qkv dump files found under {dump_root}")
    accum: Dict[Tuple[int, str, int, int], torch.Tensor] = {}
    for path in qkv_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        q = payload["q"].float()
        k = payload["k"].float()
        v = payload["v"].float()
        block_idx = int(payload.get("block_idx", 0))
        head_dim = int(payload.get("head_dim", q.shape[-1]))
        scale = head_dim ** -0.5
        prefix = 0
        for chunk_idx, chunk_len in enumerate(chunk_lengths):
            start = prefix
            end = prefix + chunk_len
            prefix = end
            if end > q.shape[2]:
                break
            future_q = q[:, :, end:, :]
            if future_q.numel() > 0:
                Hk = future_q.reshape(-1, q.shape[1], q.shape[-1]).permute(1, 0, 2)
                for h in range(q.shape[1]):
                    key = (block_idx, "k", h, chunk_idx)
                    H = Hk[h].t().matmul(Hk[h])
                    accum[key] = accum.get(key, torch.zeros_like(H)) + H
            attn = _compute_attention_weights(q[:, :, end:, :], k[:, :, :end, :], scale)
            if attn.numel() > 0:
                attn_chunk = attn[:, :, :, start:end].reshape(-1, q.shape[1], chunk_len).permute(1, 0, 2)
                for h in range(q.shape[1]):
                    key = (block_idx, "v", h, chunk_idx)
                    H = attn_chunk[h].t().matmul(attn_chunk[h])
                    accum[key] = accum.get(key, torch.zeros_like(H)) + H
    for (block_idx, kv_role, head_idx, chunk_idx), H in accum.items():
        save_path = out_root / f"block_{block_idx:02d}" / f"{kv_role}_head{head_idx:02d}_chunk{chunk_idx:04d}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"H": H.cpu()}, save_path)


def save_kvquant_codebook_from_qkv_dump(
    dump_dir: str,
    out_path: str,
    bits: int = 4,
    max_samples_per_role: int = 200000,
) -> None:
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    dump_root = Path(dump_dir).expanduser().resolve()
    qkv_files = sorted(dump_root.glob("*.pt"))
    if not qkv_files:
        raise FileNotFoundError(f"No qkv dump files found under {dump_root}")

    def _reservoir_append(existing: List[torch.Tensor], new_tensor: torch.Tensor, remaining: int) -> int:
        if remaining <= 0 or new_tensor.numel() == 0:
            return remaining
        flat = new_tensor.flatten()
        take = min(int(flat.numel()), int(remaining))
        if take < flat.numel():
            perm = torch.randperm(flat.numel())[:take]
            flat = flat[perm]
        else:
            flat = flat[:take]
        existing.append(flat.cpu())
        return remaining - take

    k_samples: List[torch.Tensor] = []
    v_samples: List[torch.Tensor] = []
    k_remaining = int(max_samples_per_role)
    v_remaining = int(max_samples_per_role)
    for path in qkv_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "k" in payload:
            k_remaining = _reservoir_append(k_samples, payload["k"].float(), k_remaining)
        if "v" in payload:
            v_remaining = _reservoir_append(v_samples, payload["v"].float(), v_remaining)
        if k_remaining <= 0 and v_remaining <= 0:
            break

    def _build_codebook(chunks: List[torch.Tensor]) -> torch.Tensor:
        if not chunks:
            return torch.linspace(-1.0, 1.0, 2 ** bits, dtype=torch.float32)
        values = torch.cat(chunks, dim=0)
        clip = torch.quantile(values.abs(), torch.tensor(0.999, dtype=torch.float32)).clamp_min(1e-6)
        normalized = (values / clip).clamp(-1.0, 1.0)
        quantiles = torch.linspace(0.0, 1.0, 2 ** bits, dtype=torch.float32)
        codebook = torch.quantile(normalized, quantiles).float()
        codebook[0] = -1.0
        codebook[-1] = 1.0
        return codebook

    out_file = Path(out_path).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bits": int(bits),
            "k_codebook": _build_codebook(k_samples),
            "v_codebook": _build_codebook(v_samples),
        },
        out_file,
    )


def load_ablation_config_file(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))

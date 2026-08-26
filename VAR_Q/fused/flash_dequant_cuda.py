"""CUDA FlashAttention-backed VAR-Q fused dequant attention.

This module is the Python boundary for the Track B CUDA/CUTLASS backend.  The
extension is intentionally optional while the CUDA kernel is under development:
callers can probe availability without importing torch CUDA extensions at
package import time.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple

import torch


EXTENSION_NAME = "_varq_fused_flash"
ENV_EXTENSION_PATH = "VARQ_FUSED_FLASH_EXT_PATH"
SUPPORTED_Q_BITS = (2, 3, 4, 6, 8)
_DTYPE_TO_CODE = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}


@dataclass(frozen=True)
class FusedFlashStatus:
    available: bool
    message: str
    module_path: Optional[str] = None


_EXTENSION: Optional[ModuleType] = None
_LOAD_ERROR: Optional[BaseException] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_extension_dir() -> Path:
    return _repo_root() / "build" / "varq_fused_flash"


def _extension_candidates() -> list[Path]:
    env_path = os.environ.get(ENV_EXTENSION_PATH)
    roots = [Path(env_path)] if env_path else []
    roots.append(_default_extension_dir())
    candidates: list[Path] = []
    for root in roots:
        if root.is_file():
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(sorted(root.glob(f"{EXTENSION_NAME}*.so")))
    return candidates


def load_extension() -> ModuleType:
    """Load the optional CUDA extension.

    The build script places the shared object under
    `build/varq_fused_flash/` by default.  `VARQ_FUSED_FLASH_EXT_PATH` can point
    either to that directory or directly to a `.so`.
    """

    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    try:
        candidates = _extension_candidates()
        if candidates:
            spec = importlib.util.spec_from_file_location(EXTENSION_NAME, candidates[-1])
            if spec is None or spec.loader is None:
                raise ImportError(f"could not create import spec for {candidates[-1]}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(EXTENSION_NAME)
        _EXTENSION = module
        return module
    except BaseException as exc:  # pragma: no cover - exercised by availability probes
        _LOAD_ERROR = exc
        raise


def availability() -> FusedFlashStatus:
    """Return whether the CUDA backend can currently be imported."""

    try:
        module = load_extension()
    except BaseException as exc:
        return FusedFlashStatus(False, str(exc))
    path = getattr(module, "__file__", None)
    return FusedFlashStatus(True, "loaded", str(path) if path else None)


def fused_flash_dequant_attention(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    step_ids: torch.Tensor,
    k_fresh: torch.Tensor,
    v_fresh: torch.Tensor,
    *,
    direct: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Call the CUDA fused FlashAttention backend.

    Shapes:
    - q, k_fresh, v_fresh: BHLc tensors.  CUDA supports fp16, bf16 and fp32
      for the generic direct path; the legacy FlashAttention bridge keeps its
      fp16/bf16 ABI.
    - k_packed, v_packed: BHLw int32 cached tensors. Q2/Q3/Q4/Q6/Q8 are
      dispatched from the packed width and head dimension.
    - k_scale, v_scale: compact or broadcastable 4-D scales.
    - step_ids: int32 cached-token to scale-id map.
    """

    ext = load_extension()
    entrypoint = "fwd_direct" if direct else "fwd"
    if not hasattr(ext, entrypoint):
        raise RuntimeError(f"{EXTENSION_NAME} is loaded but does not expose {entrypoint}()")
    # The generic CUDA path accepts fp16, bf16, and fp32. The optimized Ampere
    # loader remains specialized for fp16/head_dim=128. Cast any unsupported
    # input dtype to a supported compute dtype and restore it on output.
    orig_dtype = q.dtype
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if orig_dtype not in supported:
        target = torch.float16 if direct else torch.bfloat16
        q = q.to(target)
        k_fresh = k_fresh.to(target)
        v_fresh = v_fresh.to(target)
    # softmax_scale: Infinity uses cos_attn so the module's self.scale is 1, not
    # 1/sqrt(head_dim). Pass it through; None falls back to 1/sqrt(head_dim).
    sm_scale = (1.0 / math.sqrt(q.shape[-1])) if softmax_scale is None else float(softmax_scale)
    out = getattr(ext, entrypoint)(
        q, k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh, sm_scale
    )
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out


def _require_cuda(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor; the CUDA extension is not loaded on CPU")


def _layout_code(layout: str | int) -> int:
    if isinstance(layout, int):
        if layout not in (0, 1):
            raise ValueError("layout must be 0/1 or BLHc/BHLc")
        return int(layout)
    value = str(layout).replace("-", "").replace("_", "").lower()
    if value == "blhc":
        return 0
    if value == "bhlc":
        return 1
    raise ValueError("layout must be BLHc or BHLc")


def _dtype_code(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_TO_CODE[dtype]
    except KeyError as exc:
        raise ValueError("CUDA VAR-Q API supports float16, bfloat16 and float32") from exc


def quantize_pack(
    x: torch.Tensor,
    bits: int,
    *,
    scale: Optional[torch.Tensor] = None,
    group_ids: Optional[torch.Tensor] = None,
    scale_dtype: Optional[torch.dtype] = None,
    layout: str | int = "BLHc",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA symmetric quantization + int32 packing.

    ``x`` is packed along its last dimension.  If ``scale`` is omitted, a
    per-row max-abs scale is computed in CUDA and returned.  A supplied scale
    may be a broadcastable 4-D BLHc/BHLc tensor; ``group_ids`` optionally maps
    the token axis to compact scale groups without materializing the expansion.
    No CUDA extension is loaded merely by importing this module.
    """

    _require_cuda(x, "x")
    bits = int(bits)
    if bits not in SUPPORTED_Q_BITS:
        raise ValueError(f"unsupported VAR-Q bit width {bits}; expected {SUPPORTED_Q_BITS}")
    if scale is not None:
        _require_cuda(scale, "scale")
    if group_ids is not None:
        _require_cuda(group_ids, "group_ids")
        group_ids = group_ids.to(dtype=torch.int32).contiguous()
    if scale_dtype is None:
        scale_dtype = scale.dtype if scale is not None else torch.float32
    scale_code = _dtype_code(scale_dtype)
    ext = load_extension()
    packed, used_scale = ext.quantize_pack(
        x,
        bits,
        scale.contiguous() if scale is not None else None,
        group_ids,
        scale_code,
        _layout_code(layout),
    )
    return packed, used_scale


def pack_int8(q_int8: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed int8 values along the last dimension with a CUDA kernel."""

    _require_cuda(q_int8, "q_int8")
    if q_int8.dtype != torch.int8:
        raise TypeError("q_int8 must have dtype torch.int8")
    bits = int(bits)
    if bits not in SUPPORTED_Q_BITS:
        raise ValueError(f"unsupported VAR-Q bit width {bits}; expected {SUPPORTED_Q_BITS}")
    return load_extension().pack_int8(q_int8, bits)


def unpack_int8(
    packed: torch.Tensor,
    bits: int,
    orig_dim: int = -1,
) -> torch.Tensor:
    """Unpack int32 words into signed int8 values using CUDA."""

    _require_cuda(packed, "packed")
    if packed.dtype != torch.int32:
        raise TypeError("packed must have dtype torch.int32")
    bits = int(bits)
    if bits not in SUPPORTED_Q_BITS:
        raise ValueError(f"unsupported VAR-Q bit width {bits}; expected {SUPPORTED_Q_BITS}")
    return load_extension().unpack_int8(packed, bits, int(orig_dim))


def unpack_dequant(
    packed: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    orig_dim: int = -1,
    *,
    output_dtype: Optional[torch.dtype] = None,
    group_ids: Optional[torch.Tensor] = None,
    layout: str | int = "BLHc",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CUDA unpack + dequant with FP16/BF16/FP32 scale and output support.

    ``out`` may supply a contiguous destination workspace, avoiding an
    extension-side allocation and a subsequent Python ``copy_``.
    """

    _require_cuda(packed, "packed")
    _require_cuda(scale, "scale")
    if packed.dtype != torch.int32:
        raise TypeError("packed must have dtype torch.int32")
    if scale.dtype not in _DTYPE_TO_CODE:
        raise TypeError("scale must have dtype torch.float16/bfloat16/float32")
    if out is not None:
        _require_cuda(out, "out")
        if out.dtype not in _DTYPE_TO_CODE:
            raise TypeError("out must have dtype torch.float16/bfloat16/float32")
        if output_dtype is not None and output_dtype != out.dtype:
            raise ValueError("output_dtype must match out.dtype when out is supplied")
        output_dtype = out.dtype
    elif output_dtype is None:
        output_dtype = scale.dtype
    out_code = _dtype_code(output_dtype)
    if group_ids is not None:
        _require_cuda(group_ids, "group_ids")
        group_ids = group_ids.to(dtype=torch.int32).contiguous()
    bits = int(bits)
    if bits not in SUPPORTED_Q_BITS:
        raise ValueError(f"unsupported VAR-Q bit width {bits}; expected {SUPPORTED_Q_BITS}")
    return load_extension().unpack_dequant(
        packed,
        scale.contiguous(),
        bits,
        int(orig_dim),
        out_code,
        group_ids,
        _layout_code(layout),
        out,
    )


cuda_quantize_pack = quantize_pack
cuda_pack_int8 = pack_int8
cuda_unpack_int8 = unpack_int8
cuda_unpack_dequant = unpack_dequant

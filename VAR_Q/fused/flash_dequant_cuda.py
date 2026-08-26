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
from typing import Optional

import torch


EXTENSION_NAME = "_varq_fused_flash"
ENV_EXTENSION_PATH = "VARQ_FUSED_FLASH_EXT_PATH"


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

    Shapes for v1:
    - q, k_fresh, v_fresh: BHLc fp16/bf16 tensors with head_dim=128 for
      dense bridge; cuda-direct remains fp16-only
    - k_packed, v_packed: BHLw int32 q8-packed cached tensors
    - k_scale, v_scale: BHSd fp16 compact VARQ scales
    - step_ids: int32 cached-token to scale-id map
    """

    ext = load_extension()
    entrypoint = "fwd_direct" if direct else "fwd"
    if not hasattr(ext, entrypoint):
        raise RuntimeError(f"{EXTENSION_NAME} is loaded but does not expose {entrypoint}()")
    # The CUDA kernel accepts fp16 (both paths) or bf16 (dense bridge only).
    # Infinity runs bf16 autocast but q/k/v can arrive as fp32 (e.g. after rope),
    # so cast unsupported dtypes to the kernel's compute dtype and restore on output.
    orig_dtype = q.dtype
    supported = (torch.float16,) if direct else (torch.float16, torch.bfloat16)
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

#!/usr/bin/env python3
"""Build the optional VAR-Q fused FlashAttention CUDA extension.

This is a small Track B build harness around torch.utils.cpp_extension.  It
keeps the extension local to this repo while compiling against a checked-out
flash-attention v2.7.3 source tree for the CUTLASS headers and future kernel
injection work.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


EXTENSION_NAME = "_varq_fused_flash"


K_COPY_PATTERNS = (
    (
        """flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);""",
        """flash::varq_copy_k_tile<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), params, tKsK, tKVcKV, tKVpKV, bidb, bidh, n_block,
        varq_packed_smem, binfo.actual_seqlen_k - n_block * kBlockN);""",
    ),
    (
        """flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);""",
        """flash::varq_copy_k_tile</*Is_even_MN=*/true, Is_even_K>(
            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), params, tKsK, tKVcKV, tKVpKV, bidb, bidh, n_block - 1,
            varq_packed_smem, 0);""",
    ),
)

V_COPY_PATTERNS = (
    (
        """flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);""",
        """flash::varq_copy_v_tile</*Is_even_MN=*/true, Is_even_K>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), params, tVsV, tKVcKV, tKVpKV, bidb, bidh, n_block,
            varq_packed_smem, 0);""",
    ),
    (
        """flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );""",
        """flash::varq_copy_v_tile<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), params, tVsV, tKVcKV, tKVpKV, bidb, bidh, n_block,
                varq_packed_smem, binfo.actual_seqlen_k - n_block * kBlockN
            );""",
    ),
)


FLASH_KERNEL_PIPELINE_PATTERNS = (
    (
        """    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});""",
        """    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    int32_t *varq_packed_smem = reinterpret_cast<int32_t *>(smem_ + Kernel_traits::kSmemSize);""",
    ),
    (
        """        flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV""",
        """        flash::cp_async_wait<0>();
        __syncthreads();
        bool varq_finalized_k = false;
        if (masking_step == 0) {
            varq_finalized_k = flash::varq_finalize_k_tile<Is_even_MN, Is_even_K>(
                params, tKsK, tKVcKV, tKVpKV, varq_packed_smem, bidb, bidh, n_block,
                binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            varq_finalized_k = flash::varq_finalize_k_tile</*Is_even_MN=*/true, Is_even_K>(
                params, tKsK, tKVcKV, tKVpKV, varq_packed_smem, bidb, bidh, n_block, 0);
        }
        if (varq_finalized_k) { __syncthreads(); }

        // Advance gV""",
    ),
    (
        """        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();""",
        """        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        bool varq_finalized_v = false;
        if (masking_step > 0) {
            varq_finalized_v = flash::varq_finalize_v_tile</*Is_even_MN=*/true, Is_even_K>(
                params, tVsV, tKVcKV, tKVpKV, varq_packed_smem, bidb, bidh, n_block, 0);
        } else {
            varq_finalized_v = flash::varq_finalize_v_tile<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                params, tVsV, tKVcKV, tKVpKV, varq_packed_smem, bidb, bidh, n_block,
                binfo.actual_seqlen_k - n_block * kBlockN);
        }
        if (varq_finalized_v) { __syncthreads(); }""",
    ),
    (
        """        flash::cp_async_wait<0>();
        __syncthreads();
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);""",
        """        flash::cp_async_wait<0>();
        __syncthreads();
        if (flash::varq_finalize_k_tile</*Is_even_MN=*/true, Is_even_K>(
                params, tKsK, tKVcKV, tKVpKV, varq_packed_smem, bidb, bidh, n_block, 0)) {
            __syncthreads();
        }
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);""",
    ),
    (
        """        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(""",
        """        flash::cp_async_wait<0>();
        __syncthreads();
        if (flash::varq_finalize_v_tile</*Is_even_MN=*/true, Is_even_K>(
                params, tVsV, tKVcKV, tKVpKV, varq_packed_smem, bidb, bidh, n_block, 0)) {
            __syncthreads();
        }
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(""",
    ),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def validate_flash_attn_source(path: Path) -> None:
    required = (
        path / "csrc" / "flash_attn" / "src" / "flash_fwd_kernel.h",
        path / "csrc" / "flash_attn" / "src" / "flash.h",
        path / "csrc" / "cutlass" / "include" / "cutlass" / "cutlass.h",
    )
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "flash-attention v2.7.3 source is incomplete; missing: "
            + ", ".join(missing)
        )


def generate_varq_flash_sources(flash_attn_src: Path, build_dir: Path) -> Path:
    """Generate a small patched FlashAttention header for VAR-Q direct K/V loads."""

    generated_dir = build_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    src = flash_attn_src / "csrc" / "flash_attn" / "src" / "flash_fwd_kernel.h"
    text = src.read_text()
    for idx, (old, new) in enumerate(FLASH_KERNEL_PIPELINE_PATTERNS):
        if old not in text:
            raise RuntimeError(f"VAR-Q flash pipeline patch pattern not found:\n{old}")
        # The first three snippets also occur in split-kv. The direct VAR-Q backend
        # only patches compute_attn_1rowblock.
        replace_count = 1 if idx < 3 else -1
        text = text.replace(old, new, replace_count)
    for old, new in (*K_COPY_PATTERNS, *V_COPY_PATTERNS):
        if old not in text:
            raise RuntimeError(f"VAR-Q flash patch pattern not found:\n{old}")
        text = text.replace(old, new)
    (generated_dir / "varq_flash_fwd_kernel.h").write_text(text)
    return generated_dir


def build_extension(flash_attn_src: Path, build_dir: Path, verbose: bool) -> object:
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not set; nvcc is required for Track B")
    validate_flash_attn_source(flash_attn_src)
    build_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = generate_varq_flash_sources(flash_attn_src, build_dir)

    src_dir = repo_root() / "VAR_Q" / "csrc" / "fused_flash"
    sources = [
        src_dir / "varq_fused_flash.cpp",
        src_dir / "varq_fused_flash_cuda.cu",
        src_dir / "varq_flash_fwd_hdim128_fp16_sm80.cu",
        flash_attn_src / "csrc" / "flash_attn" / "src" / "flash_fwd_hdim128_fp16_sm80.cu",
        flash_attn_src / "csrc" / "flash_attn" / "src" / "flash_fwd_hdim128_bf16_sm80.cu",
    ]
    include_paths = [
        generated_dir,
        src_dir,
        flash_attn_src / "csrc" / "flash_attn",
        flash_attn_src / "csrc" / "flash_attn" / "src",
        flash_attn_src / "csrc" / "cutlass" / "include",
    ]

    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("VARQ_CUDA_ARCH_LIST", "8.0")
    os.environ.setdefault("MAX_JOBS", "2")

    return load(
        name=EXTENSION_NAME,
        sources=[str(p) for p in sources],
        build_directory=str(build_dir),
        extra_include_paths=[str(p) for p in include_paths],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
        ],
        with_cuda=True,
        verbose=verbose,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flash-attn-src",
        default=os.environ.get(
            "FLASH_ATTN_SOURCE",
            str(repo_root() / "third_party" / "flash-attention"),
        ),
        help="flash-attention v2.7.3 source checkout with csrc/cutlass initialized",
    )
    parser.add_argument(
        "--build-dir",
        default=str(repo_root() / "build" / "varq_fused_flash"),
        help="directory where torch cpp_extension writes the extension .so",
    )
    parser.add_argument("--quiet", action="store_true", help="disable verbose ninja output")
    args = parser.parse_args()

    module = build_extension(
        flash_attn_src=Path(args.flash_attn_src).resolve(),
        build_dir=Path(args.build_dir).resolve(),
        verbose=not args.quiet,
    )
    print("extension", getattr(module, "__file__", "<unknown>"))
    if hasattr(module, "backend_info"):
        print("backend_info", module.backend_info())
    print("torch", torch.__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

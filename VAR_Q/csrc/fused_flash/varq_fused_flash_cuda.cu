#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <cmath>
#include <initializer_list>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "varq_flash.h"

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_CONTIGUOUS_LAST(x) TORCH_CHECK((x).stride(-1) == 1, #x " must have contiguous last dimension")
#define CHECK_DTYPE(x, dtype) TORCH_CHECK((x).scalar_type() == (dtype), #x " has unexpected dtype")

struct PackedFormat {
    int bits;
    int vals_per_word;
    int packed_words;
};

// The generated FlashAttention bridge remains a head_dim=128 specialization;
// generic CUDA kernels below dispatch both 64 and 128 at runtime.
constexpr int kHeadDim = 128;

int vals_per_word_for_bits(int bits) {
    TORCH_CHECK(bits >= 2 && bits <= 8, "VAR-Q bits must be in [2, 8]");
    return bits == 3 ? 10 : 32 / bits;
}

PackedFormat packed_format_from_bits(int bits, int head_dim) {
    const int vals = vals_per_word_for_bits(bits);
    return PackedFormat{bits, vals, (head_dim + vals - 1) / vals};
}

PackedFormat packed_format_from_words(int packed_words, int head_dim) {
    // Width alone is ambiguous for (D=64,q4) and (D=128,q2), so callers must
    // pass head_dim and the dispatch resolves the format from both values.
    for (const int bits : {8, 6, 4, 3, 2}) {
        const PackedFormat format = packed_format_from_bits(bits, head_dim);
        if (format.packed_words == packed_words) {
            return format;
        }
    }
    TORCH_CHECK(
        false,
        "unsupported packed width=", packed_words,
        " for head_dim=", head_dim,
        "; expected q2/q3/q4/q6/q8 width");
    return PackedFormat{0, 0, 0};
}

void check_rank4(const at::Tensor &x, const char *name) {
    TORCH_CHECK(x.dim() == 4, name, " must be rank-4");
}

void check_bhlc_supported(const at::Tensor &x, const char *name, bool allow_fp32) {
    CHECK_CUDA(x);
    check_rank4(x, name);
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16
            || (allow_fp32 && x.scalar_type() == at::kFloat),
        name, " must be fp16, bf16", allow_fp32 ? " or fp32" : "");
    CHECK_CONTIGUOUS_LAST(x);
    TORCH_CHECK(x.size(3) == 64 || x.size(3) == 128, name, " must have head_dim=64 or 128");
}

void check_same_bhlc_shape(const at::Tensor &lhs, const at::Tensor &rhs, const char *rhs_name) {
    TORCH_CHECK(
        lhs.sizes() == rhs.sizes(),
        rhs_name,
        " must match q shape for Track B v1; got ",
        rhs.sizes(),
        " vs ",
        lhs.sizes());
}

int round_multiple(int x, int m) {
    return ((x + m - 1) / m) * m;
}

template <typename OutT, typename ScaleT>
__global__ void dequant_to_blhc_kernel(
    const int32_t *__restrict__ packed,
    const ScaleT *__restrict__ scale,
    const int32_t *__restrict__ step_ids,
    OutT *__restrict__ dense,
    int B,
    int H,
    int N,
    int S,
    int bits,
    int vals_per_word,
    int packed_words,
    int N_total,
    int head_dim,
    int dst_offset) {
    const int64_t total = static_cast<int64_t>(B) * H * N * head_dim;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int d = idx % head_dim;
        int64_t rem = idx / head_dim;
        const int n = rem % N;
        rem /= N;
        const int h = rem % H;
        const int b = rem / H;

        const int word_idx = d / vals_per_word;
        const int lane = d - word_idx * vals_per_word;
        const uint32_t word = static_cast<uint32_t>(
            packed[(((b * H + h) * N + n) * packed_words) + word_idx]);
        const uint32_t mask = (1u << bits) - 1u;
        const uint32_t unsigned_piece = (word >> (lane * bits)) & mask;
        const uint32_t sign_bit = 1u << (bits - 1);
        const int32_t signed_piece = (unsigned_piece & sign_bit)
            ? static_cast<int32_t>(unsigned_piece) - (1 << bits)
            : static_cast<int32_t>(unsigned_piece);
        const int step = step_ids[n];
        const float scale_value = static_cast<float>(scale[(((b * H + h) * S + step) * head_dim) + d]);
        dense[(((b * N_total + (dst_offset + n)) * H + h) * head_dim) + d] =
            static_cast<OutT>(static_cast<float>(signed_piece) * scale_value);
    }
}

template <typename T>
__global__ void copy_bhlc_to_blhc_kernel(
    const T *__restrict__ src,
    T *__restrict__ dst,
    int B,
    int H,
    int N,
    int N_total,
    int head_dim,
    int dst_offset) {
    const int64_t total = static_cast<int64_t>(B) * H * N * head_dim;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int d = idx % head_dim;
        int64_t rem = idx / head_dim;
        const int n = rem % N;
        rem /= N;
        const int h = rem % H;
        const int b = rem / H;
        dst[(((b * N_total + (dst_offset + n)) * H + h) * head_dim) + d] =
            src[(((b * H + h) * N + n) * head_dim) + d];
    }
}

void launch_dequant_to_blhc(
    const at::Tensor &packed,
    const at::Tensor &scale,
    const at::Tensor &step_ids,
    at::Tensor &dense,
    int dst_offset,
    cudaStream_t stream,
    int head_dim) {
    const int B = packed.size(0);
    const int H = packed.size(1);
    const int N = packed.size(2);
    const int S = scale.size(2);
    const PackedFormat format = packed_format_from_words(packed.size(3), head_dim);
    const int N_total = dense.size(1);
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(B) * H * N * head_dim;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    if (scale.scalar_type() == at::kHalf && dense.scalar_type() == at::kHalf) {
        dequant_to_blhc_kernel<at::Half, at::Half><<<blocks, threads, 0, stream>>>(
            packed.data_ptr<int32_t>(), scale.data_ptr<at::Half>(), step_ids.data_ptr<int32_t>(),
            dense.data_ptr<at::Half>(), B, H, N, S, format.bits, format.vals_per_word,
            format.packed_words, N_total, head_dim, dst_offset);
    } else if (scale.scalar_type() == at::kBFloat16 && dense.scalar_type() == at::kBFloat16) {
        dequant_to_blhc_kernel<at::BFloat16, at::BFloat16><<<blocks, threads, 0, stream>>>(
            packed.data_ptr<int32_t>(), scale.data_ptr<at::BFloat16>(), step_ids.data_ptr<int32_t>(),
            dense.data_ptr<at::BFloat16>(), B, H, N, S, format.bits, format.vals_per_word,
            format.packed_words, N_total, head_dim, dst_offset);
    } else if (scale.scalar_type() == at::kFloat && dense.scalar_type() == at::kFloat) {
        dequant_to_blhc_kernel<float, float><<<blocks, threads, 0, stream>>>(
            packed.data_ptr<int32_t>(), scale.data_ptr<float>(), step_ids.data_ptr<int32_t>(),
            dense.data_ptr<float>(), B, H, N, S, format.bits, format.vals_per_word,
            format.packed_words, N_total, head_dim, dst_offset);
    } else if (scale.scalar_type() == at::kHalf && dense.scalar_type() == at::kBFloat16) {
        dequant_to_blhc_kernel<at::BFloat16, at::Half><<<blocks, threads, 0, stream>>>(
            packed.data_ptr<int32_t>(), scale.data_ptr<at::Half>(), step_ids.data_ptr<int32_t>(),
            dense.data_ptr<at::BFloat16>(), B, H, N, S, format.bits, format.vals_per_word,
            format.packed_words, N_total, head_dim, dst_offset);
    } else if (scale.scalar_type() == at::kHalf && dense.scalar_type() == at::kFloat) {
        dequant_to_blhc_kernel<float, at::Half><<<blocks, threads, 0, stream>>>(
            packed.data_ptr<int32_t>(), scale.data_ptr<at::Half>(), step_ids.data_ptr<int32_t>(),
            dense.data_ptr<float>(), B, H, N, S, format.bits, format.vals_per_word,
            format.packed_words, N_total, head_dim, dst_offset);
    } else if (scale.scalar_type() == at::kBFloat16 && dense.scalar_type() == at::kFloat) {
        dequant_to_blhc_kernel<float, at::BFloat16><<<blocks, threads, 0, stream>>>(
            packed.data_ptr<int32_t>(), scale.data_ptr<at::BFloat16>(), step_ids.data_ptr<int32_t>(),
            dense.data_ptr<float>(), B, H, N, S, format.bits, format.vals_per_word,
            format.packed_words, N_total, head_dim, dst_offset);
    } else {
        TORCH_CHECK(false, "unsupported scale/output dtype combination for CUDA dequant");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_copy_bhlc_to_blhc(
    const at::Tensor &src,
    at::Tensor &dst,
    int dst_offset,
    cudaStream_t stream,
    int head_dim) {
    const int B = src.size(0);
    const int H = src.size(1);
    const int N = src.size(2);
    const int N_total = dst.size(1);
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(B) * H * N * head_dim;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "fresh and dense dtype must match");
    if (src.scalar_type() == at::kHalf) {
        copy_bhlc_to_blhc_kernel<at::Half><<<blocks, threads, 0, stream>>>(
            src.data_ptr<at::Half>(), dst.data_ptr<at::Half>(), B, H, N, N_total, head_dim, dst_offset);
    } else if (src.scalar_type() == at::kBFloat16) {
        copy_bhlc_to_blhc_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
            src.data_ptr<at::BFloat16>(), dst.data_ptr<at::BFloat16>(), B, H, N, N_total, head_dim, dst_offset);
    } else if (src.scalar_type() == at::kFloat) {
        copy_bhlc_to_blhc_kernel<float><<<blocks, threads, 0, stream>>>(
            src.data_ptr<float>(), dst.data_ptr<float>(), B, H, N, N_total, head_dim, dst_offset);
    } else {
        TORCH_CHECK(false, "unsupported fresh/dense dtype for CUDA copy");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void set_dense_fwd_params(
    Flash_fwd_params &params,
    const at::Tensor &q_blhc,
    const at::Tensor &k_blhc,
    const at::Tensor &v_blhc,
    at::Tensor &out_blhc,
    at::Tensor &softmax_lse) {
    params = {};
    const int B = q_blhc.size(0);
    const int seqlen_q = q_blhc.size(1);
    const int H = q_blhc.size(2);
    const int seqlen_k = k_blhc.size(1);
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    params.q_ptr = q_blhc.data_ptr();
    params.k_ptr = k_blhc.data_ptr();
    params.v_ptr = v_blhc.data_ptr();
    params.o_ptr = out_blhc.data_ptr();

    params.q_batch_stride = q_blhc.stride(0);
    params.k_batch_stride = k_blhc.stride(0);
    params.v_batch_stride = v_blhc.stride(0);
    params.o_batch_stride = out_blhc.stride(0);
    params.q_row_stride = q_blhc.stride(1);
    params.k_row_stride = k_blhc.stride(1);
    params.v_row_stride = v_blhc.stride(1);
    params.o_row_stride = out_blhc.stride(1);
    params.q_head_stride = q_blhc.stride(2);
    params.k_head_stride = k_blhc.stride(2);
    params.v_head_stride = v_blhc.stride(2);
    params.o_head_stride = out_blhc.stride(2);

    params.softmax_lse_ptr = softmax_lse.data_ptr();
    params.b = B;
    params.h = H;
    params.h_k = H;
    params.h_h_k_ratio = 1;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = round_multiple(seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(seqlen_k, 128);
    params.d = kHeadDim;
    params.d_rounded = kHeadDim;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * 1.4426950408889634f;
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.is_seqlens_k_cumulative = true;
    params.is_bf16 = q_blhc.scalar_type() == at::kBFloat16;
    params.is_causal = false;
}

void set_varq_fwd_params(
    Varq_fwd_params &params,
    const at::Tensor &q_blhc,
    at::Tensor &out,
    at::Tensor &softmax_lse,
    const at::Tensor &k_packed,
    const at::Tensor &v_packed,
    const at::Tensor &k_scale,
    const at::Tensor &v_scale,
    const at::Tensor &step_ids,
    const at::Tensor &k_fresh,
    const at::Tensor &v_fresh,
    bool output_bhlc = false) {
    params = {};
    const int B = q_blhc.size(0);
    const int seqlen_q = q_blhc.size(1);
    const int H = q_blhc.size(2);
    const int seqlen_cached = k_packed.size(2);
    const int seqlen_fresh = k_fresh.size(2);
    const int seqlen_k = seqlen_cached + seqlen_fresh;
    const PackedFormat format = packed_format_from_words(k_packed.size(3), q_blhc.size(3));
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(q_blhc.size(3)));

    params.q_ptr = q_blhc.data_ptr();
    params.k_ptr = k_fresh.data_ptr<at::Half>() - static_cast<int64_t>(seqlen_cached) * k_fresh.stride(2);
    params.v_ptr = v_fresh.data_ptr<at::Half>() - static_cast<int64_t>(seqlen_cached) * v_fresh.stride(2);
    params.o_ptr = out.data_ptr();

    params.q_batch_stride = q_blhc.stride(0);
    params.k_batch_stride = k_fresh.stride(0);
    params.v_batch_stride = v_fresh.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q_blhc.stride(1);
    params.k_row_stride = k_fresh.stride(2);
    params.v_row_stride = v_fresh.stride(2);
    params.o_row_stride = output_bhlc ? out.stride(2) : out.stride(1);
    params.q_head_stride = q_blhc.stride(2);
    params.k_head_stride = k_fresh.stride(1);
    params.v_head_stride = v_fresh.stride(1);
    params.o_head_stride = output_bhlc ? out.stride(1) : out.stride(2);

    params.softmax_lse_ptr = softmax_lse.data_ptr();
    params.b = B;
    params.h = H;
    params.h_k = H;
    params.h_h_k_ratio = 1;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_knew = seqlen_fresh;
    params.seqlen_q_rounded = round_multiple(seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(seqlen_k, 128);
    params.d = q_blhc.size(3);
    params.d_rounded = q_blhc.size(3);
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * 1.4426950408889634f;
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.is_seqlens_k_cumulative = true;
    params.is_bf16 = false;
    params.is_causal = false;

    params.varq_k_packed_ptr = k_packed.data_ptr();
    params.varq_v_packed_ptr = v_packed.data_ptr();
    params.varq_k_scale_ptr = k_scale.data_ptr();
    params.varq_v_scale_ptr = v_scale.data_ptr();
    params.varq_step_ids_ptr = step_ids.data_ptr();
    params.varq_k_fresh_ptr = k_fresh.data_ptr();
    params.varq_v_fresh_ptr = v_fresh.data_ptr();
    params.varq_cached_len = seqlen_cached;
    params.varq_num_steps = k_scale.size(2);
    params.varq_block_n = 0;
    params.varq_q_bits = format.bits;
    params.varq_vals_per_word = format.vals_per_word;
    params.varq_packed_words = format.packed_words;
}

void check_float_dtype(const at::Tensor &x, const char *name);

void validate_inputs(
    const at::Tensor &q,
    const at::Tensor &k_packed,
    const at::Tensor &v_packed,
    const at::Tensor &k_scale,
    const at::Tensor &v_scale,
    const at::Tensor &step_ids,
    const at::Tensor &k_fresh,
    const at::Tensor &v_fresh,
    bool allow_empty_fresh) {
    check_bhlc_supported(q, "q", /*allow_fp32=*/true);
    check_bhlc_supported(k_fresh, "k_fresh", /*allow_fp32=*/true);
    check_bhlc_supported(v_fresh, "v_fresh", /*allow_fp32=*/true);
    if (allow_empty_fresh && k_fresh.size(2) == 0 && v_fresh.size(2) == 0) {
        TORCH_CHECK(k_fresh.size(0) == q.size(0), "empty k_fresh batch must match q");
        TORCH_CHECK(v_fresh.size(0) == q.size(0), "empty v_fresh batch must match q");
        TORCH_CHECK(k_fresh.size(1) == q.size(1), "empty k_fresh heads must match q");
        TORCH_CHECK(v_fresh.size(1) == q.size(1), "empty v_fresh heads must match q");
        TORCH_CHECK(k_fresh.size(3) == q.size(3), "empty k_fresh head_dim must match q");
        TORCH_CHECK(v_fresh.size(3) == q.size(3), "empty v_fresh head_dim must match q");
    } else {
        check_same_bhlc_shape(q, k_fresh, "k_fresh");
        check_same_bhlc_shape(q, v_fresh, "v_fresh");
    }

    CHECK_CUDA(k_packed);
    CHECK_CUDA(v_packed);
    CHECK_CUDA(k_scale);
    CHECK_CUDA(v_scale);
    CHECK_CUDA(step_ids);
    check_rank4(k_packed, "k_packed");
    check_rank4(v_packed, "v_packed");
    check_rank4(k_scale, "k_scale");
    check_rank4(v_scale, "v_scale");
    TORCH_CHECK(step_ids.dim() == 1, "step_ids must be rank-1");
    CHECK_DTYPE(k_packed, at::kInt);
    CHECK_DTYPE(v_packed, at::kInt);
    check_float_dtype(k_scale, "k_scale");
    check_float_dtype(v_scale, "v_scale");
    TORCH_CHECK(k_scale.scalar_type() == v_scale.scalar_type(),
                "K/V scale dtypes must match");
    TORCH_CHECK(k_fresh.scalar_type() == q.scalar_type(), "k_fresh dtype must match q");
    TORCH_CHECK(v_fresh.scalar_type() == q.scalar_type(), "v_fresh dtype must match q");
    CHECK_DTYPE(step_ids, at::kInt);
    CHECK_CONTIGUOUS(k_packed);
    CHECK_CONTIGUOUS(v_packed);
    CHECK_CONTIGUOUS(k_scale);
    CHECK_CONTIGUOUS(v_scale);
    CHECK_CONTIGUOUS(step_ids);
    TORCH_CHECK(k_packed.sizes() == v_packed.sizes(), "k_packed and v_packed shapes must match");
    const PackedFormat format = packed_format_from_words(k_packed.size(3), q.size(3));
    TORCH_CHECK(
        format.packed_words == k_packed.size(3),
        "packed width does not match the inferred VAR-Q format");
    TORCH_CHECK(k_packed.size(0) == q.size(0), "packed cache batch must match q");
    TORCH_CHECK(k_packed.size(1) == q.size(1), "packed cache heads must match q");
    TORCH_CHECK(k_scale.size(0) == q.size(0), "k_scale batch must match q");
    TORCH_CHECK(v_scale.size(0) == q.size(0), "v_scale batch must match q");
    TORCH_CHECK(k_scale.size(1) == q.size(1), "k_scale heads must match q");
    TORCH_CHECK(v_scale.size(1) == q.size(1), "v_scale heads must match q");
    TORCH_CHECK(k_scale.size(3) == q.size(3), "k_scale head_dim must match q");
    TORCH_CHECK(v_scale.size(3) == q.size(3), "v_scale head_dim must match q");
    TORCH_CHECK(k_scale.size(2) == v_scale.size(2), "K/V scale step counts must match");
    TORCH_CHECK(step_ids.size(0) == k_packed.size(2), "step_ids length must match cached token count");
}

// ---------------------------------------------------------------------------
// Standalone CUDA quantization/packing ABI
// ---------------------------------------------------------------------------

struct ScaleMeta {
    int64_t sizes[4];
    int64_t strides[4];
    int token_dim;
    int use_group_ids;
    int group_rank;
    int64_t group_stride_b;
    int64_t group_stride_l;
};

ScaleMeta make_scale_meta(
    const at::Tensor &scale,
    const at::Tensor &x,
    const at::Tensor *group_ids,
    int layout) {
    TORCH_CHECK(scale.dim() >= 1 && scale.dim() <= 4, "scale must have rank 1..4");
    TORCH_CHECK(x.dim() == 4, "CUDA quant/dequant broadcast API expects rank-4 tensors");
    ScaleMeta meta{};
    const int offset = 4 - scale.dim();
    for (int i = 0; i < 4; ++i) {
        if (i < offset) {
            meta.sizes[i] = 1;
            meta.strides[i] = 0;
            continue;
        }
        const int si = i - offset;
        meta.sizes[i] = scale.size(si);
        meta.strides[i] = scale.stride(si);
        TORCH_CHECK(
            meta.sizes[i] == 1 || meta.sizes[i] == x.size(i)
                || (group_ids != nullptr && i == (layout == 0 ? 1 : 2)),
            "scale is not broadcastable to x at dimension ", i,
            "; scale=", scale.sizes(), " x=", x.sizes());
    }
    meta.token_dim = layout == 0 ? 1 : 2;
    meta.use_group_ids = group_ids != nullptr;
    meta.group_rank = 0;
    meta.group_stride_b = 0;
    meta.group_stride_l = 0;
    if (group_ids != nullptr) {
        TORCH_CHECK(group_ids->dim() == 1 || group_ids->dim() == 2,
                    "group_ids must be [L] or [B,L]");
        const int64_t tokens = x.size(meta.token_dim);
        TORCH_CHECK(group_ids->size(-1) == tokens,
                    "group_ids token length must match x");
        if (group_ids->dim() == 2) {
            TORCH_CHECK(group_ids->size(0) == x.size(0),
                        "rank-2 group_ids batch dimension must match x");
        }
        meta.group_rank = group_ids->dim();
        meta.group_stride_l = group_ids->stride(-1);
        meta.group_stride_b = group_ids->dim() == 2 ? group_ids->stride(0) : 0;
        TORCH_CHECK(meta.sizes[meta.token_dim] > 0,
                    "compact scale token/group dimension must be non-empty");
    }
    return meta;
}

__device__ __forceinline__ int64_t group_id_at(
    const int32_t *group_ids,
    const ScaleMeta &meta,
    int64_t b,
    int64_t l) {
    if (!meta.use_group_ids || group_ids == nullptr) {
        return l;
    }
    const int64_t offset = meta.group_rank == 2
        ? b * meta.group_stride_b + l * meta.group_stride_l
        : l * meta.group_stride_l;
    return static_cast<int64_t>(group_ids[offset]);
}

__device__ __forceinline__ int64_t scale_offset_4d(
    const ScaleMeta &meta,
    const int64_t c0,
    const int64_t c1,
    const int64_t c2,
    const int64_t c3,
    const int32_t *group_ids) {
    int64_t coords[4] = {c0, c1, c2, c3};
    const int64_t token = meta.token_dim == 1 ? c1 : c2;
    coords[meta.token_dim] = group_id_at(group_ids, meta, c0, token);
    int64_t offset = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int64_t coord = meta.sizes[i] == 1 ? 0 : coords[i];
        offset += coord * meta.strides[i];
    }
    return offset;
}

template <typename T>
__global__ void compute_row_scale_kernel(
    const T *__restrict__ x,
    void *__restrict__ scale_out,
    int64_t rows,
    int64_t dim,
    int scale_code,
    int bits) {
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ float partial[256];
    float local = 0.0f;
    for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
        local = fmaxf(local, fabsf(static_cast<float>(x[row * dim + d])));
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float qmax = static_cast<float>((1 << (bits - 1)) - 1);
        const float value = fmaxf(partial[0] / qmax, 1.0e-12f);
        if (scale_code == 0) {
            reinterpret_cast<at::Half *>(scale_out)[row] = static_cast<at::Half>(value);
        } else if (scale_code == 1) {
            reinterpret_cast<at::BFloat16 *>(scale_out)[row] = static_cast<at::BFloat16>(value);
        } else {
            reinterpret_cast<float *>(scale_out)[row] = value;
        }
    }
}

template <typename T, typename ScaleT>
__global__ void quantize_pack_row_scale_kernel(
    const T *__restrict__ x,
    const ScaleT *__restrict__ scale,
    int32_t *__restrict__ packed,
    int64_t rows,
    int64_t dim,
    int64_t packed_words,
    int bits,
    int vals_per_word) {
    const int64_t word = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_words = rows * packed_words;
    if (word >= total_words) {
        return;
    }
    const int64_t row = word / packed_words;
    const int word_idx = static_cast<int>(word - row * packed_words);
    const int64_t base = row * dim;
    const int64_t d0 = static_cast<int64_t>(word_idx) * vals_per_word;
    uint32_t word_value = 0;
    const int32_t qmin = -(1 << (bits - 1));
    const int32_t qmax = (1 << (bits - 1)) - 1;
    for (int lane = 0; lane < vals_per_word; ++lane) {
        const int64_t d = d0 + lane;
        if (d >= dim) {
            break;
        }
        float s = static_cast<float>(scale[row]);
        if (!(s > 0.0f) || !isfinite(s)) {
            s = 1.0e-12f;
        }
        float value = rintf(static_cast<float>(x[base + d]) / s);
        value = fminf(fmaxf(value, static_cast<float>(qmin)), static_cast<float>(qmax));
        const uint32_t encoded = static_cast<uint32_t>(static_cast<int32_t>(value)) & ((1u << bits) - 1u);
        word_value |= encoded << (lane * bits);
    }
    packed[word] = static_cast<int32_t>(word_value);
}

// Explicit 4-D row decoder used by the supplied-scale quantizer.  Keeping it
// separate avoids any dependence on tensor strides for the input, which is
// normalized contiguous by the C++ entrypoint.
__device__ __forceinline__ void decode_row_4d(
    int64_t row,
    int64_t s1,
    int64_t s2,
    int64_t &b,
    int64_t &a1,
    int64_t &a2) {
    a2 = row % s2;
    row /= s2;
    a1 = row % s1;
    b = row / s1;
}

template <typename T, typename ScaleT>
__global__ void quantize_pack_broadcast_kernel(
    const T *__restrict__ x,
    const ScaleT *__restrict__ scale,
    const int32_t *__restrict__ group_ids,
    int32_t *__restrict__ packed,
    int64_t s1,
    int64_t s2,
    int64_t batches,
    int64_t dim,
    int64_t packed_words,
    int bits,
    int vals_per_word,
    ScaleMeta scale_meta) {
    const int64_t word = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t rows = batches * s1 * s2;
    if (word >= rows * packed_words) {
        return;
    }
    const int64_t row = word / packed_words;
    const int word_idx = static_cast<int>(word - row * packed_words);
    int64_t b, a1, a2;
    decode_row_4d(row, s1, s2, b, a1, a2);
    const int64_t base = row * dim;
    const int32_t qmin = -(1 << (bits - 1));
    const int32_t qmax = (1 << (bits - 1)) - 1;
    uint32_t word_value = 0;
    for (int lane = 0; lane < vals_per_word; ++lane) {
        const int64_t d = static_cast<int64_t>(word_idx) * vals_per_word + lane;
        if (d >= dim) break;
        const int64_t scale_idx = scale_offset_4d(scale_meta, b, a1, a2, d, group_ids);
        float s = static_cast<float>(scale[scale_idx]);
        if (!(s > 0.0f) || !isfinite(s)) s = 1.0e-12f;
        float value = rintf(static_cast<float>(x[base + d]) / s);
        value = fminf(fmaxf(value, static_cast<float>(qmin)), static_cast<float>(qmax));
        const uint32_t encoded = static_cast<uint32_t>(static_cast<int32_t>(value)) & ((1u << bits) - 1u);
        word_value |= encoded << (lane * bits);
    }
    packed[word] = static_cast<int32_t>(word_value);
}

template <typename T>
__global__ void pack_int8_kernel(
    const T *__restrict__ q,
    int32_t *__restrict__ packed,
    int64_t rows,
    int64_t dim,
    int64_t words,
    int bits,
    int vals_per_word) {
    const int64_t word = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (word >= rows * words) return;
    const int64_t row = word / words;
    const int word_idx = static_cast<int>(word - row * words);
    uint32_t value = 0;
    const uint32_t mask = (1u << bits) - 1u;
    for (int lane = 0; lane < vals_per_word; ++lane) {
        const int64_t d = static_cast<int64_t>(word_idx) * vals_per_word + lane;
        if (d >= dim) break;
        value |= (static_cast<uint32_t>(static_cast<int32_t>(q[row * dim + d])) & mask)
            << (lane * bits);
    }
    packed[word] = static_cast<int32_t>(value);
}

template <typename T>
__global__ void unpack_int8_kernel(
    const int32_t *__restrict__ packed,
    T *__restrict__ out,
    int64_t rows,
    int64_t dim,
    int64_t words,
    int bits,
    int vals_per_word) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * dim) return;
    const int64_t row = idx / dim;
    const int d = static_cast<int>(idx - row * dim);
    const int word_idx = d / vals_per_word;
    const int lane = d - word_idx * vals_per_word;
    const uint32_t word = static_cast<uint32_t>(packed[row * words + word_idx]);
    const uint32_t piece = (word >> (lane * bits)) & ((1u << bits) - 1u);
    const int32_t sign = 1 << (bits - 1);
    const int32_t signed_piece = (piece & sign) ? static_cast<int32_t>(piece) - (1 << bits)
                                                 : static_cast<int32_t>(piece);
    out[idx] = static_cast<T>(signed_piece);
}

template <typename OutT, typename ScaleT>
__global__ void unpack_dequant_kernel(
    const int32_t *__restrict__ packed,
    const ScaleT *__restrict__ scale,
    const int32_t *__restrict__ group_ids,
    OutT *__restrict__ out,
    int64_t batches,
    int64_t s1,
    int64_t s2,
    int64_t dim,
    int64_t words,
    int bits,
    int vals_per_word,
    ScaleMeta scale_meta) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t rows = batches * s1 * s2;
    if (idx >= rows * dim) return;
    const int64_t row = idx / dim;
    const int d = static_cast<int>(idx - row * dim);
    int64_t b, a1, a2;
    decode_row_4d(row, s1, s2, b, a1, a2);
    const int word_idx = d / vals_per_word;
    const int lane = d - word_idx * vals_per_word;
    const uint32_t word = static_cast<uint32_t>(packed[row * words + word_idx]);
    const uint32_t piece = (word >> (lane * bits)) & ((1u << bits) - 1u);
    const int32_t sign = 1 << (bits - 1);
    const int32_t signed_piece = (piece & sign)
        ? static_cast<int32_t>(piece) - (1 << bits)
        : static_cast<int32_t>(piece);
    const int64_t scale_idx = scale_offset_4d(scale_meta, b, a1, a2, d, group_ids);
    out[idx] = static_cast<OutT>(static_cast<float>(signed_piece)
                                 * static_cast<float>(scale[scale_idx]));
}

template <typename T, typename ScaleT, int HeadDim>
__device__ __forceinline__ float load_packed_or_fresh(
    const int32_t *__restrict__ packed,
    const ScaleT *__restrict__ scale,
    const int32_t *__restrict__ step_ids,
    const T *__restrict__ fresh,
    int b,
    int h,
    int n,
    int d,
    int heads,
    int cached_len,
    int fresh_len,
    int num_steps,
    int bits,
    int vals_per_word,
    int packed_words) {
    if (n >= cached_len) {
        const int fresh_n = n - cached_len;
        return static_cast<float>(fresh[
            (((b * heads + h) * fresh_len + fresh_n) * HeadDim) + d]);
    }
    const int word_idx = d / vals_per_word;
    const int lane = d - word_idx * vals_per_word;
    const uint32_t word = static_cast<uint32_t>(packed[
        (((b * heads + h) * cached_len + n) * packed_words) + word_idx]);
    const uint32_t piece = (word >> (lane * bits)) & ((1u << bits) - 1u);
    const int32_t sign = 1 << (bits - 1);
    const int32_t signed_piece = (piece & sign)
        ? static_cast<int32_t>(piece) - (1 << bits)
        : static_cast<int32_t>(piece);
    const int step = step_ids[n];
    const float s = static_cast<float>(scale[
        (((b * heads + h) * num_steps + step) * HeadDim) + d]);
    return static_cast<float>(signed_piece) * s;
}

template <typename T, typename ScaleT, int HeadDim>
__global__ void packed_attention_kernel(
    const T *__restrict__ q,
    const int32_t *__restrict__ k_packed,
    const int32_t *__restrict__ v_packed,
    const ScaleT *__restrict__ k_scale,
    const ScaleT *__restrict__ v_scale,
    const int32_t *__restrict__ step_ids,
    const T *__restrict__ k_fresh,
    const T *__restrict__ v_fresh,
    T *__restrict__ out,
    int batches,
    int heads,
    int query_len,
    int cached_len,
    int fresh_len,
    int num_steps,
    int bits,
    int vals_per_word,
    int packed_words,
    float softmax_scale) {
    const int64_t query_row = blockIdx.x;
    const int64_t total_rows = static_cast<int64_t>(batches) * heads * query_len;
    if (query_row >= total_rows) return;
    const int d = threadIdx.x;
    const int q_idx = query_row % query_len;
    int64_t rem = query_row / query_len;
    const int h = rem % heads;
    const int b = rem / heads;
    const int total_k = cached_len + fresh_len;

    __shared__ float q_shared[HeadDim];
    __shared__ float partial[HeadDim];
    __shared__ float score;
    __shared__ float max_score;
    __shared__ float sum_exp;
    if (d < HeadDim) {
        q_shared[d] = static_cast<float>(q[
            (((b * heads + h) * query_len + q_idx) * HeadDim) + d]);
    }
    if (d == 0) {
        max_score = -CUDART_INF_F;
        sum_exp = 0.0f;
    }
    __syncthreads();

    for (int n = 0; n < total_k; ++n) {
        if (d < HeadDim) {
            partial[d] = q_shared[d] * load_packed_or_fresh<T, ScaleT, HeadDim>(
                k_packed, k_scale, step_ids, k_fresh, b, h, n, d, heads,
                cached_len, fresh_len, num_steps, bits, vals_per_word, packed_words);
        }
        __syncthreads();
        if (d == 0) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < HeadDim; ++i) dot += partial[i];
            score = dot * softmax_scale;
            max_score = fmaxf(max_score, score);
        }
        __syncthreads();
    }

    for (int n = 0; n < total_k; ++n) {
        if (d < HeadDim) {
            partial[d] = q_shared[d] * load_packed_or_fresh<T, ScaleT, HeadDim>(
                k_packed, k_scale, step_ids, k_fresh, b, h, n, d, heads,
                cached_len, fresh_len, num_steps, bits, vals_per_word, packed_words);
        }
        __syncthreads();
        if (d == 0) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < HeadDim; ++i) dot += partial[i];
            score = __expf(dot * softmax_scale - max_score);
            sum_exp += score;
        }
        __syncthreads();
    }

    float accumulator = 0.0f;
    for (int n = 0; n < total_k; ++n) {
        if (d < HeadDim) {
            partial[d] = q_shared[d] * load_packed_or_fresh<T, ScaleT, HeadDim>(
                k_packed, k_scale, step_ids, k_fresh, b, h, n, d, heads,
                cached_len, fresh_len, num_steps, bits, vals_per_word, packed_words);
        }
        __syncthreads();
        if (d == 0) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < HeadDim; ++i) dot += partial[i];
            score = __expf(dot * softmax_scale - max_score) / fmaxf(sum_exp, 1.0e-20f);
        }
        __syncthreads();
        if (d < HeadDim) {
            accumulator += score * load_packed_or_fresh<T, ScaleT, HeadDim>(
                v_packed, v_scale, step_ids, v_fresh, b, h, n, d, heads,
                cached_len, fresh_len, num_steps, bits, vals_per_word, packed_words);
        }
        __syncthreads();
    }
    if (d < HeadDim) {
        out[(((b * heads + h) * query_len + q_idx) * HeadDim) + d] =
            static_cast<T>(total_k > 0 ? accumulator : 0.0f);
    }
}

template <typename T, typename ScaleT>
at::Tensor launch_packed_attention_typed(
    const at::Tensor &q,
    const at::Tensor &k_packed,
    const at::Tensor &v_packed,
    const at::Tensor &k_scale,
    const at::Tensor &v_scale,
    const at::Tensor &step_ids,
    const at::Tensor &k_fresh,
    const at::Tensor &v_fresh,
    double softmax_scale) {
    const int head_dim = q.size(3);
    const PackedFormat format = packed_format_from_words(k_packed.size(3), head_dim);
    auto out = at::empty_like(q);
    const int64_t rows = q.size(0) * q.size(1) * q.size(2);
    const int threads = head_dim == 64 ? 64 : 128;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (head_dim == 64) {
        packed_attention_kernel<T, ScaleT, 64><<<rows, threads, 0, stream>>>(
            q.data_ptr<T>(), k_packed.data_ptr<int32_t>(), v_packed.data_ptr<int32_t>(),
            k_scale.data_ptr<ScaleT>(), v_scale.data_ptr<ScaleT>(), step_ids.data_ptr<int32_t>(),
            k_fresh.data_ptr<T>(), v_fresh.data_ptr<T>(), out.data_ptr<T>(),
            q.size(0), q.size(1), q.size(2), k_packed.size(2), k_fresh.size(2),
            k_scale.size(2), format.bits, format.vals_per_word, format.packed_words,
            static_cast<float>(softmax_scale));
    } else {
        packed_attention_kernel<T, ScaleT, 128><<<rows, threads, 0, stream>>>(
            q.data_ptr<T>(), k_packed.data_ptr<int32_t>(), v_packed.data_ptr<int32_t>(),
            k_scale.data_ptr<ScaleT>(), v_scale.data_ptr<ScaleT>(), step_ids.data_ptr<int32_t>(),
            k_fresh.data_ptr<T>(), v_fresh.data_ptr<T>(), out.data_ptr<T>(),
            q.size(0), q.size(1), q.size(2), k_packed.size(2), k_fresh.size(2),
            k_scale.size(2), format.bits, format.vals_per_word, format.packed_words,
            static_cast<float>(softmax_scale));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor launch_packed_attention(
    const at::Tensor &q,
    const at::Tensor &k_packed,
    const at::Tensor &v_packed,
    const at::Tensor &k_scale,
    const at::Tensor &v_scale,
    const at::Tensor &step_ids,
    const at::Tensor &k_fresh,
    const at::Tensor &v_fresh,
    double softmax_scale) {
    TORCH_CHECK(k_scale.scalar_type() == v_scale.scalar_type(),
                "K/V scale dtypes must match");
#define VARQ_DISPATCH_SCALE(QTYPE, CTYPE) \
    if (k_scale.scalar_type() == at::kHalf) \
        return launch_packed_attention_typed<QTYPE, at::Half>(q, k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh, softmax_scale); \
    if (k_scale.scalar_type() == at::kBFloat16) \
        return launch_packed_attention_typed<QTYPE, at::BFloat16>(q, k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh, softmax_scale); \
    if (k_scale.scalar_type() == at::kFloat) \
        return launch_packed_attention_typed<QTYPE, float>(q, k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh, softmax_scale)
    if (q.scalar_type() == at::kHalf) {
        VARQ_DISPATCH_SCALE(at::Half, at::kHalf);
    } else if (q.scalar_type() == at::kBFloat16) {
        VARQ_DISPATCH_SCALE(at::BFloat16, at::kBFloat16);
    } else if (q.scalar_type() == at::kFloat) {
        VARQ_DISPATCH_SCALE(float, at::kFloat);
    }
#undef VARQ_DISPATCH_SCALE
    TORCH_CHECK(false, "unsupported q/scale dtype for packed attention");
    return at::Tensor();
}

void check_supported_bits(int bits) {
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8,
                "bits must be one of 2, 3, 4, 6, 8");
}

at::ScalarType scalar_type_from_code(int64_t code, at::ScalarType fallback) {
    if (code < 0) return fallback;
    if (code == 0) return at::kHalf;
    if (code == 1) return at::kBFloat16;
    if (code == 2) return at::kFloat;
    TORCH_CHECK(false, "dtype code must be -1, 0(fp16), 1(bf16), or 2(fp32)");
    return fallback;
}

void check_float_dtype(const at::Tensor &x, const char *name) {
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16
                    || x.scalar_type() == at::kFloat,
                name, " must be fp16, bf16, or fp32");
}

template <typename X, typename S>
void launch_quantize_row_scale(
    const at::Tensor &x,
    at::Tensor &scale,
    at::Tensor &packed,
    int bits,
    const PackedFormat &format,
    cudaStream_t stream) {
    const int64_t dim = x.size(3);
    const int64_t rows = x.numel() / dim;
    compute_row_scale_kernel<X><<<rows, 256, 0, stream>>>(
        x.data_ptr<X>(), scale.data_ptr(), rows, dim,
        scale.scalar_type() == at::kHalf ? 0 : scale.scalar_type() == at::kBFloat16 ? 1 : 2,
        bits);
    const int64_t total_words = rows * format.packed_words;
    const int blocks = static_cast<int>((total_words + 255) / 256);
    quantize_pack_row_scale_kernel<X, S><<<blocks, 256, 0, stream>>>(
        x.data_ptr<X>(), scale.data_ptr<S>(), packed.data_ptr<int32_t>(),
        rows, dim, format.packed_words, bits, format.vals_per_word);
}

template <typename X, typename S>
void launch_quantize_broadcast(
    const at::Tensor &x,
    const at::Tensor &scale,
    const at::Tensor *group_ids,
    at::Tensor &packed,
    int bits,
    const PackedFormat &format,
    int layout,
    cudaStream_t stream) {
    const ScaleMeta meta = make_scale_meta(scale, x, group_ids, layout);
    const int64_t rows = x.size(0) * x.size(1) * x.size(2);
    const int64_t total_words = rows * format.packed_words;
    const int blocks = static_cast<int>((total_words + 255) / 256);
    quantize_pack_broadcast_kernel<X, S><<<blocks, 256, 0, stream>>>(
        x.data_ptr<X>(), scale.data_ptr<S>(),
        group_ids == nullptr ? nullptr : group_ids->data_ptr<int32_t>(),
        packed.data_ptr<int32_t>(), x.size(1), x.size(2), x.size(0), x.size(3),
        format.packed_words, bits, format.vals_per_word, meta);
}

template <typename X>
void dispatch_quantize_scale_type(
    const at::Tensor &x,
    at::Tensor &scale,
    const at::Tensor *group_ids,
    at::Tensor &packed,
    int bits,
    const PackedFormat &format,
    int layout,
    bool row_scale,
    cudaStream_t stream) {
    if (scale.scalar_type() == at::kHalf) {
        if (row_scale) launch_quantize_row_scale<X, at::Half>(x, scale, packed, bits, format, stream);
        else launch_quantize_broadcast<X, at::Half>(x, scale, group_ids, packed, bits, format, layout, stream);
    } else if (scale.scalar_type() == at::kBFloat16) {
        if (row_scale) launch_quantize_row_scale<X, at::BFloat16>(x, scale, packed, bits, format, stream);
        else launch_quantize_broadcast<X, at::BFloat16>(x, scale, group_ids, packed, bits, format, layout, stream);
    } else if (scale.scalar_type() == at::kFloat) {
        if (row_scale) launch_quantize_row_scale<X, float>(x, scale, packed, bits, format, stream);
        else launch_quantize_broadcast<X, float>(x, scale, group_ids, packed, bits, format, layout, stream);
    } else {
        TORCH_CHECK(false, "scale must be fp16, bf16, or fp32");
    }
}

template <typename OutT, typename ScaleT>
void launch_unpack_dequant_typed(
    const at::Tensor &packed,
    const at::Tensor &scale,
    const at::Tensor *group_ids,
    at::Tensor &out,
    int bits,
    const PackedFormat &format,
    const ScaleMeta &meta,
    cudaStream_t stream) {
    const int64_t total = out.numel();
    const int blocks = static_cast<int>((total + 255) / 256);
    unpack_dequant_kernel<OutT, ScaleT><<<blocks, 256, 0, stream>>>(
        packed.data_ptr<int32_t>(), scale.data_ptr<ScaleT>(),
        group_ids == nullptr ? nullptr : group_ids->data_ptr<int32_t>(), out.data_ptr<OutT>(),
        out.size(0), out.size(1), out.size(2), out.size(3), format.packed_words,
        bits, format.vals_per_word, meta);
}

template <typename OutT>
void dispatch_unpack_scale_type(
    const at::Tensor &packed,
    const at::Tensor &scale,
    const at::Tensor *group_ids,
    at::Tensor &out,
    int bits,
    const PackedFormat &format,
    const ScaleMeta &meta,
    cudaStream_t stream) {
    if (scale.scalar_type() == at::kHalf) {
        launch_unpack_dequant_typed<OutT, at::Half>(packed, scale, group_ids, out, bits, format, meta, stream);
    } else if (scale.scalar_type() == at::kBFloat16) {
        launch_unpack_dequant_typed<OutT, at::BFloat16>(packed, scale, group_ids, out, bits, format, meta, stream);
    } else if (scale.scalar_type() == at::kFloat) {
        launch_unpack_dequant_typed<OutT, float>(packed, scale, group_ids, out, bits, format, meta, stream);
    } else {
        TORCH_CHECK(false, "scale must be fp16, bf16, or fp32");
    }
}

}  // namespace

std::string backend_info() {
    std::ostringstream oss;
    oss << "VAR-Q CUDA runtime"
        << "; target=sm80"
        << "; dtype=fp16,bf16,fp32"
        << "; head_dim=64,128"
        << "; attention_q_bits=8,6,4,3,2"
        << "; standalone_q_bits=8,6,4,3,2"
        << "; standalone=quantize_pack,pack_int8,unpack_int8,unpack_dequant"
        << "; direct=generic-packed-kv";
    return oss.str();
}

std::tuple<at::Tensor, at::Tensor> quantize_pack(
    at::Tensor x,
    int64_t bits,
    c10::optional<at::Tensor> scale_opt,
    c10::optional<at::Tensor> group_ids_opt,
    int64_t scale_dtype,
    int64_t layout) {
    CHECK_CUDA(x);
    check_float_dtype(x, "x");
    TORCH_CHECK(x.dim() == 4, "quantize_pack expects rank-4 BLHc/BHLc x");
    TORCH_CHECK(x.size(3) > 0, "x last dimension must be non-empty");
    TORCH_CHECK(layout == 0 || layout == 1, "layout must be 0(BLHc) or 1(BHLc)");
    check_supported_bits(bits);
    const at::cuda::CUDAGuard device_guard(x.device());
    x = x.contiguous();

    at::Tensor group_ids;
    const at::Tensor *group_ptr = nullptr;
    if (group_ids_opt.has_value()) {
        group_ids = group_ids_opt.value();
        CHECK_CUDA(group_ids);
        TORCH_CHECK(group_ids.device() == x.device(), "group_ids must be on x device");
        group_ids = group_ids.to(at::kInt).contiguous();
        group_ptr = &group_ids;
    }

    const PackedFormat format = packed_format_from_bits(bits, x.size(3));
    std::vector<int64_t> packed_shape = x.sizes().vec();
    packed_shape.back() = format.packed_words;
    at::Tensor packed = at::empty(packed_shape, x.options().dtype(at::kInt));

    at::Tensor used_scale;
    bool row_scale = !scale_opt.has_value();
    if (row_scale) {
        TORCH_CHECK(group_ptr == nullptr,
                    "group_ids require an explicit compact/broadcast scale");
        const at::ScalarType dtype = scalar_type_from_code(scale_dtype, at::kFloat);
        std::vector<int64_t> scale_shape = x.sizes().vec();
        scale_shape.back() = 1;
        used_scale = at::empty(scale_shape, x.options().dtype(dtype));
    } else {
        used_scale = scale_opt.value();
        CHECK_CUDA(used_scale);
        TORCH_CHECK(used_scale.device() == x.device(), "scale must be on x device");
        check_float_dtype(used_scale, "scale");
        if (scale_dtype >= 0) {
            TORCH_CHECK(used_scale.scalar_type() == scalar_type_from_code(scale_dtype, used_scale.scalar_type()),
                        "supplied scale dtype does not match scale_dtype");
        }
        used_scale = used_scale.contiguous();
        if (group_ptr != nullptr && group_ids.numel() > 0) {
            const int token_dim = layout == 0 ? 1 : 2;
            const int64_t max_group = group_ids.max().item<int64_t>();
            const int64_t min_group = group_ids.min().item<int64_t>();
            const int offset = 4 - used_scale.dim();
            const int scale_axis = token_dim - offset;
            TORCH_CHECK(scale_axis >= 0, "compact scale must expose the token/group axis");
            TORCH_CHECK(min_group >= 0 && max_group < used_scale.size(scale_axis),
                        "group_ids are outside compact scale group dimension");
        }
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (x.scalar_type() == at::kHalf) {
        dispatch_quantize_scale_type<at::Half>(x, used_scale, group_ptr, packed, bits, format, layout, row_scale, stream);
    } else if (x.scalar_type() == at::kBFloat16) {
        dispatch_quantize_scale_type<at::BFloat16>(x, used_scale, group_ptr, packed, bits, format, layout, row_scale, stream);
    } else {
        dispatch_quantize_scale_type<float>(x, used_scale, group_ptr, packed, bits, format, layout, row_scale, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(packed, used_scale);
}

at::Tensor pack_int8(at::Tensor q_int8, int64_t bits) {
    CHECK_CUDA(q_int8);
    CHECK_DTYPE(q_int8, at::kChar);
    TORCH_CHECK(q_int8.dim() >= 1, "q_int8 must have rank >= 1");
    TORCH_CHECK(q_int8.size(-1) > 0, "q_int8 last dimension must be non-empty");
    check_supported_bits(bits);
    const at::cuda::CUDAGuard device_guard(q_int8.device());
    q_int8 = q_int8.contiguous();
    const int64_t dim = q_int8.size(-1);
    const int64_t rows = q_int8.numel() / dim;
    const PackedFormat format = packed_format_from_bits(bits, dim);
    std::vector<int64_t> shape = q_int8.sizes().vec();
    shape.back() = format.packed_words;
    at::Tensor packed = at::empty(shape, q_int8.options().dtype(at::kInt));
    const int64_t total = rows * format.packed_words;
    pack_int8_kernel<int8_t><<<static_cast<int>((total + 255) / 256), 256, 0,
                                at::cuda::getCurrentCUDAStream().stream()>>>(
        q_int8.data_ptr<int8_t>(), packed.data_ptr<int32_t>(), rows, dim,
        format.packed_words, bits, format.vals_per_word);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return packed;
}

at::Tensor unpack_int8(at::Tensor packed, int64_t bits, int64_t orig_dim) {
    CHECK_CUDA(packed);
    CHECK_DTYPE(packed, at::kInt);
    TORCH_CHECK(packed.dim() >= 1, "packed must have rank >= 1");
    check_supported_bits(bits);
    const at::cuda::CUDAGuard device_guard(packed.device());
    packed = packed.contiguous();
    const int vals = vals_per_word_for_bits(bits);
    if (orig_dim < 0) orig_dim = packed.size(-1) * vals;
    TORCH_CHECK(orig_dim > 0, "orig_dim must be positive");
    const PackedFormat format = packed_format_from_bits(bits, orig_dim);
    TORCH_CHECK(packed.size(-1) == format.packed_words,
                "packed width does not match bits/orig_dim");
    std::vector<int64_t> shape = packed.sizes().vec();
    shape.back() = orig_dim;
    at::Tensor out = at::empty(shape, packed.options().dtype(at::kChar));
    const int64_t rows = packed.numel() / packed.size(-1);
    const int64_t total = rows * orig_dim;
    unpack_int8_kernel<int8_t><<<static_cast<int>((total + 255) / 256), 256, 0,
                                  at::cuda::getCurrentCUDAStream().stream()>>>(
        packed.data_ptr<int32_t>(), out.data_ptr<int8_t>(), rows, orig_dim,
        format.packed_words, bits, format.vals_per_word);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor unpack_dequant(
    at::Tensor packed,
    at::Tensor scale,
    int64_t bits,
    int64_t orig_dim,
    int64_t output_dtype,
    c10::optional<at::Tensor> group_ids_opt,
    int64_t layout,
    c10::optional<at::Tensor> out_opt) {
    CHECK_CUDA(packed);
    CHECK_CUDA(scale);
    CHECK_DTYPE(packed, at::kInt);
    check_float_dtype(scale, "scale");
    TORCH_CHECK(packed.dim() == 4, "unpack_dequant expects rank-4 packed BLHw/BHLw");
    TORCH_CHECK(layout == 0 || layout == 1, "layout must be 0(BLHc) or 1(BHLc)");
    check_supported_bits(bits);
    const at::cuda::CUDAGuard device_guard(packed.device());
    TORCH_CHECK(scale.device() == packed.device(), "scale must be on packed device");
    packed = packed.contiguous();
    scale = scale.contiguous();
    const int vals = vals_per_word_for_bits(bits);
    if (orig_dim < 0) orig_dim = packed.size(3) * vals;
    TORCH_CHECK(orig_dim > 0, "orig_dim must be positive");
    const PackedFormat format = packed_format_from_bits(bits, orig_dim);
    TORCH_CHECK(packed.size(3) == format.packed_words,
                "packed width does not match bits/orig_dim");

    std::vector<int64_t> out_shape = packed.sizes().vec();
    out_shape.back() = orig_dim;
    const at::ScalarType out_type = scalar_type_from_code(output_dtype, scale.scalar_type());
    at::Tensor out;
    if (out_opt.has_value()) {
        out = out_opt.value();
        CHECK_CUDA(out);
        TORCH_CHECK(out.device() == packed.device(), "out must be on packed device");
        check_float_dtype(out, "out");
        TORCH_CHECK(out.scalar_type() == out_type,
                    "out dtype does not match output_dtype");
        TORCH_CHECK(out.sizes().vec() == out_shape,
                    "out shape must match the unpacked tensor; got ", out.sizes());
        CHECK_CONTIGUOUS(out);
    } else {
        out = at::empty(out_shape, packed.options().dtype(out_type));
    }

    at::Tensor group_ids;
    const at::Tensor *group_ptr = nullptr;
    if (group_ids_opt.has_value()) {
        group_ids = group_ids_opt.value();
        CHECK_CUDA(group_ids);
        TORCH_CHECK(group_ids.device() == packed.device(), "group_ids must be on packed device");
        group_ids = group_ids.to(at::kInt).contiguous();
        group_ptr = &group_ids;
    }
    const ScaleMeta meta = make_scale_meta(scale, out, group_ptr, layout);
    if (group_ptr != nullptr && group_ids.numel() > 0) {
        const int token_dim = layout == 0 ? 1 : 2;
        const int offset = 4 - scale.dim();
        const int scale_axis = token_dim - offset;
        TORCH_CHECK(scale_axis >= 0, "compact scale must expose the token/group axis");
        const int64_t max_group = group_ids.max().item<int64_t>();
        const int64_t min_group = group_ids.min().item<int64_t>();
        TORCH_CHECK(min_group >= 0 && max_group < scale.size(scale_axis),
                    "group_ids are outside compact scale group dimension");
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (out_type == at::kHalf) {
        dispatch_unpack_scale_type<at::Half>(packed, scale, group_ptr, out, bits, format, meta, stream);
    } else if (out_type == at::kBFloat16) {
        dispatch_unpack_scale_type<at::BFloat16>(packed, scale, group_ptr, out, bits, format, meta, stream);
    } else {
        dispatch_unpack_scale_type<float>(packed, scale, group_ptr, out, bits, format, meta, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor fwd(
    at::Tensor q,
    at::Tensor k_packed,
    at::Tensor v_packed,
    at::Tensor k_scale,
    at::Tensor v_scale,
    at::Tensor step_ids,
    at::Tensor k_fresh,
    at::Tensor v_fresh,
    double softmax_scale) {
    const at::cuda::CUDAGuard device_guard(q.device());
    validate_inputs(q, k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh, false);

    q = q.contiguous();
    k_fresh = k_fresh.contiguous();
    v_fresh = v_fresh.contiguous();
    // Preserve the PR15 BF16/fp16 FlashAttention bridge for its supported
    // head_dim=128 case.  All other CUDA dtypes/shapes use the common direct
    // dispatch below.
    const bool bridge_scale_supported =
        (q.scalar_type() == at::kHalf && k_scale.scalar_type() == at::kHalf)
        || (q.scalar_type() == at::kBFloat16
            && (k_scale.scalar_type() == at::kHalf || k_scale.scalar_type() == at::kBFloat16));
    const bool use_legacy_bridge = q.size(3) == 128
        && (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16)
        && bridge_scale_supported
        && k_fresh.size(2) > 0;
    if (!use_legacy_bridge) {
        return launch_packed_attention(q, k_packed, v_packed, k_scale, v_scale,
                                       step_ids, k_fresh, v_fresh, softmax_scale);
    }

    const int B = q.size(0);
    const int H = q.size(1);
    const int seqlen_q = q.size(2);
    const int seqlen_cached = k_packed.size(2);
    const int seqlen_fresh = k_fresh.size(2);
    const int seqlen_k = seqlen_cached + seqlen_fresh;
    auto opts = q.options();

    at::Tensor k_dense = at::empty({B, seqlen_k, H, kHeadDim}, opts);
    at::Tensor v_dense = at::empty({B, seqlen_k, H, kHeadDim}, opts);
    at::Tensor q_blhc = q.transpose(1, 2);
    at::Tensor out_blhc = at::empty({B, seqlen_q, H, kHeadDim}, opts);
    at::Tensor softmax_lse = at::empty({B, H, seqlen_q}, opts.dtype(at::kFloat));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_dequant_to_blhc(k_packed, k_scale, step_ids, k_dense, 0, stream, kHeadDim);
    launch_dequant_to_blhc(v_packed, v_scale, step_ids, v_dense, 0, stream, kHeadDim);
    launch_copy_bhlc_to_blhc(k_fresh, k_dense, seqlen_cached, stream, kHeadDim);
    launch_copy_bhlc_to_blhc(v_fresh, v_dense, seqlen_cached, stream, kHeadDim);

    Flash_fwd_params params;
    set_dense_fwd_params(params, q_blhc, k_dense, v_dense, out_blhc, softmax_lse);
    params.scale_softmax = static_cast<float>(softmax_scale);
    params.scale_softmax_log2 = params.scale_softmax * 1.4426950408889634f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    if (q.scalar_type() == at::kHalf) {
        run_mha_fwd_<cutlass::half_t, kHeadDim, false>(params, stream);
    } else {
        run_mha_fwd_<cutlass::bfloat16_t, kHeadDim, false>(params, stream);
    }
    return out_blhc.transpose(1, 2).contiguous();
}

at::Tensor fwd_direct(
    at::Tensor q,
    at::Tensor k_packed,
    at::Tensor v_packed,
    at::Tensor k_scale,
    at::Tensor v_scale,
    at::Tensor step_ids,
    at::Tensor k_fresh,
    at::Tensor v_fresh,
    double softmax_scale) {
    const at::cuda::CUDAGuard device_guard(q.device());
    validate_inputs(q, k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh, true);
    q = q.contiguous();
    k_fresh = k_fresh.contiguous();
    v_fresh = v_fresh.contiguous();
    const PackedFormat direct_format = packed_format_from_words(k_packed.size(3), q.size(3));
    const bool use_flash_direct = q.scalar_type() == at::kHalf
        && q.size(3) == 128
        && direct_format.bits != 6
        && k_scale.scalar_type() == at::kHalf
        && k_fresh.size(2) > 0;
    if (!use_flash_direct) {
        return launch_packed_attention(q, k_packed, v_packed, k_scale, v_scale,
                                       step_ids, k_fresh, v_fresh, softmax_scale);
    }

    const int B = q.size(0);
    const int H = q.size(1);
    const int seqlen_q = q.size(2);
    at::Tensor q_blhc = q.transpose(1, 2);
    at::Tensor out_bhlc = at::empty({B, H, seqlen_q, 128}, q.options());
    at::Tensor softmax_lse = at::empty({B, H, seqlen_q}, q.options().dtype(at::kFloat));
    Varq_fwd_params params;
    set_varq_fwd_params(
        params, q_blhc, out_bhlc, softmax_lse,
        k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh,
        /*output_bhlc=*/true);
    params.scale_softmax = static_cast<float>(softmax_scale);
    params.scale_softmax_log2 = params.scale_softmax * 1.4426950408889634f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    run_varq_mha_fwd_<cutlass::half_t, 128, false>(
        params, at::cuda::getCurrentCUDAStream().stream());
    return out_bhlc;
}

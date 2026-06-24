#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>

#include <cmath>
#include <sstream>
#include <string>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "varq_flash.h"

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_CONTIGUOUS_LAST(x) TORCH_CHECK((x).stride(-1) == 1, #x " must have contiguous last dimension")
#define CHECK_DTYPE(x, dtype) TORCH_CHECK((x).scalar_type() == (dtype), #x " has unexpected dtype")

constexpr int kHeadDim = 128;
constexpr int kPackedWords = 32;

void check_rank4(const at::Tensor &x, const char *name) {
    TORCH_CHECK(x.dim() == 4, name, " must be rank-4");
}

void check_bhlc_128(const at::Tensor &x, const char *name, bool allow_bf16) {
    CHECK_CUDA(x);
    check_rank4(x, name);
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || (allow_bf16 && x.scalar_type() == at::kBFloat16),
        name,
        " has unexpected dtype");
    CHECK_CONTIGUOUS_LAST(x);
    TORCH_CHECK(x.size(3) == kHeadDim, name, " must have head_dim=128");
}

void check_same_dtype(const at::Tensor &lhs, const at::Tensor &rhs, const char *rhs_name) {
    TORCH_CHECK(rhs.scalar_type() == lhs.scalar_type(), rhs_name, " dtype must match q dtype");
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

template <typename OutT>
__global__ void dequant_q8_to_blhc_kernel(
    const int32_t *__restrict__ packed,
    const at::Half *__restrict__ scale,
    const int32_t *__restrict__ step_ids,
    OutT *__restrict__ dense,
    int B,
    int H,
    int N,
    int S,
    int N_total,
    int dst_offset) {
    const int64_t total = static_cast<int64_t>(B) * H * N * kHeadDim;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int d = idx % kHeadDim;
        int64_t rem = idx / kHeadDim;
        const int n = rem % N;
        rem /= N;
        const int h = rem % H;
        const int b = rem / H;

        const int word_idx = d / 4;
        const int lane = d & 3;
        const int32_t word = packed[(((b * H + h) * N + n) * kPackedWords) + word_idx];
        const int32_t unsigned_piece = (word >> (lane * 8)) & 0xff;
        const int32_t signed_piece = unsigned_piece >= 128 ? unsigned_piece - 256 : unsigned_piece;
        const int step = step_ids[n];
        const float scale_value = static_cast<float>(scale[(((b * H + h) * S + step) * kHeadDim) + d]);
        dense[(((b * N_total + (dst_offset + n)) * H + h) * kHeadDim) + d] =
            static_cast<OutT>(static_cast<float>(signed_piece) * scale_value);
    }
}

template <typename OutT>
__global__ void copy_bhlc_to_blhc_kernel(
    const OutT *__restrict__ src,
    OutT *__restrict__ dst,
    int B,
    int H,
    int N,
    int N_total,
    int dst_offset) {
    const int64_t total = static_cast<int64_t>(B) * H * N * kHeadDim;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int d = idx % kHeadDim;
        int64_t rem = idx / kHeadDim;
        const int n = rem % N;
        rem /= N;
        const int h = rem % H;
        const int b = rem / H;
        dst[(((b * N_total + (dst_offset + n)) * H + h) * kHeadDim) + d] =
            src[(((b * H + h) * N + n) * kHeadDim) + d];
    }
}

template <typename OutT>
void launch_dequant_q8_to_blhc(
    const at::Tensor &packed,
    const at::Tensor &scale,
    const at::Tensor &step_ids,
    at::Tensor &dense,
    int dst_offset,
    cudaStream_t stream) {
    const int B = packed.size(0);
    const int H = packed.size(1);
    const int N = packed.size(2);
    const int S = scale.size(2);
    const int N_total = dense.size(1);
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(B) * H * N * kHeadDim;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    dequant_q8_to_blhc_kernel<OutT><<<blocks, threads, 0, stream>>>(
        packed.data_ptr<int32_t>(),
        scale.data_ptr<at::Half>(),
        step_ids.data_ptr<int32_t>(),
        dense.data_ptr<OutT>(),
        B,
        H,
        N,
        S,
        N_total,
        dst_offset);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename OutT>
void launch_copy_bhlc_to_blhc(
    const at::Tensor &src,
    at::Tensor &dst,
    int dst_offset,
    cudaStream_t stream) {
    const int B = src.size(0);
    const int H = src.size(1);
    const int N = src.size(2);
    const int N_total = dst.size(1);
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(B) * H * N * kHeadDim;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    copy_bhlc_to_blhc_kernel<OutT><<<blocks, threads, 0, stream>>>(
        src.data_ptr<OutT>(),
        dst.data_ptr<OutT>(),
        B,
        H,
        N,
        N_total,
        dst_offset);
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
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

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
}

void validate_inputs(
    const at::Tensor &q,
    const at::Tensor &k_packed,
    const at::Tensor &v_packed,
    const at::Tensor &k_scale,
    const at::Tensor &v_scale,
    const at::Tensor &step_ids,
    const at::Tensor &k_fresh,
    const at::Tensor &v_fresh,
    bool allow_bf16) {
    check_bhlc_128(q, "q", allow_bf16);
    check_bhlc_128(k_fresh, "k_fresh", allow_bf16);
    check_bhlc_128(v_fresh, "v_fresh", allow_bf16);
    check_same_dtype(q, k_fresh, "k_fresh");
    check_same_dtype(q, v_fresh, "v_fresh");
    check_same_bhlc_shape(q, k_fresh, "k_fresh");
    check_same_bhlc_shape(q, v_fresh, "v_fresh");

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
    CHECK_DTYPE(k_scale, at::kHalf);
    CHECK_DTYPE(v_scale, at::kHalf);
    CHECK_DTYPE(step_ids, at::kInt);
    CHECK_CONTIGUOUS(k_packed);
    CHECK_CONTIGUOUS(v_packed);
    CHECK_CONTIGUOUS(k_scale);
    CHECK_CONTIGUOUS(v_scale);
    CHECK_CONTIGUOUS(step_ids);
    TORCH_CHECK(k_packed.size(3) == kPackedWords, "q8 packed head_dim=128 requires 32 int32 words");
    TORCH_CHECK(v_packed.size(3) == kPackedWords, "q8 packed head_dim=128 requires 32 int32 words");
    TORCH_CHECK(k_packed.sizes() == v_packed.sizes(), "k_packed and v_packed shapes must match");
    TORCH_CHECK(k_packed.size(0) == q.size(0), "packed cache batch must match q");
    TORCH_CHECK(k_packed.size(1) == q.size(1), "packed cache heads must match q");
    TORCH_CHECK(k_scale.size(0) == q.size(0), "k_scale batch must match q");
    TORCH_CHECK(v_scale.size(0) == q.size(0), "v_scale batch must match q");
    TORCH_CHECK(k_scale.size(1) == q.size(1), "k_scale heads must match q");
    TORCH_CHECK(v_scale.size(1) == q.size(1), "v_scale heads must match q");
    TORCH_CHECK(k_scale.size(3) == kHeadDim, "k_scale head_dim must be 128");
    TORCH_CHECK(v_scale.size(3) == kHeadDim, "v_scale head_dim must be 128");
    TORCH_CHECK(k_scale.size(2) == v_scale.size(2), "K/V scale step counts must match");
    TORCH_CHECK(step_ids.size(0) == k_packed.size(2), "step_ids length must match cached token count");
}

}  // namespace

std::string backend_info() {
    std::ostringstream oss;
    oss << "VAR-Q Track B fused FlashAttention CUDA bridge"
        << "; target=sm80"
        << "; dtype=fp16/bf16"
        << "; head_dim=128"
        << "; status=dense-bridge"
        << "; direct=experimental";
    return oss.str();
}

at::Tensor fwd(
    at::Tensor q,
    at::Tensor k_packed,
    at::Tensor v_packed,
    at::Tensor k_scale,
    at::Tensor v_scale,
    at::Tensor step_ids,
    at::Tensor k_fresh,
    at::Tensor v_fresh) {
    const at::cuda::CUDAGuard device_guard(q.device());
    validate_inputs(q, k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh, /*allow_bf16=*/true);

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
    if (q.scalar_type() == at::kHalf) {
        launch_dequant_q8_to_blhc<at::Half>(k_packed, k_scale, step_ids, k_dense, 0, stream);
        launch_dequant_q8_to_blhc<at::Half>(v_packed, v_scale, step_ids, v_dense, 0, stream);
        launch_copy_bhlc_to_blhc<at::Half>(k_fresh, k_dense, seqlen_cached, stream);
        launch_copy_bhlc_to_blhc<at::Half>(v_fresh, v_dense, seqlen_cached, stream);
    } else {
        launch_dequant_q8_to_blhc<at::BFloat16>(k_packed, k_scale, step_ids, k_dense, 0, stream);
        launch_dequant_q8_to_blhc<at::BFloat16>(v_packed, v_scale, step_ids, v_dense, 0, stream);
        launch_copy_bhlc_to_blhc<at::BFloat16>(k_fresh, k_dense, seqlen_cached, stream);
        launch_copy_bhlc_to_blhc<at::BFloat16>(v_fresh, v_dense, seqlen_cached, stream);
    }

    Flash_fwd_params params;
    set_dense_fwd_params(params, q_blhc, k_dense, v_dense, out_blhc, softmax_lse);
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
    at::Tensor v_fresh) {
    const at::cuda::CUDAGuard device_guard(q.device());
    validate_inputs(q, k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh, /*allow_bf16=*/false);

    const int B = q.size(0);
    const int H = q.size(1);
    const int seqlen_q = q.size(2);
    auto opts = q.options();

    at::Tensor q_blhc = q.transpose(1, 2);
    at::Tensor out_bhlc = at::empty({B, H, seqlen_q, kHeadDim}, opts);
    at::Tensor softmax_lse = at::empty({B, H, seqlen_q}, opts.dtype(at::kFloat));

    Varq_fwd_params params;
    set_varq_fwd_params(
        params, q_blhc, out_bhlc, softmax_lse,
        k_packed, v_packed, k_scale, v_scale, step_ids, k_fresh, v_fresh,
        /*output_bhlc=*/true);
    run_varq_mha_fwd_<cutlass::half_t, kHeadDim, false>(
        params, at::cuda::getCurrentCUDAStream().stream());
    return out_bhlc;
}

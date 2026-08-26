#pragma once

#include <cstdint>

#include <cuda_fp16.h>

#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include <cute/tensor.hpp>

struct Varq_fwd_params : public Flash_fwd_params {
    void *__restrict__ varq_k_packed_ptr;
    void *__restrict__ varq_v_packed_ptr;
    void *__restrict__ varq_k_scale_ptr;
    void *__restrict__ varq_v_scale_ptr;
    void *__restrict__ varq_step_ids_ptr;
    void *__restrict__ varq_k_fresh_ptr;
    void *__restrict__ varq_v_fresh_ptr;
    int varq_cached_len;
    int varq_num_steps;
    int varq_block_n;
};

template<typename T, int Headdim, bool Is_causal>
void run_varq_mha_fwd_(Varq_fwd_params &params, cudaStream_t stream);

namespace flash {

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename EngineS, typename LayoutS, typename EngineD, typename LayoutD,
          typename EngineCoord, typename LayoutCoord, typename EnginePred, typename LayoutPred>
__forceinline__ __device__ void varq_copy_fresh_tile(
    TiledCopy tiled_copy,
    cute::Tensor<EngineS, LayoutS> const &S,
    cute::Tensor<EngineD, LayoutD> &D,
    cute::Tensor<EngineCoord, LayoutCoord> const &identity_MN,
    cute::Tensor<EnginePred, LayoutPred> const &predicate_K,
    const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(cute::rank(S) == cute::Int<3>{});
    CUTE_STATIC_ASSERT_V(cute::rank(D) == cute::Int<3>{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(S) == cute::size<0>(D));
    CUTE_STATIC_ASSERT_V(cute::size<1>(S) == cute::size<1>(D));
    CUTE_STATIC_ASSERT_V(cute::size<2>(S) == cute::size<2>(D));
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < cute::size<1>(S); ++m) {
        if (Is_even_MN || cute::get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < cute::size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(cute::_, m, k), D(cute::_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(cute::_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(cute::_, m, cute::_));
        }
    }
}

template <bool IsK>
__forceinline__ __device__ cutlass::half_t varq_load_kv_value(
    const Varq_fwd_params &params,
    const int bidb,
    const int bidh,
    const int n,
    const int d) {
    constexpr int kPackedWords = 32;
    if (n < params.varq_cached_len) {
        const int32_t *packed = reinterpret_cast<const int32_t *>(
            IsK ? params.varq_k_packed_ptr : params.varq_v_packed_ptr);
        const cutlass::half_t *scale = reinterpret_cast<const cutlass::half_t *>(
            IsK ? params.varq_k_scale_ptr : params.varq_v_scale_ptr);
        const int32_t *step_ids = reinterpret_cast<const int32_t *>(params.varq_step_ids_ptr);
        const int word_idx = d / 4;
        const int lane = d & 3;
        const int32_t word = packed[(((bidb * params.h + bidh) * params.varq_cached_len + n) * kPackedWords) + word_idx];
        const int32_t unsigned_piece = (word >> (lane * 8)) & 0xff;
        const int32_t signed_piece = unsigned_piece >= 128 ? unsigned_piece - 256 : unsigned_piece;
        const int step = step_ids[n];
        const float scale_value = static_cast<float>(
            scale[(((bidb * params.h + bidh) * params.varq_num_steps + step) * params.d) + d]);
        return cutlass::half_t(static_cast<float>(signed_piece) * scale_value);
    }
    const int fresh_n = n - params.varq_cached_len;
    const cutlass::half_t *fresh = reinterpret_cast<const cutlass::half_t *>(
        IsK ? params.varq_k_fresh_ptr : params.varq_v_fresh_ptr);
    return fresh[(((bidb * params.h + bidh) * params.seqlen_knew + fresh_n) * params.d) + d];
}

template <bool IsK>
__forceinline__ __device__ float varq_load_scale_value(
    const Varq_fwd_params &params,
    const int bidb,
    const int bidh,
    const int step,
    const int d) {
    const cutlass::half_t *scale = reinterpret_cast<const cutlass::half_t *>(
        IsK ? params.varq_k_scale_ptr : params.varq_v_scale_ptr);
    return static_cast<float>(
        scale[(((bidb * params.h + bidh) * params.varq_num_steps + step) * params.d) + d]);
}

template <bool IsK>
__forceinline__ __device__ cutlass::half_t varq_load_scale_half(
    const Varq_fwd_params &params,
    const int bidb,
    const int bidh,
    const int step,
    const int d) {
    const cutlass::half_t *scale = reinterpret_cast<const cutlass::half_t *>(
        IsK ? params.varq_k_scale_ptr : params.varq_v_scale_ptr);
    return scale[(((bidb * params.h + bidh) * params.varq_num_steps + step) * params.d) + d];
}

template <bool IsK>
__forceinline__ __device__ cutlass::half_t varq_load_cached_q8_value(
    const Varq_fwd_params &params,
    const int bidb,
    const int bidh,
    const int n,
    const int d,
    const float scale_value) {
    constexpr int kPackedWords = 32;
    const int32_t *packed = reinterpret_cast<const int32_t *>(
        IsK ? params.varq_k_packed_ptr : params.varq_v_packed_ptr);
    const int word_idx = d / 4;
    const int lane = d & 3;
    const int32_t word = packed[(((bidb * params.h + bidh) * params.varq_cached_len + n) * kPackedWords) + word_idx];
    const int32_t unsigned_piece = (word >> (lane * 8)) & 0xff;
    const int32_t signed_piece = unsigned_piece >= 128 ? unsigned_piece - 256 : unsigned_piece;
    return cutlass::half_t(static_cast<float>(signed_piece) * scale_value);
}

__forceinline__ __device__ cutlass::half_t varq_dequant_q8_lane(
    const int32_t word,
    const int lane,
    const cutlass::half_t scale_value) {
    const int32_t signed_piece = static_cast<int32_t>(
        static_cast<int8_t>((word >> (lane * 8)) & 0xff));
    return cutlass::half_t(static_cast<float>(signed_piece)) * scale_value;
}

__forceinline__ __device__ __half varq_q8_lane_as_half(
    const int32_t word,
    const int lane) {
    const int32_t signed_piece = static_cast<int32_t>(
        static_cast<int8_t>((word >> (lane * 8)) & 0xff));
    return __int2half_rn(signed_piece);
}

__forceinline__ __device__ __half2 varq_dequant_q8_pair(
    const int32_t word,
    const int lane,
    const __half2 scale_value) {
    const __half2 q8_pair = __halves2half2(
        varq_q8_lane_as_half(word, lane),
        varq_q8_lane_as_half(word, lane + 1));
    return __hmul2(q8_pair, scale_value);
}

constexpr int kVarqPackedWords = 32;

template <typename Kernel_traits>
struct VarqPackedTileSmem {
    static constexpr int kBytes =
        Kernel_traits::kBlockN * kVarqPackedWords * static_cast<int>(sizeof(int32_t));
};

__forceinline__ __device__ bool varq_is_cached_uniform_tile(
    const Varq_fwd_params &params,
    const int n_block,
    int &step) {
    const int first_n = n_block * params.varq_block_n;
    const int last_n = first_n + params.varq_block_n - 1;
    if (last_n >= params.varq_cached_len) {
        return false;
    }
    const int32_t *step_ids = reinterpret_cast<const int32_t *>(params.varq_step_ids_ptr);
    const int first_step = step_ids[first_n];
    const int last_step = step_ids[last_n];
    if (first_step != last_step) {
        return false;
    }
    step = first_step;
    return true;
}

__forceinline__ __device__ void varq_cp_async_cg_16(
    void *smem_ptr,
    const void *gmem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : :
                 "r"(smem_addr), "l"(gmem_ptr));
#else
    *reinterpret_cast<uint4 *>(smem_ptr) =
        *reinterpret_cast<const uint4 *>(gmem_ptr);
#endif
}

template <bool IsK>
__forceinline__ __device__ void varq_stage_cached_uniform_tile(
    const Varq_fwd_params &params,
    int32_t *packed_smem,
    const int bidb,
    const int bidh,
    const int n_block) {
    constexpr int kWordsPerCopy = 4;
    constexpr int kVecsPerRow = kVarqPackedWords / kWordsPerCopy;
    const int64_t head_idx = bidb * params.h + bidh;
    const int n_block_base = n_block * params.varq_block_n;
    const int32_t *packed = reinterpret_cast<const int32_t *>(
        IsK ? params.varq_k_packed_ptr : params.varq_v_packed_ptr);
    const int64_t packed_head_offset =
        head_idx * params.varq_cached_len * kVarqPackedWords;
    const int total_vecs = params.varq_block_n * kVecsPerRow;
    for (int vec_idx = threadIdx.x; vec_idx < total_vecs; vec_idx += blockDim.x) {
        const int row = vec_idx / kVecsPerRow;
        const int vec_in_row = vec_idx - row * kVecsPerRow;
        const int word_idx = vec_in_row * kWordsPerCopy;
        const int64_t global_word =
            packed_head_offset
            + static_cast<int64_t>(n_block_base + row) * kVarqPackedWords
            + word_idx;
        const int smem_word = row * kVarqPackedWords + word_idx;
        varq_cp_async_cg_16(
            packed_smem + smem_word,
            packed + global_word);
    }
}

template <bool IsK, bool Is_even_MN, bool Is_even_K, bool Clear_OOB_MN, bool Clear_OOB_K,
          typename EngineD, typename LayoutD, typename EngineCoord, typename LayoutCoord,
          typename EnginePred, typename LayoutPred>
__forceinline__ __device__ void varq_copy_cached_uniform_tile(
    const Varq_fwd_params &params,
    const int32_t *packed_smem,
    cute::Tensor<EngineD, LayoutD> &D,
    cute::Tensor<EngineCoord, LayoutCoord> const &identity_MN,
    cute::Tensor<EnginePred, LayoutPred> const &predicate_K,
    const int bidb,
    const int bidh,
    const int n_block,
    const int step,
    const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(cute::rank(D) == cute::Int<3>{});
    static_cast<void>(n_block);
    const int64_t head_idx = bidb * params.h + bidh;
    const cutlass::half_t *scale = reinterpret_cast<const cutlass::half_t *>(
        IsK ? params.varq_k_scale_ptr : params.varq_v_scale_ptr);
    const int64_t scale_head_step_offset =
        (head_idx * params.varq_num_steps + step) * params.d;
    #pragma unroll
    for (int k = 0; k < cute::size<2>(D); ++k) {
        if (Is_even_K || predicate_K(k)) {
            const int d0 = cute::get<1>(identity_MN(0, 0, k));
            constexpr int kVecSize = decltype(cute::size(D(cute::_, 0, k)))::value;
            if constexpr (kVecSize == 8) {
                if ((d0 & 7) == 0) {
                    const int word_idx0 = d0 / 4;
                    const __half2 *scale_pair = reinterpret_cast<const __half2 *>(
                        scale + scale_head_step_offset + d0);
                    #pragma unroll
                    for (int m = 0; m < cute::size<1>(D); ++m) {
                        if (Is_even_MN || cute::get<0>(identity_MN(0, m, 0)) < max_MN) {
                            auto d_vec = D(cute::_, m, k);
                            cutlass::Array<cutlass::half_t, kVecSize> values;
                            const int local_n = cute::get<0>(identity_MN(0, m, k));
                            const int64_t packed_offset =
                                static_cast<int64_t>(local_n) * kVarqPackedWords + word_idx0;
                            const uint64_t packed64 =
                                *reinterpret_cast<const uint64_t *>(packed_smem + packed_offset);
                            const int32_t word0 = static_cast<int32_t>(packed64 & 0xffffffffu);
                            const int32_t word1 = static_cast<int32_t>(packed64 >> 32);
                            __half2 *value_pair = reinterpret_cast<__half2 *>(values.data());
                            value_pair[0] = varq_dequant_q8_pair(word0, 0, scale_pair[0]);
                            value_pair[1] = varq_dequant_q8_pair(word0, 2, scale_pair[1]);
                            value_pair[2] = varq_dequant_q8_pair(word1, 0, scale_pair[2]);
                            value_pair[3] = varq_dequant_q8_pair(word1, 2, scale_pair[3]);
                            auto r_vec = cute::make_tensor(
                                cute::make_rmem_ptr<cutlass::half_t>(values.data()), d_vec.layout());
                            cute::copy(r_vec, d_vec);
                        } else if (Clear_OOB_MN) {
                            cute::clear(D(cute::_, m, cute::_));
                        }
                    }
                    continue;
                }
            }
            int word_idx_values[8];
            int lane_values[8];
            cutlass::half_t scale_values[8];
            #pragma unroll
            for (int c = 0; c < cute::size<0>(D); ++c) {
                const int d = cute::get<1>(identity_MN(c, 0, k));
                word_idx_values[c] = d / 4;
                lane_values[c] = d & 3;
                scale_values[c] = scale[scale_head_step_offset + d];
            }
            #pragma unroll
            for (int m = 0; m < cute::size<1>(D); ++m) {
                if (Is_even_MN || cute::get<0>(identity_MN(0, m, 0)) < max_MN) {
                    auto d_vec = D(cute::_, m, k);
                    constexpr int kVecSize = decltype(cute::size(d_vec))::value;
                    cutlass::Array<cutlass::half_t, kVecSize> values;
                    int cached_word_idx = -1;
                    int32_t cached_word = 0;
                    const int local_n0 = cute::get<0>(identity_MN(0, m, k));
                    #pragma unroll
                    for (int c = 0; c < cute::size<0>(D); ++c) {
                        const int word_idx = word_idx_values[c];
                        if (word_idx != cached_word_idx) {
                            cached_word = packed_smem[static_cast<int64_t>(local_n0) * kVarqPackedWords + word_idx];
                            cached_word_idx = word_idx;
                        }
                        values[c] = varq_dequant_q8_lane(cached_word, lane_values[c], scale_values[c]);
                    }
                    auto r_vec = cute::make_tensor(cute::make_rmem_ptr<cutlass::half_t>(values.data()), d_vec.layout());
                    cute::copy(r_vec, d_vec);
                } else if (Clear_OOB_MN) {
                    cute::clear(D(cute::_, m, cute::_));
                }
            }
        } else if (Clear_OOB_K) {
            #pragma unroll
            for (int m = 0; m < cute::size<1>(D); ++m) {
                cute::clear(D(cute::_, m, k));
            }
        }
    }
}

template <bool IsK, bool Is_even_MN, bool Is_even_K, bool Clear_OOB_MN, bool Clear_OOB_K,
          typename EngineD, typename LayoutD, typename EngineCoord, typename LayoutCoord,
          typename EnginePred, typename LayoutPred>
__forceinline__ __device__ bool varq_finalize_cached_uniform_tile(
    const Varq_fwd_params &params,
    const int32_t *packed_smem,
    cute::Tensor<EngineD, LayoutD> &D,
    cute::Tensor<EngineCoord, LayoutCoord> const &identity_MN,
    cute::Tensor<EnginePred, LayoutPred> const &predicate_K,
    const int bidb,
    const int bidh,
    const int n_block,
    const int max_MN=0) {
    if (packed_smem == nullptr) {
        return false;
    }
    int step = 0;
    if (!varq_is_cached_uniform_tile(params, n_block, step)) {
        return false;
    }
    varq_copy_cached_uniform_tile<IsK, Is_even_MN, Is_even_K, Clear_OOB_MN, Clear_OOB_K>(
        params, packed_smem, D, identity_MN, predicate_K, bidb, bidh, n_block, step, max_MN);
    return true;
}

template <bool IsK, bool Is_even_MN, bool Is_even_K, bool Clear_OOB_MN, bool Clear_OOB_K,
          typename TiledCopy, typename EngineS, typename LayoutS,
          typename EngineD, typename LayoutD, typename EngineCoord, typename LayoutCoord,
          typename EnginePred, typename LayoutPred>
__forceinline__ __device__ void varq_copy_mixed_tile(
    TiledCopy tiled_copy,
    cute::Tensor<EngineS, LayoutS> const &S,
    const Varq_fwd_params &params,
    cute::Tensor<EngineD, LayoutD> &D,
    cute::Tensor<EngineCoord, LayoutCoord> const &identity_MN,
    cute::Tensor<EnginePred, LayoutPred> const &predicate_K,
    const int bidb,
    const int bidh,
    const int n_block,
    const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(cute::rank(S) == cute::Int<3>{});
    CUTE_STATIC_ASSERT_V(cute::rank(D) == cute::Int<3>{});
    constexpr int kPackedWords = 32;
    const int64_t head_idx = bidb * params.h + bidh;
    const int32_t *packed = reinterpret_cast<const int32_t *>(
        IsK ? params.varq_k_packed_ptr : params.varq_v_packed_ptr);
    const cutlass::half_t *scale = reinterpret_cast<const cutlass::half_t *>(
        IsK ? params.varq_k_scale_ptr : params.varq_v_scale_ptr);
    const int32_t *step_ids = reinterpret_cast<const int32_t *>(params.varq_step_ids_ptr);
    const int64_t packed_head_offset = head_idx * params.varq_cached_len * kPackedWords;
    const int n_block_base = n_block * params.varq_block_n;
    #pragma unroll
    for (int k = 0; k < cute::size<2>(D); ++k) {
        if (Is_even_K || predicate_K(k)) {
            const int d0 = cute::get<1>(identity_MN(0, 0, k));
            constexpr int kVecSize = decltype(cute::size(D(cute::_, 0, k)))::value;
            if constexpr (kVecSize == 8) {
                if ((d0 & 7) == 0) {
                    const int word_idx0 = d0 / 4;
                    #pragma unroll
                    for (int m = 0; m < cute::size<1>(D); ++m) {
                        if (Is_even_MN || cute::get<0>(identity_MN(0, m, 0)) < max_MN) {
                            const int local_n = cute::get<0>(identity_MN(0, m, k));
                            const int n = n_block_base + local_n;
                            if (n < params.varq_cached_len) {
                                auto d_vec = D(cute::_, m, k);
                                cutlass::Array<cutlass::half_t, kVecSize> values;
                                const int step = step_ids[n];
                                const __half2 *scale_pair = reinterpret_cast<const __half2 *>(
                                    scale + (head_idx * params.varq_num_steps + step) * params.d + d0);
                                const int64_t packed_offset =
                                    packed_head_offset + static_cast<int64_t>(n) * kPackedWords + word_idx0;
                                const uint64_t packed64 =
                                    *reinterpret_cast<const uint64_t *>(packed + packed_offset);
                                const int32_t word0 = static_cast<int32_t>(packed64 & 0xffffffffu);
                                const int32_t word1 = static_cast<int32_t>(packed64 >> 32);
                                __half2 *value_pair = reinterpret_cast<__half2 *>(values.data());
                                value_pair[0] = varq_dequant_q8_pair(word0, 0, scale_pair[0]);
                                value_pair[1] = varq_dequant_q8_pair(word0, 2, scale_pair[1]);
                                value_pair[2] = varq_dequant_q8_pair(word1, 0, scale_pair[2]);
                                value_pair[3] = varq_dequant_q8_pair(word1, 2, scale_pair[3]);
                                auto r_vec = cute::make_tensor(
                                    cute::make_rmem_ptr<cutlass::half_t>(values.data()), d_vec.layout());
                                cute::copy(r_vec, d_vec);
                            } else {
                                cute::copy(tiled_copy, S(cute::_, m, k), D(cute::_, m, k));
                            }
                        } else if (Clear_OOB_MN) {
                            cute::clear(D(cute::_, m, cute::_));
                        }
                    }
                    continue;
                }
            }
            #pragma unroll
            for (int m = 0; m < cute::size<1>(D); ++m) {
                if (Is_even_MN || cute::get<0>(identity_MN(0, m, 0)) < max_MN) {
                    #pragma unroll
                    for (int c = 0; c < cute::size<0>(D); ++c) {
                        const int local_n = cute::get<0>(identity_MN(c, m, k));
                        const int d = cute::get<1>(identity_MN(c, m, k));
                        const int n = n_block_base + local_n;
                        D(c, m, k) = varq_load_kv_value<IsK>(params, bidb, bidh, n, d);
                    }
                } else if (Clear_OOB_MN) {
                    cute::clear(D(cute::_, m, cute::_));
                }
            }
        } else if (Clear_OOB_K) {
            #pragma unroll
            for (int m = 0; m < cute::size<1>(D); ++m) {
                cute::clear(D(cute::_, m, k));
            }
        }
    }
}

template <bool IsK, bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename EngineS, typename LayoutS,
          typename EngineD, typename LayoutD, typename EngineCoord, typename LayoutCoord,
          typename EnginePred, typename LayoutPred>
__forceinline__ __device__ void varq_copy_kv_tile(
    TiledCopy tiled_copy,
    cute::Tensor<EngineS, LayoutS> const &S,
    const Varq_fwd_params &params,
    cute::Tensor<EngineD, LayoutD> &D,
    cute::Tensor<EngineCoord, LayoutCoord> const &identity_MN,
    cute::Tensor<EnginePred, LayoutPred> const &predicate_K,
    const int bidb,
    const int bidh,
    const int n_block,
    int32_t *packed_smem,
    const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(cute::rank(D) == cute::Int<3>{});
    const int first_n = n_block * params.varq_block_n;
    const int last_n = first_n + params.varq_block_n - 1;
    if (first_n >= params.varq_cached_len) {
        varq_copy_fresh_tile<Is_even_MN, Is_even_K, Clear_OOB_MN, Clear_OOB_K>(
            tiled_copy, S, D, identity_MN, predicate_K, max_MN);
        return;
    }
    if (last_n < params.varq_cached_len) {
        const int32_t *step_ids = reinterpret_cast<const int32_t *>(params.varq_step_ids_ptr);
        const int first_step = step_ids[first_n];
        const int last_step = step_ids[last_n];
        if (first_step == last_step) {
            if (packed_smem != nullptr) {
                varq_stage_cached_uniform_tile<IsK>(params, packed_smem, bidb, bidh, n_block);
                return;
            }
            // Keep a correct fallback if an older generated call site omits scratch smem.
            varq_copy_mixed_tile<IsK, Is_even_MN, Is_even_K, Clear_OOB_MN, Clear_OOB_K>(
                tiled_copy, S, params, D, identity_MN, predicate_K, bidb, bidh, n_block, max_MN);
            return;
        }
    }
    varq_copy_mixed_tile<IsK, Is_even_MN, Is_even_K, Clear_OOB_MN, Clear_OOB_K>(
        tiled_copy, S, params, D, identity_MN, predicate_K, bidb, bidh, n_block, max_MN);
}

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename EngineS, typename LayoutS,
          typename EngineD, typename LayoutD, typename EngineCoord, typename LayoutCoord,
          typename EnginePred, typename LayoutPred>
__forceinline__ __device__ void varq_copy_k_tile(
    TiledCopy tiled_copy,
    cute::Tensor<EngineS, LayoutS> const &S,
    const Varq_fwd_params &params,
    cute::Tensor<EngineD, LayoutD> &D,
    cute::Tensor<EngineCoord, LayoutCoord> const &identity_MN,
    cute::Tensor<EnginePred, LayoutPred> const &predicate_K,
    const int bidb,
    const int bidh,
    const int n_block,
    int32_t *packed_smem,
    const int max_MN=0) {
    varq_copy_kv_tile<true, Is_even_MN, Is_even_K, Clear_OOB_MN, Clear_OOB_K>(
        tiled_copy, S, params, D, identity_MN, predicate_K, bidb, bidh, n_block, packed_smem, max_MN);
}

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename EngineS, typename LayoutS,
          typename EngineD, typename LayoutD, typename EngineCoord, typename LayoutCoord,
          typename EnginePred, typename LayoutPred>
__forceinline__ __device__ void varq_copy_v_tile(
    TiledCopy tiled_copy,
    cute::Tensor<EngineS, LayoutS> const &S,
    const Varq_fwd_params &params,
    cute::Tensor<EngineD, LayoutD> &D,
    cute::Tensor<EngineCoord, LayoutCoord> const &identity_MN,
    cute::Tensor<EnginePred, LayoutPred> const &predicate_K,
    const int bidb,
    const int bidh,
    const int n_block,
    int32_t *packed_smem,
    const int max_MN=0) {
    varq_copy_kv_tile<false, Is_even_MN, Is_even_K, Clear_OOB_MN, Clear_OOB_K>(
        tiled_copy, S, params, D, identity_MN, predicate_K, bidb, bidh, n_block, packed_smem, max_MN);
}

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename EngineD, typename LayoutD, typename EngineCoord, typename LayoutCoord,
          typename EnginePred, typename LayoutPred>
__forceinline__ __device__ bool varq_finalize_k_tile(
    const Varq_fwd_params &params,
    cute::Tensor<EngineD, LayoutD> &D,
    cute::Tensor<EngineCoord, LayoutCoord> const &identity_MN,
    cute::Tensor<EnginePred, LayoutPred> const &predicate_K,
    const int32_t *packed_smem,
    const int bidb,
    const int bidh,
    const int n_block,
    const int max_MN=0) {
    return varq_finalize_cached_uniform_tile<true, Is_even_MN, Is_even_K, Clear_OOB_MN, Clear_OOB_K>(
        params, packed_smem, D, identity_MN, predicate_K, bidb, bidh, n_block, max_MN);
}

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename EngineD, typename LayoutD, typename EngineCoord, typename LayoutCoord,
          typename EnginePred, typename LayoutPred>
__forceinline__ __device__ bool varq_finalize_v_tile(
    const Varq_fwd_params &params,
    cute::Tensor<EngineD, LayoutD> &D,
    cute::Tensor<EngineCoord, LayoutCoord> const &identity_MN,
    cute::Tensor<EnginePred, LayoutPred> const &predicate_K,
    const int32_t *packed_smem,
    const int bidb,
    const int bidh,
    const int n_block,
    const int max_MN=0) {
    return varq_finalize_cached_uniform_tile<false, Is_even_MN, Is_even_K, Clear_OOB_MN, Clear_OOB_K>(
        params, packed_smem, D, identity_MN, predicate_K, bidb, bidh, n_block, max_MN);
}

}  // namespace flash

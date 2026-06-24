#include <c10/cuda/CUDAException.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "hardware_info.h"
#include "kernel_traits.h"
#include "varq_flash.h"
#include "varq_flash_fwd_kernel.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires sm80 or newer.\n");

template<typename Kernel_traits, bool Is_even_MN>
__global__ void varq_flash_fwd_kernel(KERNEL_PARAM_MODIFIER const Varq_fwd_params params) {
#if defined(ARCH_SUPPORTS_FLASH)
    flash::compute_attn<Kernel_traits,
                        /*Is_dropout=*/false,
                        /*Is_causal=*/false,
                        /*Is_local=*/false,
                        /*Has_alibi=*/false,
                        Is_even_MN,
                        /*Is_even_K=*/true,
                        /*Is_softcap=*/false,
                        /*Return_softmax=*/false,
                        Varq_fwd_params>(params);
#else
    FLASH_UNSUPPORTED_ARCH
#endif
}

template<typename Kernel_traits>
void run_varq_flash_fwd(Varq_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    params.varq_block_n = Kernel_traits::kBlockN;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_mn = params.seqlen_k % Kernel_traits::kBlockN == 0
        && params.seqlen_q % Kernel_traits::kBlockM == 0;
    if (is_even_mn) {
        auto kernel = &varq_flash_fwd_kernel<Kernel_traits, true>;
        if (smem_size >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    } else {
        auto kernel = &varq_flash_fwd_kernel<Kernel_traits, false>;
        if (smem_size >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<>
void run_varq_mha_fwd_<cutlass::half_t, 128, false>(Varq_fwd_params &params, cudaStream_t stream) {
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    const bool is_sm8x = cc_major == 8 && cc_minor > 0;
    if (is_sm8x) {
        run_varq_flash_fwd<Flash_fwd_kernel_traits<128, 128, 32, 4, false, false, cutlass::half_t>>(params, stream);
    } else {
        run_varq_flash_fwd<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::half_t>>(params, stream);
    }
}

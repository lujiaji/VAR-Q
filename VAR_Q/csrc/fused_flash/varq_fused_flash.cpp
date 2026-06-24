#include <torch/extension.h>

#include <string>

at::Tensor fwd(
    at::Tensor q,
    at::Tensor k_packed,
    at::Tensor v_packed,
    at::Tensor k_scale,
    at::Tensor v_scale,
    at::Tensor step_ids,
    at::Tensor k_fresh,
    at::Tensor v_fresh);

at::Tensor fwd_direct(
    at::Tensor q,
    at::Tensor k_packed,
    at::Tensor v_packed,
    at::Tensor k_scale,
    at::Tensor v_scale,
    at::Tensor step_ids,
    at::Tensor k_fresh,
    at::Tensor v_fresh);

std::string backend_info();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd, "VAR-Q fused dequant FlashAttention forward");
    m.def("fwd_direct", &fwd_direct, "VAR-Q direct q8/fp16 FlashAttention forward");
    m.def("backend_info", &backend_info, "Return build/runtime metadata");
}

#include <torch/extension.h>

#include <c10/util/Optional.h>

#include <string>
#include <tuple>

at::Tensor fwd(
    at::Tensor q,
    at::Tensor k_packed,
    at::Tensor v_packed,
    at::Tensor k_scale,
    at::Tensor v_scale,
    at::Tensor step_ids,
    at::Tensor k_fresh,
    at::Tensor v_fresh,
    double softmax_scale);

at::Tensor fwd_direct(
    at::Tensor q,
    at::Tensor k_packed,
    at::Tensor v_packed,
    at::Tensor k_scale,
    at::Tensor v_scale,
    at::Tensor step_ids,
    at::Tensor k_fresh,
    at::Tensor v_fresh,
    double softmax_scale);

std::tuple<at::Tensor, at::Tensor> quantize_pack(
    at::Tensor x,
    int64_t bits,
    c10::optional<at::Tensor> scale,
    c10::optional<at::Tensor> group_ids,
    int64_t scale_dtype,
    int64_t layout);

at::Tensor pack_int8(at::Tensor q_int8, int64_t bits);

at::Tensor unpack_int8(at::Tensor packed, int64_t bits, int64_t orig_dim);

at::Tensor unpack_dequant(
    at::Tensor packed,
    at::Tensor scale,
    int64_t bits,
    int64_t orig_dim,
    int64_t output_dtype,
    c10::optional<at::Tensor> group_ids,
    int64_t layout,
    c10::optional<at::Tensor> out);

std::string backend_info();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd, "VAR-Q fused dequant FlashAttention forward", py::arg("q"),
          py::arg("k_packed"), py::arg("v_packed"), py::arg("k_scale"),
          py::arg("v_scale"), py::arg("step_ids"), py::arg("k_fresh"),
          py::arg("v_fresh"), py::arg("softmax_scale"));
    m.def("fwd_direct", &fwd_direct, "VAR-Q direct packed q8/q4/q3/q2 FlashAttention forward",
          py::arg("q"), py::arg("k_packed"), py::arg("v_packed"), py::arg("k_scale"),
          py::arg("v_scale"), py::arg("step_ids"), py::arg("k_fresh"),
          py::arg("v_fresh"), py::arg("softmax_scale"));
    m.def("quantize_pack", &quantize_pack,
          "CUDA symmetric quantization followed by int32 packing",
          py::arg("x"), py::arg("bits"), py::arg("scale") = c10::nullopt,
          py::arg("group_ids") = c10::nullopt, py::arg("scale_dtype") = -1,
          py::arg("layout") = 0);
    m.def("pack_int8", &pack_int8, "CUDA pack signed int8 values into int32 words",
          py::arg("q_int8"), py::arg("bits"));
    m.def("unpack_int8", &unpack_int8, "CUDA unpack int32 words into signed int8 values",
          py::arg("packed"), py::arg("bits"), py::arg("orig_dim") = -1);
    m.def("unpack_dequant", &unpack_dequant,
          "CUDA unpack and broadcast-scale dequantization",
          py::arg("packed"), py::arg("scale"), py::arg("bits"),
          py::arg("orig_dim") = -1, py::arg("output_dtype") = -1,
          py::arg("group_ids") = c10::nullopt, py::arg("layout") = 0,
          py::arg("out") = c10::nullopt);
    m.def("backend_info", &backend_info, "Return build/runtime metadata");
}

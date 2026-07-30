#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> tmix_mix6_forward_v5_cuda(
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);

std::vector<torch::Tensor> tmix_mix6_backward_v5_cuda(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);

std::vector<torch::Tensor> tmix_mix6_state_forward_v5_cuda(
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);

std::vector<torch::Tensor> tmix_mix6_state_backward_v5_cuda(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor grad_new_shift,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);

namespace {

void check_bf16_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

void check_vec(const torch::Tensor& x, int64_t c, const char* name) {
    TORCH_CHECK(x.dim() == 1, name, " must have shape [C]");
    TORCH_CHECK(x.size(0) == c, name, " shape mismatch");
}

void check_shift_state(const torch::Tensor& shift_state, int64_t b, int64_t c, const char* name) {
    check_bf16_cuda(shift_state, name);
    TORCH_CHECK(shift_state.dim() == 2, name, " must have shape [B, C]");
    TORCH_CHECK(shift_state.size(0) == b, name, " batch size mismatch");
    TORCH_CHECK(shift_state.size(1) == c, name, " channel size mismatch");
}

} // namespace

std::vector<torch::Tensor> forward(
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    check_bf16_cuda(x, "x");
    check_bf16_cuda(x_r, "x_r");
    check_bf16_cuda(x_w, "x_w");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(x_v, "x_v");
    check_bf16_cuda(x_a, "x_a");
    check_bf16_cuda(x_g, "x_g");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
    int64_t c = x.size(2);
    TORCH_CHECK((c % 2) == 0, "tmix_mix6_v5 currently requires even C");
    check_vec(x_r, c, "x_r");
    check_vec(x_w, c, "x_w");
    check_vec(x_k, c, "x_k");
    check_vec(x_v, c, "x_v");
    check_vec(x_a, c, "x_a");
    check_vec(x_g, c, "x_g");
    return tmix_mix6_forward_v5_cuda(x, x_r, x_w, x_k, x_v, x_a, x_g);
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    check_bf16_cuda(grad_r, "grad_r");
    check_bf16_cuda(grad_w, "grad_w");
    check_bf16_cuda(grad_k, "grad_k");
    check_bf16_cuda(grad_v, "grad_v");
    check_bf16_cuda(grad_a, "grad_a");
    check_bf16_cuda(grad_g, "grad_g");
    check_bf16_cuda(x, "x");
    check_bf16_cuda(x_r, "x_r");
    check_bf16_cuda(x_w, "x_w");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(x_v, "x_v");
    check_bf16_cuda(x_a, "x_a");
    check_bf16_cuda(x_g, "x_g");
    TORCH_CHECK(grad_r.sizes() == x.sizes(), "grad_r shape mismatch");
    TORCH_CHECK(grad_w.sizes() == x.sizes(), "grad_w shape mismatch");
    TORCH_CHECK(grad_k.sizes() == x.sizes(), "grad_k shape mismatch");
    TORCH_CHECK(grad_v.sizes() == x.sizes(), "grad_v shape mismatch");
    TORCH_CHECK(grad_a.sizes() == x.sizes(), "grad_a shape mismatch");
    TORCH_CHECK(grad_g.sizes() == x.sizes(), "grad_g shape mismatch");
    TORCH_CHECK((x.size(2) % 2) == 0, "tmix_mix6_v5 currently requires even C");
    return tmix_mix6_backward_v5_cuda(grad_r, grad_w, grad_k, grad_v, grad_a, grad_g, x, x_r, x_w, x_k, x_v, x_a, x_g);
}

std::vector<torch::Tensor> state_forward(
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    check_bf16_cuda(x, "x");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
    int64_t b = x.size(0);
    int64_t c = x.size(2);
    TORCH_CHECK((c % 2) == 0, "tmix_mix6_v5 currently requires even C");
    check_shift_state(shift_state, b, c, "shift_state");
    check_bf16_cuda(x_r, "x_r");
    check_bf16_cuda(x_w, "x_w");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(x_v, "x_v");
    check_bf16_cuda(x_a, "x_a");
    check_bf16_cuda(x_g, "x_g");
    check_vec(x_r, c, "x_r");
    check_vec(x_w, c, "x_w");
    check_vec(x_k, c, "x_k");
    check_vec(x_v, c, "x_v");
    check_vec(x_a, c, "x_a");
    check_vec(x_g, c, "x_g");
    return tmix_mix6_state_forward_v5_cuda(x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g);
}

std::vector<torch::Tensor> state_backward(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor grad_new_shift,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    check_bf16_cuda(grad_r, "grad_r");
    check_bf16_cuda(grad_w, "grad_w");
    check_bf16_cuda(grad_k, "grad_k");
    check_bf16_cuda(grad_v, "grad_v");
    check_bf16_cuda(grad_a, "grad_a");
    check_bf16_cuda(grad_g, "grad_g");
    check_bf16_cuda(grad_new_shift, "grad_new_shift");
    check_bf16_cuda(x, "x");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
    int64_t b = x.size(0);
    int64_t c = x.size(2);
    check_shift_state(shift_state, b, c, "shift_state");
    TORCH_CHECK(grad_new_shift.sizes() == shift_state.sizes(), "grad_new_shift shape mismatch");
    check_bf16_cuda(x_r, "x_r");
    check_bf16_cuda(x_w, "x_w");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(x_v, "x_v");
    check_bf16_cuda(x_a, "x_a");
    check_bf16_cuda(x_g, "x_g");
    TORCH_CHECK(grad_r.sizes() == x.sizes(), "grad_r shape mismatch");
    TORCH_CHECK(grad_w.sizes() == x.sizes(), "grad_w shape mismatch");
    TORCH_CHECK(grad_k.sizes() == x.sizes(), "grad_k shape mismatch");
    TORCH_CHECK(grad_v.sizes() == x.sizes(), "grad_v shape mismatch");
    TORCH_CHECK(grad_a.sizes() == x.sizes(), "grad_a shape mismatch");
    TORCH_CHECK(grad_g.sizes() == x.sizes(), "grad_g shape mismatch");
    TORCH_CHECK((c % 2) == 0, "tmix_mix6_v5 currently requires even C");
    return tmix_mix6_state_backward_v5_cuda(
        grad_r, grad_w, grad_k, grad_v, grad_a, grad_g, grad_new_shift,
        x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g);
}

TORCH_LIBRARY(rwkv7_tmix_mix6_bf16_v5, m) {
    m.def("forward(Tensor x, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
    m.def("backward(Tensor grad_r, Tensor grad_w, Tensor grad_k, Tensor grad_v, Tensor grad_a, Tensor grad_g, Tensor x, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
    m.def("state_forward(Tensor x, Tensor shift_state, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
    m.def("state_backward(Tensor grad_r, Tensor grad_w, Tensor grad_k, Tensor grad_v, Tensor grad_a, Tensor grad_g, Tensor grad_new_shift, Tensor x, Tensor shift_state, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_tmix_mix6_bf16_v5, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
    m.impl("state_forward", &state_forward);
    m.impl("state_backward", &state_backward);
}

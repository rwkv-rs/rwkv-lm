#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> cmix_layer_forward_v5_cuda(
    torch::Tensor x,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight);
std::vector<torch::Tensor> cmix_layer_backward_v5_cuda(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight,
    torch::Tensor mixed,
    torch::Tensor act);
torch::Tensor cmix_mix_forward_v5_cuda(torch::Tensor x, torch::Tensor x_k);
std::vector<torch::Tensor> cmix_mix_backward_v5_cuda(torch::Tensor grad_out, torch::Tensor x, torch::Tensor x_k);
std::vector<torch::Tensor> cmix_layer_state_forward_v5_cuda(
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight);
std::vector<torch::Tensor> cmix_layer_state_backward_v5_cuda(
    torch::Tensor grad_out,
    torch::Tensor grad_new_shift,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight,
    torch::Tensor mixed,
    torch::Tensor act);

namespace {

void check_bf16_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

void check_shift_state(const torch::Tensor& shift_state, int64_t b, int64_t c, const char* name) {
    check_bf16_cuda(shift_state, name);
    TORCH_CHECK(shift_state.dim() == 2, name, " must have shape [B, C]");
    TORCH_CHECK(shift_state.size(0) == b, name, " batch size mismatch");
    TORCH_CHECK(shift_state.size(1) == c, name, " channel size mismatch");
}

} // namespace

std::vector<torch::Tensor> cmix_layer_forward_v5(
    torch::Tensor x,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight) {
    check_bf16_cuda(x, "x");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(key_weight, "key_weight");
    check_bf16_cuda(value_weight, "value_weight");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
    TORCH_CHECK(x_k.dim() == 1, "x_k must have shape [C]");
    TORCH_CHECK(key_weight.dim() == 2, "key_weight must have shape [4C, C]");
    TORCH_CHECK(value_weight.dim() == 2, "value_weight must have shape [C, 4C]");
    TORCH_CHECK((x.size(2) % 2) == 0, "cmix v5 currently requires even C");
    TORCH_CHECK(x.size(2) == x_k.size(0), "channel size mismatch for x_k");
    TORCH_CHECK(key_weight.size(1) == x.size(2), "key_weight input dim mismatch");
    TORCH_CHECK(value_weight.size(1) == key_weight.size(0), "value_weight input dim mismatch");
    return cmix_layer_forward_v5_cuda(x, x_k, key_weight, value_weight);
}

std::vector<torch::Tensor> cmix_layer_backward_v5(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight,
    torch::Tensor mixed,
    torch::Tensor act) {
    check_bf16_cuda(grad_out, "grad_out");
    check_bf16_cuda(x, "x");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(key_weight, "key_weight");
    check_bf16_cuda(value_weight, "value_weight");
    check_bf16_cuda(mixed, "mixed");
    check_bf16_cuda(act, "act");
    TORCH_CHECK((x.size(2) % 2) == 0, "cmix v5 currently requires even C");
    return cmix_layer_backward_v5_cuda(grad_out, x, x_k, key_weight, value_weight, mixed, act);
}

std::vector<torch::Tensor> cmix_layer_state_forward_v5(
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight) {
    check_bf16_cuda(x, "x");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(key_weight, "key_weight");
    check_bf16_cuda(value_weight, "value_weight");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
    TORCH_CHECK(x_k.dim() == 1, "x_k must have shape [C]");
    TORCH_CHECK(key_weight.dim() == 2, "key_weight must have shape [4C, C]");
    TORCH_CHECK(value_weight.dim() == 2, "value_weight must have shape [C, 4C]");
    TORCH_CHECK((x.size(2) % 2) == 0, "cmix v5 currently requires even C");
    TORCH_CHECK(x.size(2) == x_k.size(0), "channel size mismatch for x_k");
    check_shift_state(shift_state, x.size(0), x.size(2), "shift_state");
    TORCH_CHECK(key_weight.size(1) == x.size(2), "key_weight input dim mismatch");
    TORCH_CHECK(value_weight.size(1) == key_weight.size(0), "value_weight input dim mismatch");
    return cmix_layer_state_forward_v5_cuda(x, shift_state, x_k, key_weight, value_weight);
}

std::vector<torch::Tensor> cmix_layer_state_backward_v5(
    torch::Tensor grad_out,
    torch::Tensor grad_new_shift,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight,
    torch::Tensor mixed,
    torch::Tensor act) {
    check_bf16_cuda(grad_out, "grad_out");
    check_bf16_cuda(grad_new_shift, "grad_new_shift");
    check_bf16_cuda(x, "x");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(key_weight, "key_weight");
    check_bf16_cuda(value_weight, "value_weight");
    check_bf16_cuda(mixed, "mixed");
    check_bf16_cuda(act, "act");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
    check_shift_state(shift_state, x.size(0), x.size(2), "shift_state");
    TORCH_CHECK(grad_new_shift.sizes() == shift_state.sizes(), "grad_new_shift shape mismatch");
    TORCH_CHECK((x.size(2) % 2) == 0, "cmix v5 currently requires even C");
    return cmix_layer_state_backward_v5_cuda(
        grad_out, grad_new_shift, x, shift_state, x_k, key_weight, value_weight, mixed, act);
}

TORCH_LIBRARY(rwkv7_cmix_bf16_v5, m) {
    m.def("forward", cmix_layer_forward_v5);
    m.def("backward", cmix_layer_backward_v5);
    m.def("state_forward", cmix_layer_state_forward_v5);
    m.def("state_backward", cmix_layer_state_backward_v5);
}

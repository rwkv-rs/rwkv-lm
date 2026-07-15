#include <torch/extension.h>

#ifdef _FP32_
using bf = float;
#else
#include <cuda_bf16.h>
using bf = __nv_bfloat16;
#endif

void cuda_forward(
    int B,
    int T,
    int H,
    float* s0,
    bf* r,
    bf* w,
    bf* k,
    bf* v,
    bf* a,
    bf* b,
    bf* y,
    float* sT,
    float* s,
    float* sa);

void cuda_backward(
    int B,
    int T,
    int H,
    bf* r,
    bf* w,
    bf* k,
    bf* v,
    bf* a,
    bf* b,
    bf* dy,
    float* dsT,
    float* s,
    float* sa,
    float* ds0,
    bf* dr,
    bf* dw,
    bf* dk,
    bf* dv,
    bf* da,
    bf* db);

namespace {

void check_vectors(
    const torch::Tensor& state,
    const std::vector<torch::Tensor>& values) {
  TORCH_CHECK(state.is_cuda(), "RWKV7 state must be CUDA");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "RWKV7 state must be FP32");
  TORCH_CHECK(state.is_contiguous(), "RWKV7 state must be contiguous");
  TORCH_CHECK(state.dim() == 4, "RWKV7 state must be [B,H,N,N]");
  for (const auto& value : values) {
    TORCH_CHECK(value.is_cuda(), "RWKV7 vectors must be CUDA");
    TORCH_CHECK(value.scalar_type() == torch::kBFloat16, "RWKV7 vectors must be BF16");
    TORCH_CHECK(value.is_contiguous(), "RWKV7 vectors must be contiguous");
    TORCH_CHECK(value.dim() == 4, "RWKV7 vectors must be [B,T,H,N]");
    TORCH_CHECK(value.sizes() == values.front().sizes(), "RWKV7 vector shapes differ");
  }
  const auto B = values.front().size(0);
  const auto H = values.front().size(2);
  const auto N = values.front().size(3);
  TORCH_CHECK(
      state.size(0) == B && state.size(1) == H && state.size(2) == N &&
          state.size(3) == N,
      "RWKV7 state/vector geometry mismatch");
}

}  // namespace

std::vector<torch::Tensor> statepassing_forward(
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b) {
  std::vector<torch::Tensor> values{r, w, k, v, a, b};
  check_vectors(state, values);
  const int B = static_cast<int>(r.size(0));
  const int T = static_cast<int>(r.size(1));
  const int H = static_cast<int>(r.size(2));
  const int N = static_cast<int>(r.size(3));
  TORCH_CHECK(N == _N_, "RWKV7 vector head size differs from compiled _N_");
  TORCH_CHECK(T % _CHUNK_LEN_ == 0, "RWKV7 token count must divide chunk length");

  auto y = torch::empty_like(v);
  auto final_state = torch::empty_like(state);
  auto snapshots = torch::empty(
      {B, H, T / _CHUNK_LEN_, N, N}, state.options());
  auto sa = torch::empty({B, T, H, N}, state.options());
  cuda_forward(
      B,
      T,
      H,
      static_cast<float*>(state.data_ptr()),
      reinterpret_cast<bf*>(r.data_ptr()),
      reinterpret_cast<bf*>(w.data_ptr()),
      reinterpret_cast<bf*>(k.data_ptr()),
      reinterpret_cast<bf*>(v.data_ptr()),
      reinterpret_cast<bf*>(a.data_ptr()),
      reinterpret_cast<bf*>(b.data_ptr()),
      reinterpret_cast<bf*>(y.data_ptr()),
      static_cast<float*>(final_state.data_ptr()),
      static_cast<float*>(snapshots.data_ptr()),
      static_cast<float*>(sa.data_ptr()));
  return {y, final_state, snapshots, sa};
}

std::vector<torch::Tensor> statepassing_backward(
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y,
    torch::Tensor grad_final_state,
    torch::Tensor snapshots,
    torch::Tensor sa) {
  TORCH_CHECK(grad_y.is_contiguous(), "RWKV7 output gradient must be contiguous");
  TORCH_CHECK(
      grad_final_state.is_contiguous(),
      "RWKV7 final-state gradient must be contiguous");
  const int B = static_cast<int>(r.size(0));
  const int T = static_cast<int>(r.size(1));
  const int H = static_cast<int>(r.size(2));
  auto grad_state = torch::empty_like(grad_final_state);
  auto dr = torch::empty_like(r);
  auto dw = torch::empty_like(w);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);
  auto da = torch::empty_like(a);
  auto db = torch::empty_like(b);
  cuda_backward(
      B,
      T,
      H,
      reinterpret_cast<bf*>(r.data_ptr()),
      reinterpret_cast<bf*>(w.data_ptr()),
      reinterpret_cast<bf*>(k.data_ptr()),
      reinterpret_cast<bf*>(v.data_ptr()),
      reinterpret_cast<bf*>(a.data_ptr()),
      reinterpret_cast<bf*>(b.data_ptr()),
      reinterpret_cast<bf*>(grad_y.data_ptr()),
      static_cast<float*>(grad_final_state.data_ptr()),
      static_cast<float*>(snapshots.data_ptr()),
      static_cast<float*>(sa.data_ptr()),
      static_cast<float*>(grad_state.data_ptr()),
      reinterpret_cast<bf*>(dr.data_ptr()),
      reinterpret_cast<bf*>(dw.data_ptr()),
      reinterpret_cast<bf*>(dk.data_ptr()),
      reinterpret_cast<bf*>(dv.data_ptr()),
      reinterpret_cast<bf*>(da.data_ptr()),
      reinterpret_cast<bf*>(db.data_ptr()));
  return {grad_state, dr, dw, dk, dv, da, db};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &statepassing_forward);
  module.def("backward", &statepassing_backward);
}

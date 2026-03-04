#include <llm/nn.hpp>
#include <llm/ops.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace llm {

Tensor scaled_dot_product_attention(const Tensor& Q,
                                    const Tensor& K,
                                    const Tensor& V,
                                    bool causal) {
  if (Q.dtype() != DType::Float32 || K.dtype() != DType::Float32 || V.dtype() != DType::Float32)
    throw std::invalid_argument("scaled_dot_product_attention: Q, K, V must be float32");
  if (Q.dim() != 2 || K.dim() != 2 || V.dim() != 2)
    throw std::invalid_argument("scaled_dot_product_attention: Q, K, V must be 2D (T, D)");
  const auto& qsh = Q.shape();
  const auto& ksh = K.shape();
  const auto& vsh = V.shape();
  if (qsh != ksh || ksh != vsh)
    throw std::invalid_argument("scaled_dot_product_attention: Q, K, V must have same shape");

  const int64_t T = qsh[0];
  const int64_t D = qsh[1];
  const float scale = 1.f / std::sqrt(static_cast<float>(D));

  // scores = Q @ K^T  -> (T, T)
  Tensor K_t = transpose(K);
  Tensor scores_raw = matmul(Q, K_t);

  // scale
  std::vector<float> scale_data(static_cast<size_t>(T * T), scale);
  Tensor scale_const = Tensor::from_data(scale_data, {T, T}, false);
  Tensor scores_scaled = mul(scores_raw, scale_const);

  // causal mask: (i, j) with j > i -> -inf so softmax gives 0
  std::vector<float> mask_data(static_cast<size_t>(T * T));
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = 0; j < T; ++j)
      mask_data[static_cast<size_t>(i * T + j)] = (j > i && causal) ? neg_inf : 0.f;
  Tensor mask = Tensor::from_data(mask_data, {T, T}, false);
  Tensor scores_masked = add(scores_scaled, mask);

  Tensor attn = softmax(scores_masked);
  Tensor out = matmul(attn, V);
  return out;
}

Tensor ScaledDotProductAttention::operator()(const Tensor& Q,
                                             const Tensor& K,
                                             const Tensor& V,
                                             bool causal) {
  return scaled_dot_product_attention(Q, K, V, causal);
}

}  // namespace llm

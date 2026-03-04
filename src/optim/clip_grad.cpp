#include <llm/optim.hpp>

#include <cmath>

namespace llm {

float clip_grad_norm_(std::vector<Parameter*>& params, float max_norm) {
  double total_sq = 0.0;
  for (Parameter* p : params) {
    if (!p || p->dtype() != DType::Float32) continue;
    std::shared_ptr<Tensor> g = p->grad();
    if (!g) continue;
    const float* gw = g->data_float();
    const int64_t n = p->numel();
    for (int64_t i = 0; i < n; ++i) {
      const double x = static_cast<double>(gw[i]);
      total_sq += x * x;
    }
  }
  const float total_norm = static_cast<float>(std::sqrt(total_sq));
  if (total_norm <= max_norm || total_norm <= 0.f) return total_norm;

  const float scale = max_norm / total_norm;
  for (Parameter* p : params) {
    if (!p || p->dtype() != DType::Float32) continue;
    std::shared_ptr<Tensor> g = p->grad();
    if (!g) continue;
    float* gw = g->data_float();
    const int64_t n = p->numel();
    for (int64_t i = 0; i < n; ++i)
      gw[i] *= scale;
  }
  return total_norm;
}

}  // namespace llm

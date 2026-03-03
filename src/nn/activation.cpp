#include <llm/nn.hpp>
#include <llm/autograd.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>

namespace llm {

namespace {

// GELU tanh approximation:
// y = 0.5 x (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
// dy/dx = 0.5(1 + t) + 0.5 x (1 - t^2) * a' where t=tanh(a),
// a = k*(x + c x^3), a' = k*(1 + 3c x^2), k=sqrt(2/pi), c=0.044715.
inline float gelu_tanh(float x) {
  constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
  constexpr float c = 0.044715f;
  float x3 = x * x * x;
  float a = k * (x + c * x3);
  float t = std::tanh(a);
  return 0.5f * x * (1.0f + t);
}

inline float gelu_tanh_grad(float x) {
  constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
  constexpr float c = 0.044715f;
  float x2 = x * x;
  float x3 = x2 * x;
  float a = k * (x + c * x3);
  float t = std::tanh(a);
  float ap = k * (1.0f + 3.0f * c * x2);
  float dt = (1.0f - t * t) * ap;
  return 0.5f * (1.0f + t) + 0.5f * x * dt;
}

class GeluBackward : public AutogradNode {
public:
  GeluBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> x_val)
      : x_(std::move(x)), x_val_(std::move(x_val)) {}

  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {x_};
  }

  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (!x_ || !x_->requires_grad()) return;
    NoGradGuard guard;
    const float* go = grad_output->data_float();
    const float* xv = x_val_->data_float();

    Tensor gx(x_->shape(), DType::Float32, x_->device(), false);
    float* pgx = gx.data_float();
    for (int64_t i = 0; i < gx.numel(); ++i) {
      pgx[i] = go[i] * gelu_tanh_grad(xv[i]);
    }
    x_->accumulate_grad(gx);
  }

private:
  std::shared_ptr<Tensor> x_;
  std::shared_ptr<Tensor> x_val_;
};

}  // namespace

Tensor gelu(const Tensor& x) {
  if (x.dtype() != DType::Float32)
    throw std::invalid_argument("gelu: input must be float32");

  Tensor out(x.shape(), DType::Float32, x.device(), false);
  const float* px = x.data_float();
  float* po = out.data_float();
  for (int64_t i = 0; i < out.numel(); ++i) {
    po[i] = gelu_tanh(px[i]);
  }

  if (is_grad_enabled() && x.requires_grad()) {
    out.set_requires_grad(true);
    auto x_copy = std::make_shared<Tensor>(x);
    auto node = std::make_shared<GeluBackward>(x_copy, x_copy);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor GELU::operator()(const Tensor& x) {
  return gelu(x);
}

}  // namespace llm


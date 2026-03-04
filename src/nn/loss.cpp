#include <llm/nn.hpp>
#include <llm/autograd.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace llm {

namespace {

class CrossEntropyBackward : public AutogradNode {
public:
  CrossEntropyBackward(std::shared_ptr<Tensor> logits,
                       std::shared_ptr<Tensor> probs,
                       std::shared_ptr<Tensor> targets,
                       int64_t N,
                       int64_t C)
      : logits_(std::move(logits)),
        probs_(std::move(probs)),
        targets_(std::move(targets)),
        N_(N),
        C_(C) {}

  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {logits_};
  }

  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (!logits_ || !logits_->requires_grad()) return;
    NoGradGuard guard;

    if (grad_output->dtype() != DType::Float32 || grad_output->numel() != 1)
      throw std::runtime_error("cross_entropy backward: grad_output must be scalar float32");

    const float upstream = grad_output->data_float()[0];
    const float invN = upstream / static_cast<float>(N_);

    const float* p = probs_->data_float();       // (N,C)
    const int64_t* t = targets_->data_int64();   // (N)

    Tensor g({N_, C_}, DType::Float32, logits_->device(), false);
    float* pg = g.data_float();

    // grad = (probs - one_hot(target)) / N
    for (int64_t i = 0; i < N_; ++i) {
      int64_t yi = t[i];
      for (int64_t j = 0; j < C_; ++j) {
        float v = p[i * C_ + j];
        if (j == yi) v -= 1.f;
        pg[i * C_ + j] = v * invN;
      }
    }

    logits_->accumulate_grad(g);
  }

private:
  std::shared_ptr<Tensor> logits_;
  std::shared_ptr<Tensor> probs_;     // softmax probs
  std::shared_ptr<Tensor> targets_;   // int64 targets
  int64_t N_, C_;
};

}  // namespace

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
  if (logits.dtype() != DType::Float32)
    throw std::invalid_argument("cross_entropy: logits must be float32");
  if (targets.dtype() != DType::Int64)
    throw std::invalid_argument("cross_entropy: targets must be int64");
  if (logits.dim() != 2)
    throw std::invalid_argument("cross_entropy: logits must be 2D (N, C)");
  if (targets.dim() != 1)
    throw std::invalid_argument("cross_entropy: targets must be 1D (N)");
  const int64_t N = logits.shape()[0];
  const int64_t C = logits.shape()[1];
  if (targets.shape()[0] != N)
    throw std::invalid_argument("cross_entropy: targets length must equal N");

  const float* x = logits.data_float();
  const int64_t* t = targets.data_int64();

  // We'll compute stable softmax probabilities and the mean negative log-likelihood.
  auto probs = std::make_shared<Tensor>(std::vector<int64_t>{N, C},
                                        DType::Float32, logits.device(), false);
  float* p = probs->data_float();

  float total = 0.f;
  for (int64_t i = 0; i < N; ++i) {
    // max for numerical stability
    float m = x[i * C];
    for (int64_t j = 1; j < C; ++j) {
      float v = x[i * C + j];
      if (v > m) m = v;
    }

    // exp and sum
    float sum_exp = 0.f;
    for (int64_t j = 0; j < C; ++j) {
      float e = std::exp(x[i * C + j] - m);
      p[i * C + j] = e;
      sum_exp += e;
    }

    // normalize to probs
    float inv = 1.f / sum_exp;
    for (int64_t j = 0; j < C; ++j)
      p[i * C + j] *= inv;

    int64_t yi = t[i];
    if (yi < 0 || yi >= C)
      throw std::out_of_range("cross_entropy: target out of range");

    // -log(prob[target])
    float pt = p[i * C + yi];
    // Guard against log(0) in pathological cases.
    if (pt <= 0.f) pt = 1e-12f;
    total += -std::log(pt);
  }

  Tensor out({1}, DType::Float32, logits.device(), false);
  out.data_float()[0] = total / static_cast<float>(N);

  if (is_grad_enabled() && logits.requires_grad()) {
    out.set_requires_grad(true);
    auto node = std::make_shared<CrossEntropyBackward>(
        std::make_shared<Tensor>(logits),
        probs,
        std::make_shared<Tensor>(targets),
        N, C);
    out.set_grad_fn(node);
  }

  return out;
}

Tensor CrossEntropyLoss::operator()(const Tensor& logits, const Tensor& targets) {
  return cross_entropy(logits, targets);
}

}  // namespace llm


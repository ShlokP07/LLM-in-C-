#include <llm/nn.hpp>
#include <llm/autograd.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>

namespace llm {

namespace {

class SoftmaxBackward : public AutogradNode {
public:
  SoftmaxBackward(std::shared_ptr<Tensor> x,
                  std::shared_ptr<Tensor> y,
                  int64_t N,
                  int64_t D)
      : x_(std::move(x)), y_(std::move(y)), N_(N), D_(D) {}

  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {x_};
  }

  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (!x_ || !x_->requires_grad()) return;
    NoGradGuard guard;

    const float* go = grad_output->data_float();
    const float* y = y_->data_float();

    Tensor gx({N_, D_}, DType::Float32, x_->device(), false);
    float* pgx = gx.data_float();

    for (int64_t i = 0; i < N_; ++i) {
      // dot = sum_j go[i,j] * y[i,j]
      float dot = 0.f;
      for (int64_t j = 0; j < D_; ++j)
        dot += go[i * D_ + j] * y[i * D_ + j];

      for (int64_t j = 0; j < D_; ++j) {
        float yi = y[i * D_ + j];
        pgx[i * D_ + j] = yi * (go[i * D_ + j] - dot);
      }
    }

    x_->accumulate_grad(gx);
  }

private:
  std::shared_ptr<Tensor> x_;
  std::shared_ptr<Tensor> y_;  // softmax output
  int64_t N_, D_;
};

class LogSoftmaxBackward : public AutogradNode {
public:
  LogSoftmaxBackward(std::shared_ptr<Tensor> x,
                     std::shared_ptr<Tensor> y,
                     int64_t N,
                     int64_t D)
      : x_(std::move(x)), y_(std::move(y)), N_(N), D_(D) {}

  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {x_};
  }

  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (!x_ || !x_->requires_grad()) return;
    NoGradGuard guard;

    const float* go = grad_output->data_float();
    const float* y = y_->data_float();  // log-softmax output

    Tensor gx({N_, D_}, DType::Float32, x_->device(), false);
    float* pgx = gx.data_float();

    for (int64_t i = 0; i < N_; ++i) {
      float sum_go = 0.f;
      for (int64_t j = 0; j < D_; ++j)
        sum_go += go[i * D_ + j];

      for (int64_t j = 0; j < D_; ++j) {
        float soft = std::exp(y[i * D_ + j]);  // softmax = exp(log_softmax)
        pgx[i * D_ + j] = go[i * D_ + j] - soft * sum_go;
      }
    }

    x_->accumulate_grad(gx);
  }

private:
  std::shared_ptr<Tensor> x_;
  std::shared_ptr<Tensor> y_;  // log-softmax output
  int64_t N_, D_;
};

}  // namespace

Tensor softmax(const Tensor& x) {
  if (x.dtype() != DType::Float32)
    throw std::invalid_argument("softmax: input must be float32");
  if (x.dim() != 2)
    throw std::invalid_argument("softmax: only 2D input (N, D) supported");

  const int64_t N = x.shape()[0];
  const int64_t D = x.shape()[1];
  const float* px = x.data_float();

  Tensor out({N, D}, DType::Float32, x.device(), false);
  float* po = out.data_float();

  for (int64_t i = 0; i < N; ++i) {
    // max for numerical stability
    float m = px[i * D];
    for (int64_t j = 1; j < D; ++j) {
      float v = px[i * D + j];
      if (v > m) m = v;
    }

    // exp and sum
    float sum_exp = 0.f;
    for (int64_t j = 0; j < D; ++j) {
      float e = std::exp(px[i * D + j] - m);
      po[i * D + j] = e;
      sum_exp += e;
    }

    // normalize
    float inv = 1.f / sum_exp;
    for (int64_t j = 0; j < D; ++j)
      po[i * D + j] *= inv;
  }

  if (is_grad_enabled() && x.requires_grad()) {
    out.set_requires_grad(true);
    auto x_copy = std::make_shared<Tensor>(x);
    auto y_copy = std::make_shared<Tensor>(out);
    out.set_grad_fn(std::make_shared<SoftmaxBackward>(x_copy, y_copy, N, D));
  }
  return out;
}

Tensor log_softmax(const Tensor& x) {
  if (x.dtype() != DType::Float32)
    throw std::invalid_argument("log_softmax: input must be float32");
  if (x.dim() != 2)
    throw std::invalid_argument("log_softmax: only 2D input (N, D) supported");

  const int64_t N = x.shape()[0];
  const int64_t D = x.shape()[1];
  const float* px = x.data_float();

  Tensor out({N, D}, DType::Float32, x.device(), false);
  float* po = out.data_float();

  for (int64_t i = 0; i < N; ++i) {
    // max for numerical stability
    float m = px[i * D];
    for (int64_t j = 1; j < D; ++j) {
      float v = px[i * D + j];
      if (v > m) m = v;
    }

    float sum_exp = 0.f;
    for (int64_t j = 0; j < D; ++j)
      sum_exp += std::exp(px[i * D + j] - m);

    float lse = m + std::log(sum_exp);
    for (int64_t j = 0; j < D; ++j)
      po[i * D + j] = px[i * D + j] - lse;
  }

  if (is_grad_enabled() && x.requires_grad()) {
    out.set_requires_grad(true);
    auto x_copy = std::make_shared<Tensor>(x);
    auto y_copy = std::make_shared<Tensor>(out);
    out.set_grad_fn(std::make_shared<LogSoftmaxBackward>(x_copy, y_copy, N, D));
  }
  return out;
}

Tensor Softmax::operator()(const Tensor& x) {
  return softmax(x);
}

Tensor LogSoftmax::operator()(const Tensor& x) {
  return log_softmax(x);
}

}  // namespace llm


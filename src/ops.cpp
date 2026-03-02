#include <llm/ops.hpp>
#include <llm/autograd.hpp>

#include <cstring>
#include <stdexcept>

namespace llm {

namespace {

void expect_float32(const Tensor& t, const char* name) {
  if (t.dtype() != DType::Float32)
    throw std::invalid_argument(std::string(name) + ": expected float32");
}

void expect_same_shape(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape())
    throw std::invalid_argument("ops: shape mismatch");
}

// --- Add ---
class AddBackward : public AutogradNode {
public:
  AddBackward(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
      : a_(std::move(a)), b_(std::move(b)) {}
  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {a_, b_};
  }
  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (a_ && a_->requires_grad()) {
      a_->accumulate_grad(*grad_output);
    }
    if (b_ && b_->requires_grad()) {
      b_->accumulate_grad(*grad_output);
    }
  }
private:
  std::shared_ptr<Tensor> a_, b_;
};

// --- Mul ---
class MulBackward : public AutogradNode {
public:
  MulBackward(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
              std::shared_ptr<Tensor> a_val, std::shared_ptr<Tensor> b_val)
      : a_(std::move(a)), b_(std::move(b)),
        a_val_(std::move(a_val)), b_val_(std::move(b_val)) {}
  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {a_, b_};
  }
  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    NoGradGuard guard;
    if (a_ && a_->requires_grad()) {
      Tensor ga = mul(*grad_output, *b_val_);
      a_->accumulate_grad(ga);
    }
    if (b_ && b_->requires_grad()) {
      Tensor gb = mul(*grad_output, *a_val_);
      b_->accumulate_grad(gb);
    }
  }
private:
  std::shared_ptr<Tensor> a_, b_, a_val_, b_val_;
};

// --- Sum ---
class SumBackward : public AutogradNode {
public:
  explicit SumBackward(std::shared_ptr<Tensor> a) : a_(std::move(a)) {}
  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {a_};
  }
  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (!a_ || !a_->requires_grad()) return;
    float g = grad_output->data_float()[0];
    std::shared_ptr<Tensor> grad_a = std::make_shared<Tensor>(
        a_->shape(), DType::Float32, a_->device(), false);
    float* p = grad_a->data_float();
    for (int64_t i = 0; i < grad_a->numel(); ++i) p[i] = g;
    a_->accumulate_grad(*grad_a);
  }
private:
  std::shared_ptr<Tensor> a_;
};

// --- Transpose ---
class TransposeBackward : public AutogradNode {
public:
  explicit TransposeBackward(std::shared_ptr<Tensor> a) : a_(std::move(a)) {}
  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {a_};
  }
  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (!a_ || !a_->requires_grad()) return;
    NoGradGuard guard;
    Tensor g = transpose(*grad_output);
    a_->accumulate_grad(g);
  }
private:
  std::shared_ptr<Tensor> a_;
};

// --- Matmul ---
class MatmulBackward : public AutogradNode {
public:
  MatmulBackward(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                 std::shared_ptr<Tensor> a_val, std::shared_ptr<Tensor> b_val)
      : a_(std::move(a)), b_(std::move(b)),
        a_val_(std::move(a_val)), b_val_(std::move(b_val)) {}
  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {a_, b_};
  }
  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    NoGradGuard guard;
    // d(out)/dA = grad_output @ B^T, d(out)/dB = A^T @ grad_output
    if (a_ && a_->requires_grad()) {
      Tensor bt = transpose(*b_val_);
      Tensor ga = matmul(*grad_output, bt);
      a_->accumulate_grad(ga);
    }
    if (b_ && b_->requires_grad()) {
      Tensor at = transpose(*a_val_);
      Tensor gb = matmul(at, *grad_output);
      b_->accumulate_grad(gb);
    }
  }
private:
  std::shared_ptr<Tensor> a_, b_, a_val_, b_val_;
};

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
  expect_float32(a, "add");
  expect_float32(b, "add");
  expect_same_shape(a, b);

  Tensor out(a.shape(), DType::Float32, a.device(), false);
  const float* pa = a.data_float();
  const float* pb = b.data_float();
  float* po = out.data_float();
  for (int64_t i = 0; i < out.numel(); ++i)
    po[i] = pa[i] + pb[i];

  if (is_grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    out.set_requires_grad(true);
    auto node = std::make_shared<AddBackward>(
        std::make_shared<Tensor>(a),
        std::make_shared<Tensor>(b));
    out.set_grad_fn(node);
  }
  return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  expect_float32(a, "mul");
  expect_float32(b, "mul");
  expect_same_shape(a, b);

  Tensor out(a.shape(), DType::Float32, a.device(), false);
  const float* pa = a.data_float();
  const float* pb = b.data_float();
  float* po = out.data_float();
  for (int64_t i = 0; i < out.numel(); ++i)
    po[i] = pa[i] * pb[i];

  if (is_grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    out.set_requires_grad(true);
    auto a_copy = std::make_shared<Tensor>(a);
    auto b_copy = std::make_shared<Tensor>(b);
    auto node = std::make_shared<MulBackward>(a_copy, b_copy, a_copy, b_copy);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor sum(const Tensor& a) {
  expect_float32(a, "sum");
  std::vector<int64_t> scalar_shape = {1};
  Tensor out(scalar_shape, DType::Float32, a.device(), false);
  float s = 0;
  const float* p = a.data_float();
  for (int64_t i = 0; i < a.numel(); ++i) s += p[i];
  out.data_float()[0] = s;

  if (is_grad_enabled() && a.requires_grad()) {
    out.set_requires_grad(true);
    auto node = std::make_shared<SumBackward>(std::make_shared<Tensor>(a));
    out.set_grad_fn(node);
  }
  return out;
}

Tensor transpose(const Tensor& a) {
  expect_float32(a, "transpose");
  if (a.dim() < 2)
    throw std::invalid_argument("transpose: need at least 2 dimensions");
  const auto& sh = a.shape();
  std::vector<int64_t> new_shape(sh.size());
  for (size_t i = 0; i < sh.size(); ++i) new_shape[i] = sh[i];
  std::swap(new_shape[new_shape.size() - 1], new_shape[new_shape.size() - 2]);

  Tensor out(new_shape, DType::Float32, a.device(), false);
  int64_t M = sh[sh.size() - 2];
  int64_t N = sh[sh.size() - 1];
  const float* src = a.data_float();
  float* dst = out.data_float();
  if (a.dim() == 2) {
    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < N; ++j)
        dst[j * M + i] = src[i * N + j];
  } else {
    int64_t batch = 1;
    for (size_t i = 0; i < sh.size() - 2; ++i) batch *= sh[i];
    for (int64_t b = 0; b < batch; ++b) {
      const float* s = src + b * M * N;
      float* d = dst + b * M * N;
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
          d[j * M + i] = s[i * N + j];
    }
  }

  if (is_grad_enabled() && a.requires_grad()) {
    out.set_requires_grad(true);
    auto node = std::make_shared<TransposeBackward>(std::make_shared<Tensor>(a));
    out.set_grad_fn(node);
  }
  return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  expect_float32(a, "matmul");
  expect_float32(b, "matmul");
  if (a.dim() != 2 || b.dim() != 2)
    throw std::invalid_argument("matmul: 2D tensors only");
  int64_t M = a.shape()[0];
  int64_t K = a.shape()[1];
  if (b.shape()[0] != K)
    throw std::invalid_argument("matmul: incompatible shapes");
  int64_t N = b.shape()[1];

  Tensor out({M, N}, DType::Float32, a.device(), false);
  const float* pa = a.data_float();
  const float* pb = b.data_float();
  float* po = out.data_float();
  std::memset(po, 0, static_cast<size_t>(M * N) * sizeof(float));
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) {
      float aik = pa[i * K + k];
      for (int64_t j = 0; j < N; ++j)
        po[i * N + j] += aik * pb[k * N + j];
    }

  if (is_grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    out.set_requires_grad(true);
    auto a_copy = std::make_shared<Tensor>(a);
    auto b_copy = std::make_shared<Tensor>(b);
    auto node = std::make_shared<MatmulBackward>(a_copy, b_copy, a_copy, b_copy);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor ones_like(const Tensor& t) {
  Tensor out(t.shape(), DType::Float32, t.device(), false);
  float* p = out.data_float();
  for (int64_t i = 0; i < out.numel(); ++i) p[i] = 1.0f;
  return out;
}

}  // namespace llm

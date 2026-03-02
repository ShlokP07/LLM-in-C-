/**
 * Test entry point for basic Tensor sanity checks and autograd gradient checks.
 */

#include <llm/llm.hpp>
#include <llm/ops.hpp>
#include <llm/autograd.hpp>
#include <llm/module.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>

using llm::DType;
using llm::Device;
using llm::Tensor;
using llm::Module;
using llm::Parameter;
using llm::add;
using llm::mul;
using llm::sum;
using llm::matmul;
using llm::transpose;

// Verify that version() returns some non-null, non-empty string.
static void test_version() {
  const char* v = llm::version();
  assert(v != nullptr);
  assert(v[0] != '\0');
}

// Construct a small tensor and check basic metadata: shape, numel, dtype, device.
static void test_tensor_basic_shape() {
  std::vector<long long> shape = {2, 3};
  Tensor t(shape, DType::Float32, Device::cpu(), /*requires_grad=*/false);

  assert(t.dim() == 2);
  assert(t.numel() == 6);
  assert(t.dtype() == DType::Float32);
  assert(t.device().type == llm::DeviceType::CPU);

  const auto& s = t.shape();
  assert(s.size() == 2);
  assert(s[0] == 2);
  assert(s[1] == 3);
}

// Check that zeros() constructs a tensor of the requested size and type.
static void test_tensor_zeros() {
  std::vector<long long> shape = {4};
  Tensor t = Tensor::zeros(shape, DType::Int64, Device::cpu(), /*requires_grad=*/true);

  assert(t.numel() == 4);
  assert(t.dtype() == DType::Int64);
  assert(t.requires_grad());
}

// Check that from_data enforces matching shape and data size and copies data.
static void test_tensor_from_data() {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  std::vector<long long> shape = {4};

  Tensor t = Tensor::from_data(data, shape, /*requires_grad=*/false);
  assert(t.numel() == 4);

  const float* p = t.data_float();
  assert(p != nullptr);
  for (int i = 0; i < 4; ++i) {
    assert(p[i] == data[i]);
  }

  // Mismatched shape should throw.
  bool threw = false;
  try {
    Tensor::from_data(data, std::vector<long long>{2, 2, 2}, false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);
}

// Check that reshape preserves numel and shares storage.
static void test_tensor_reshape() {
  std::vector<float> data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  std::vector<long long> shape = {2, 3};

  Tensor t = Tensor::from_data(data, shape, /*requires_grad=*/false);
  Tensor r = t.reshape({3, 2});

  assert(r.numel() == t.numel());
  assert(r.shape().size() == 2);
  assert(r.shape()[0] == 3);
  assert(r.shape()[1] == 2);

  // Reshape with incompatible total size should throw.
  bool threw = false;
  try {
    (void)t.reshape({7});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);
}

// Finite-difference gradient check: compare autograd grad with numerical grad.
static void grad_check_add() {
  Tensor a = Tensor::from_data({1.f, 2.f, 3.f}, {3}, true);
  Tensor b = Tensor::from_data({0.5f, 1.f, 1.5f}, {3}, true);
  Tensor c = add(a, b);
  Tensor loss = sum(c);
  loss.backward();

  assert(a.grad() != nullptr);
  assert(b.grad() != nullptr);
  const float eps = 1e-4f;
  for (int i = 0; i < 3; ++i) {
    assert(std::fabs(a.grad()->data_float()[i] - 1.f) < 1e-5f);
    assert(std::fabs(b.grad()->data_float()[i] - 1.f) < 1e-5f);
  }
}

static void grad_check_mul() {
  Tensor a = Tensor::from_data({1.f, 2.f}, {2}, true);
  Tensor b = Tensor::from_data({3.f, 4.f}, {2}, true);
  Tensor c = mul(a, b);
  Tensor loss = sum(c);
  loss.backward();

  assert(a.grad() != nullptr);
  assert(b.grad() != nullptr);
  assert(std::fabs(a.grad()->data_float()[0] - 3.f) < 1e-5f);
  assert(std::fabs(a.grad()->data_float()[1] - 4.f) < 1e-5f);
  assert(std::fabs(b.grad()->data_float()[0] - 1.f) < 1e-5f);
  assert(std::fabs(b.grad()->data_float()[1] - 2.f) < 1e-5f);
}

static void grad_check_sum() {
  Tensor a = Tensor::from_data({1.f, 2.f, 3.f}, {3}, true);
  Tensor s = sum(a);
  s.backward();

  assert(a.grad() != nullptr);
  for (int i = 0; i < 3; ++i)
    assert(std::fabs(a.grad()->data_float()[i] - 1.f) < 1e-5f);
}

static void grad_check_matmul() {
  Tensor a = Tensor::from_data({1.f, 2.f, 3.f, 4.f}, {2, 2}, true);
  Tensor b = Tensor::from_data({1.f, 0.f, 0.f, 1.f}, {2, 2}, true);
  Tensor c = matmul(a, b);
  Tensor loss = sum(c);
  loss.backward();

  assert(a.grad() != nullptr);
  assert(b.grad() != nullptr);
  // d(sum(A@B))/dA = ones, d(sum(A@B))/dB = ones (for B=I, A@I=A, sum(A)=sum of A elements, grad w.r.t. A is ones)
  assert(std::fabs(a.grad()->data_float()[0] - 1.f) < 1e-5f);
  assert(std::fabs(a.grad()->data_float()[1] - 1.f) < 1e-5f);
  assert(std::fabs(a.grad()->data_float()[2] - 1.f) < 1e-5f);
  assert(std::fabs(a.grad()->data_float()[3] - 1.f) < 1e-5f);
}

static void grad_check_transpose() {
  Tensor a = Tensor::from_data({1.f, 2.f, 3.f, 4.f}, {2, 2}, true);
  Tensor t = transpose(a);
  Tensor loss = sum(t);
  loss.backward();

  assert(a.grad() != nullptr);
  for (int i = 0; i < 4; ++i)
    assert(std::fabs(a.grad()->data_float()[i] - 1.f) < 1e-5f);
}

static void test_no_grad() {
  Tensor a = Tensor::from_data({1.f}, {1}, true);
  Tensor b = Tensor::from_data({2.f}, {1}, true);
  Tensor c;
  {
    llm::NoGradGuard guard;  // ops in this scope don't build the graph
    c = add(a, b);
  }
  assert(!c.requires_grad());
  assert(c.grad_fn() == nullptr);
  assert(c.numel() == 1);
  assert(std::fabs(c.data_float()[0] - 3.f) < 1e-5f);
}

static void test_detach() {
  Tensor a = Tensor::from_data({1.f, 2.f}, {2}, true);
  Tensor b = add(a, a);
  Tensor c = b.detach();  // same data, no grad_fn (stops backward here)
  assert(!c.requires_grad());
  assert(c.grad_fn() == nullptr);
  assert(c.numel() == 2);
  assert(std::fabs(c.data_float()[0] - 2.f) < 1e-5f);
}

// Simple test modules to verify parameter registration and train/eval propagation.
class LeafModule : public Module {
public:
  LeafModule() {
    register_parameter("w", Parameter::zeros({1}));
  }
};

class ParentModule : public Module {
public:
  ParentModule() {
    register_parameter("b", Parameter::zeros({1}));
    child = std::make_shared<LeafModule>();
    register_module("child", child);
  }

  std::shared_ptr<LeafModule> child;
};

static void test_module_parameters_and_modes() {
  ParentModule m;

  // parameters() should include both parent and child parameters.
  auto params = m.parameters();
  assert(params.size() == 2);
  for (Parameter* p : params) {
    assert(p != nullptr);
    assert(p->requires_grad());
  }

  // train()/eval() should propagate to submodules.
  assert(m.is_training());
  assert(m.child->is_training());

  m.eval();
  assert(!m.is_training());
  assert(!m.child->is_training());

  m.train();
  assert(m.is_training());
  assert(m.child->is_training());
}

int main() {
  std::cout << "Running LLM tests..." << std::endl;

  test_version();
  test_tensor_basic_shape();
  test_tensor_zeros();
  test_tensor_from_data();
  test_tensor_reshape();

  grad_check_add();
  grad_check_mul();
  grad_check_sum();
  grad_check_matmul();
  grad_check_transpose();
  test_no_grad();
  test_detach();
  test_module_parameters_and_modes();

  std::cout << "All Tensor and autograd tests passed." << std::endl;
  return 0;
}

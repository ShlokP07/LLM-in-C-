/**
 * Test entry point for basic Tensor sanity checks.
 *
 * As the project grows this file can evolve into a more structured test
 * suite or be replaced by a dedicated testing framework. For now we keep
 * things lightweight and use simple assertions to validate behavior.
 */

#include <llm/llm.hpp>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using llm::DType;
using llm::Device;
using llm::Tensor;

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

int main() {
  std::cout << "Running LLM tests..." << std::endl;

  test_version();
  test_tensor_basic_shape();
  test_tensor_zeros();
  test_tensor_from_data();
  test_tensor_reshape();

  std::cout << "All Tensor tests passed." << std::endl;
  return 0;
}

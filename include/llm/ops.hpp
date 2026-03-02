#pragma once

#include <llm/tensor.hpp>

namespace llm {

/** Element-wise add: out = a + b. Same shape only (no broadcast yet). */
Tensor add(const Tensor& a, const Tensor& b);

/** Element-wise multiply: out = a * b. Same shape. */
Tensor mul(const Tensor& a, const Tensor& b);

/** Sum all elements to scalar, or over dims (for now: all elements → scalar). */
Tensor sum(const Tensor& a);

/** Matrix multiply (2D): out = a @ b. a: (M,K), b: (K,N) → (M,N). */
Tensor matmul(const Tensor& a, const Tensor& b);

/** Transpose last two dimensions. For 2D: out = a^T. */
Tensor transpose(const Tensor& a);

/** Ones with same shape as t (float32, no grad). */
Tensor ones_like(const Tensor& t);

}  // namespace llm

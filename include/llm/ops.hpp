#pragma once

#include <llm/tensor.hpp>

namespace llm {

/** Element-wise add: out = a + b. Supports same-shape and (N,D)+(D) bias-style broadcast. */
Tensor add(const Tensor& a, const Tensor& b);

/** Element-wise multiply: out = a * b. Supports same-shape and (N,D)*(D) bias-style broadcast. */
Tensor mul(const Tensor& a, const Tensor& b);

/** Element-wise subtract: out = a - b. Same-shape only. */
Tensor sub(const Tensor& a, const Tensor& b);

/** Element-wise divide: out = a / b. Same-shape only. */
Tensor div(const Tensor& a, const Tensor& b);

/** Element-wise negation: out = -a. */
Tensor neg(const Tensor& a);

/** Sum all elements to scalar. */
Tensor sum(const Tensor& a);

/** Sum along a dimension (currently supports 2D, dim=0 or 1). */
Tensor sum(const Tensor& a, int64_t dim, bool keepdim);

/** Mean along a dimension (currently supports 2D, dim=0 or 1). */
Tensor mean(const Tensor& a, int64_t dim, bool keepdim);

/** Element-wise exponential: out = exp(a). */
Tensor exp(const Tensor& a);

/** Max along a dimension (returns values only; supports 2D, dim=0 or 1). */
Tensor max(const Tensor& a, int64_t dim, bool keepdim);

/** Matrix multiply (2D): out = a @ b. a: (M,K), b: (K,N) → (M,N). */
Tensor matmul(const Tensor& a, const Tensor& b);

/** Transpose last two dimensions. For 2D: out = a^T. */
Tensor transpose(const Tensor& a);

/** Ones with same shape as t (float32, no grad). */
Tensor ones_like(const Tensor& t);

/**
 * View a 2D tensor (T, dim) as 3D (T, num_heads, head_dim) where dim = num_heads * head_dim.
 * This is a reshape-based view (shares storage, autograd-aware).
 */
Tensor view_as_heads(const Tensor& x, int64_t num_heads);

/**
 * Slice a tensor along a single dimension, returning a copy-based slice.
 * Currently supports 2D and 3D float32 tensors; dims outside [0, rank) will throw.
 *
 * Example: slice(x, 1, h, h+1) on shape (T, H, Dh) returns (T, 1, Dh).
 */
Tensor slice(const Tensor& x, int64_t dim, int64_t start, int64_t end);

/**
 * Gather rows from a 2D tensor by int64 indices.
 * weight: (V, D) float32, indices: (N,) int64 -> output (N, D).
 * output[i, j] = weight[indices[i], j]. Indices must be in [0, V).
 */
Tensor gather(const Tensor& weight, const Tensor& indices);

}  // namespace llm

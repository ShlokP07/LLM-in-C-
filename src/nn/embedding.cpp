#include <llm/nn.hpp>
#include <llm/ops.hpp>
#include <llm/init.hpp>

#include <stdexcept>

namespace llm {

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim) {
  if (num_embeddings <= 0 || embedding_dim <= 0)
    throw std::invalid_argument("Embedding: num_embeddings and embedding_dim must be positive");

  Parameter weight = Parameter::zeros({num_embeddings, embedding_dim});
  normal_(weight, 0.f, 0.02f);
  register_parameter("weight", weight);
}

Tensor Embedding::operator()(const Tensor& indices) {
  if (indices.dtype() != DType::Int64)
    throw std::invalid_argument("Embedding: indices must be int64");
  if (indices.dim() != 1)
    throw std::invalid_argument("Embedding: indices must be 1D (N)");

  Parameter& weight = parameters_.at("weight");
  return gather(weight, indices);
}

}  // namespace llm

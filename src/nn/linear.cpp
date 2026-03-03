#include <llm/nn.hpp>
#include <llm/ops.hpp>
#include <llm/init.hpp>

#include <stdexcept>

namespace llm {

Linear::Linear(int64_t in_features, int64_t out_features, bool bias)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(bias) {
  if (in_features <= 0 || out_features <= 0)
    throw std::invalid_argument("Linear: in_features and out_features must be positive");

  Parameter weight = Parameter::zeros({out_features, in_features});
  xavier_uniform_(weight);
  register_parameter("weight", weight);

  if (has_bias_) {
    Parameter b = Parameter::zeros({out_features});
    zeros_(b);
    register_parameter("bias", b);
  }
}

Tensor Linear::operator()(const Tensor& x) {
  if (x.dtype() != DType::Float32)
    throw std::invalid_argument("Linear: input must be float32");
  if (x.dim() != 2)
    throw std::invalid_argument("Linear: input must be 2D (batch, in_features)");
  if (x.shape()[1] != in_features_)
    throw std::invalid_argument("Linear: input last dim must equal in_features");

  Parameter& weight = parameters_.at("weight");
  Tensor wT = transpose(weight);
  Tensor out = matmul(x, wT);

  if (has_bias_) {
    Parameter& bias = parameters_.at("bias");
    out = add(out, bias);
  }
  return out;
}

}  // namespace llm

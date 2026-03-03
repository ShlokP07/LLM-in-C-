#pragma once

#include <llm/module.hpp>
#include <llm/tensor.hpp>

namespace llm {

/** Fully connected layer: y = x @ W^T + b. Weight (out_features, in_features), bias (out_features). */
class Linear : public Module {
public:
  Linear(int64_t in_features, int64_t out_features, bool bias = true);

  /** Forward: x shape (batch, in_features) -> out shape (batch, out_features). */
  Tensor operator()(const Tensor& x);

  int64_t in_features() const { return in_features_; }
  int64_t out_features() const { return out_features_; }
  bool has_bias() const { return has_bias_; }

private:
  int64_t in_features_;
  int64_t out_features_;
  bool has_bias_;
};

}  // namespace llm

#pragma once

#include <llm/tensor.hpp>
#include <memory>
#include <vector>

namespace llm {

/**
 * Node in the autograd computation graph. Created during forward pass;
 * backward() is invoked during reverse traversal with the gradient of
 * this node's output.
 */
class AutogradNode {
public:
  virtual ~AutogradNode() = default;

  /** Input tensors that this node depends on (for topo sort and backward). */
  virtual std::vector<std::shared_ptr<Tensor>> inputs() const = 0;

  /**
   * Compute gradients w.r.t. inputs and accumulate into their .grad.
   * grad_output has the same shape as this node's output.
   */
  virtual void backward(const std::shared_ptr<Tensor>& grad_output) = 0;
};

/**
 * Run backward from the given root tensor. If root.grad() is null and
 * root.requires_grad(), initializes root.grad to ones_like(root).
 * Performs reverse topo sort and calls each node's backward().
 */
void run_backward(Tensor& root);

/** RAII guard: ops executed inside do not record to the graph. */
class NoGradGuard {
public:
  NoGradGuard();
  ~NoGradGuard();
  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;
};

/** True if we are currently inside a no_grad() scope (no recording). */
bool is_grad_enabled();

}  // namespace llm

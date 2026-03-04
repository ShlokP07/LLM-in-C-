#pragma once

#include <llm/module.hpp>

#include <cstdint>
#include <vector>

namespace llm {

/**
 * Stochastic Gradient Descent optimizer.
 *
 * Updates parameters with:
 *   p = p - lr * (grad + weight_decay * p)
 * for each float32 Parameter in the parameter list.
 */
class SGD {
public:
  SGD(const std::vector<Parameter*>& params,
      float lr,
      float weight_decay = 0.0f);

  void step();
  void zero_grad();

  float lr() const { return lr_; }
  float weight_decay() const { return weight_decay_; }
  void set_lr(float lr) { lr_ = lr; }

private:
  std::vector<Parameter*> params_;
  float lr_;
  float weight_decay_;
};

/**
 * AdamW optimizer (Adam with decoupled weight decay).
 *
 * Maintains per-parameter first (m) and second (v) moment estimates,
 * applies bias correction, then updates:
 *   param = param - lr * m_hat / (sqrt(v_hat) + eps)
 *   param = param * (1 - lr * weight_decay)  [decoupled weight decay]
 */
class AdamW {
public:
  AdamW(const std::vector<Parameter*>& params,
        float lr,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weight_decay = 0.0f);

  void step();
  void zero_grad();

  float lr() const { return lr_; }
  float beta1() const { return beta1_; }
  float beta2() const { return beta2_; }
  float eps() const { return eps_; }
  float weight_decay() const { return weight_decay_; }
  int64_t step_count() const { return step_count_; }

  void set_lr(float lr) { lr_ = lr; }

  /** Optimizer state for checkpointing: step_count and per-parameter m/v (keys "step_count", "0_m", "0_v", ...). */
  Module::StateDict state_dict() const;
  void load_state_dict(const Module::StateDict& state);

private:
  std::vector<Parameter*> params_;
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
  int64_t step_count_;
  // Per-parameter state: state_m_[i] and state_v_[i] for params_[i] (same size as param numel)
  std::vector<std::vector<float>> state_m_;
  std::vector<std::vector<float>> state_v_;
};

/**
 * Clips the gradient norm of the parameters in place (L2 norm over all elements).
 * If the total norm exceeds max_norm, scales all gradients by (max_norm / total_norm).
 * Only considers float32 parameters that have a non-null grad.
 *
 * @param params  Parameters whose grads are clipped (in place).
 * @param max_norm  Maximum allowed L2 norm of the concatenated gradients.
 * @return Total gradient norm before clipping (0 if no grads).
 */
float clip_grad_norm_(std::vector<Parameter*>& params, float max_norm);

/**
 * Simple step learning rate scheduler:
 *   lr(step) = base_lr * gamma^(floor(step / step_size))
 *
 * Does not own an optimizer; use apply() to update any optimizer that
 * exposes set_lr(float).
 */
class StepLR {
public:
  StepLR(float base_lr, int64_t step_size, float gamma = 0.1f)
      : base_lr_(base_lr), step_size_(step_size), gamma_(gamma) {}

  float base_lr() const { return base_lr_; }
  int64_t step_size() const { return step_size_; }
  float gamma() const { return gamma_; }

  float get_lr(int64_t step) const {
    if (step_size_ <= 0) return base_lr_;
    if (step < 0) step = 0;
    int64_t k = step / step_size_;
    float factor = 1.0f;
    for (int64_t i = 0; i < k; ++i)
      factor *= gamma_;
    return base_lr_ * factor;
  }

  template <typename Optim>
  void apply(Optim& opt, int64_t step) const {
    opt.set_lr(get_lr(step));
  }

private:
  float base_lr_;
  int64_t step_size_;
  float gamma_;
};

}  // namespace llm


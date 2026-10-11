#pragma once

#include "sovereigngrad/tensor.hpp"
#include <vector>
#include <cstdint>

namespace sovereigngrad {

// Shape helpers
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape_a, const std::vector<int64_t>& shape_b);
void accumulate_grad(std::shared_ptr<TensorImpl> target, const Tensor& grad);

// Arithmetic ops
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor neg(const Tensor& a);

// Matrix Multiplication
Tensor matmul(const Tensor& a, const Tensor& b);

// Activations
Tensor relu(const Tensor& a);
Tensor gelu(const Tensor& a);
Tensor silu(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor tanh_op(const Tensor& a);
Tensor exp_op(const Tensor& a);
Tensor log_op(const Tensor& a);
Tensor sqrt_op(const Tensor& a);
Tensor pow_op(const Tensor& a, float exponent);
Tensor clamp_op(const Tensor& a, float min_val, float max_val);
Tensor dropout_op(const Tensor& a, float p, bool training = true);

// Reductions
Tensor sum(const Tensor& a, int dim, bool keepdim = false);
Tensor sum(const Tensor& a);
Tensor mean(const Tensor& a, int dim, bool keepdim = false);
Tensor mean(const Tensor& a);
Tensor max_op(const Tensor& a, int dim, bool keepdim = false);

// Loss Functions
Tensor mse_loss(const Tensor& pred, const Tensor& target);
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);
Tensor softmax_cross_entropy(const Tensor& logits, const Tensor& targets);
Tensor bce_with_logits_loss(const Tensor& logits, const Tensor& targets);
Tensor l1_loss(const Tensor& pred, const Tensor& target);

// Normalization & Neural Ops
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f);
Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps = 1e-5f);
Tensor masked_fill(const Tensor& x, const Tensor& mask, float value);
Tensor softmax(const Tensor& x, int dim = -1);
Tensor max_pool2d(const Tensor& x, int kernel_size = 2, int stride = 2, int padding = 0);
Tensor avg_pool2d(const Tensor& x, int kernel_size = 2, int stride = 2, int padding = 0);
Tensor conv1d(const Tensor& x, const Tensor& weight, const Tensor& bias, int stride = 1, int padding = 0);
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias, int stride = 1, int padding = 0);
Tensor batch_norm1d(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                    Tensor& running_mean, Tensor& running_var,
                    float momentum = 0.1f, float eps = 1e-5f, bool training = true);
Tensor batch_norm2d(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                    Tensor& running_mean, Tensor& running_var,
                    float momentum = 0.1f, float eps = 1e-5f, bool training = true);
Tensor batch_norm3d(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                    Tensor& running_mean, Tensor& running_var,
                    float momentum = 0.1f, float eps = 1e-5f, bool training = true);


Tensor smooth_l1_loss(const Tensor& pred, const Tensor& target, float beta = 1.0f);
Tensor kl_div_loss(const Tensor& input, const Tensor& target);
Tensor embedding(const Tensor& indices, const Tensor& weight);
Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);
Tensor rotary_embedding(const Tensor& x, const Tensor& cos, const Tensor& sin);
Tensor slice_op(const Tensor& x, int dim, int64_t start, int64_t end, int64_t step = 1);
Tensor select_op(const Tensor& x, int dim, int64_t index);
Tensor index_select_op(const Tensor& x, int dim, const Tensor& index);
Tensor flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V, bool is_causal = false, float scale = -1.0f);

// Operator & Kernel Fusion
Tensor fused_linear_gelu(const Tensor& x, const Tensor& weight, const Tensor& bias);
Tensor fused_add_layernorm(const Tensor& x, const Tensor& residual, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f);
Tensor fused_scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V, float scale);

} // namespace sovereigngrad

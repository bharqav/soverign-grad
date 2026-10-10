#pragma once

#include "sovereigngrad/cuda_mock.hpp"
#include <cstdint>

namespace sovereigngrad {
namespace cuda {

// Element-wise operations
void launch_add(float* out, const float* a, const float* b, int N);
void launch_sub(float* out, const float* a, const float* b, int N);
void launch_mul(float* out, const float* a, const float* b, int N);
void launch_div(float* out, const float* a, const float* b, int N);
void launch_neg(float* out, const float* in, int N);

// Activations
void launch_relu(float* out, const float* in, int N);
void launch_relu_backward(float* d_in, const float* d_out, const float* in, int N);
void launch_gelu(float* out, const float* in, int N);
void launch_gelu_backward(float* d_in, const float* d_out, const float* in, int N);
void launch_silu(float* out, const float* in, int N);
void launch_silu_backward(float* d_in, const float* d_out, const float* in, int N);
void launch_sigmoid(float* out, const float* in, int N);
void launch_sigmoid_backward(float* d_in, const float* d_out, const float* out, int N);
void launch_tanh(float* out, const float* in, int N);
void launch_tanh_backward(float* d_in, const float* d_out, const float* out, int N);
void launch_exp(float* out, const float* in, int N);
void launch_log(float* out, const float* in, int N);
void launch_sqrt(float* out, const float* in, int N);
void launch_pow(float* out, const float* in, float exponent, int N);
void launch_clamp(float* out, const float* in, float min_val, float max_val, int N);

// Matrix Multiplication
void launch_matmul_naive(float* C, const float* A, const float* B, int M, int N, int K);
void launch_matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K);

// Reductions
void launch_sum_all(float* out, const float* in, int N);
void launch_mean_all(float* out, const float* in, int N);

// Loss Functions
void launch_mse_loss_forward(float* out, const float* pred, const float* target, int N);
void launch_mse_loss_backward(float* d_pred, const float* pred, const float* target, float scale, int N);
void launch_softmax_cross_entropy_forward(float* loss, const float* logits, const float* targets, int B, int C);
void launch_softmax_cross_entropy_backward(float* d_logits, const float* logits, const float* targets, int B, int C);
void launch_bce_with_logits_forward(float* out, const float* logits, const float* targets, int N);
void launch_bce_with_logits_backward(float* d_logits, const float* logits, const float* targets, int N);
void launch_smooth_l1_loss_forward(float* out, const float* pred, const float* target, float beta, int N);
void launch_smooth_l1_loss_backward(float* d_pred, const float* pred, const float* target, float beta, float scale, int N);
void launch_kl_div_loss_forward(float* out, const float* input, const float* target, int N);
void launch_kl_div_loss_backward(float* d_input, const float* target, float scale, int N);

// Layer Normalization & RMSNorm
void launch_layernorm_forward(float* out, const float* in, const float* gamma, const float* beta,
                              int B, int D, float eps);
void launch_layernorm_backward(float* d_in, float* d_gamma, float* d_beta,
                               const float* d_out, const float* in, const float* gamma,
                               int B, int D, float eps);
void launch_rmsnorm_forward(float* out, const float* in, const float* gamma,
                            int B, int D, float eps);
void launch_rmsnorm_backward(float* d_in, float* d_gamma,
                             const float* d_out, const float* in, const float* gamma,
                             int B, int D, float eps);
void launch_masked_fill(float* out, const float* in, const float* mask, float value, int N);

// Convolution & BatchNorm
void launch_conv2d_forward(float* out, const float* in, const float* weight, const float* bias,
                           int B, int C, int H, int W, int O, int Kh, int Kw,
                           int H_out, int W_out, int stride, int padding);
void launch_conv2d_backward_input(float* d_in, const float* d_out, const float* weight,
                                 int B, int C, int H, int W, int O, int Kh, int Kw,
                                 int H_out, int W_out, int stride, int padding);
void launch_conv2d_backward_weight(float* d_weight, const float* d_out, const float* in,
                                  int B, int C, int H, int W, int O, int Kh, int Kw,
                                  int H_out, int W_out, int stride, int padding);
void launch_batchnorm_forward(float* out, const float* in, const float* gamma, const float* beta,
                              const float* mean, const float* var,
                              int B, int C, int H, int W, float eps);
void launch_batchnorm_backward(float* d_in, float* d_gamma, float* d_beta,
                               const float* d_out, const float* in, const float* gamma,
                               const float* mean, const float* var,
                               int B, int C, int H, int W, float eps);

// Pooling, Conv1D, Embedding, Softmax
void launch_softmax_forward(float* out, const float* in, int outer_size, int dim_size, int inner_size);
void launch_softmax_backward(float* d_in, const float* d_out, const float* out, int outer_size, int dim_size, int inner_size);
void launch_max_pool2d_forward(float* out, float* argmax, const float* in, int B, int C, int H, int W, int Kh, int Kw, int H_out, int W_out, int stride, int padding);
void launch_max_pool2d_backward(float* d_in, const float* d_out, const float* argmax, int B, int C, int H, int W, int H_out, int W_out);
void launch_avg_pool2d_forward(float* out, const float* in, int B, int C, int H, int W, int Kh, int Kw, int H_out, int W_out, int stride, int padding);
void launch_avg_pool2d_backward(float* d_in, const float* d_out, int B, int C, int H, int W, int Kh, int Kw, int H_out, int W_out, int stride, int padding);
void launch_conv1d_forward(float* out, const float* in, const float* weight, const float* bias, int B, int C, int L, int O, int K, int L_out, int stride, int padding);
void launch_conv1d_backward(float* d_in, float* d_weight, float* d_bias, const float* d_out, const float* in, const float* weight, int B, int C, int L, int O, int K, int L_out, int stride, int padding);
void launch_embedding_forward(float* out, const float* indices, const float* weight, int num_indices, int embedding_dim);
void launch_embedding_backward(float* d_weight, const float* d_out, const float* indices, int num_indices, int embedding_dim);

// Slicing & Indexing
void launch_slice_forward(float* out, const float* in, const int64_t* in_shape, const int64_t* in_strides,
                          const int64_t* out_shape, const int64_t* out_strides,
                          int rank, int dim, int64_t start, int64_t step, int total_out);
void launch_slice_backward(float* d_in, const float* d_out, const int64_t* in_shape, const int64_t* in_strides,
                           const int64_t* out_shape, const int64_t* out_strides,
                           int rank, int dim, int64_t start, int64_t step, int total_out);
void launch_index_select_forward(float* out, const float* in, const float* indices,
                                 int outer_size, int in_dim_size, int out_dim_size, int inner_size);
void launch_index_select_backward(float* d_in, const float* d_out, const float* indices,
                                  int outer_size, int in_dim_size, int out_dim_size, int inner_size);

// FlashAttention-2 Tiled Online Softmax
void launch_flash_attention_forward(float* out, float* lse, const float* Q, const float* K, const float* V,
                                    int B, int num_heads, int L, int S, int D, float scale, bool is_causal);
void launch_flash_attention_backward(float* dQ, float* dK, float* dV, const float* dO,
                                     const float* Q, const float* K, const float* V, const float* O, const float* lse,
                                     int B, int num_heads, int L, int S, int D, float scale, bool is_causal);

// Native GPU Optimizers
void launch_adamw_step(float* p, const float* g, float* m, float* v,
                       float lr, float beta1, float beta2, float eps, float weight_decay,
                       float correction1, float correction2, int N);
void launch_sgd_step(float* p, const float* g, float* v,
                     float lr, float momentum, float weight_decay, bool nesterov, int N);

} // namespace cuda
} // namespace sovereigngrad

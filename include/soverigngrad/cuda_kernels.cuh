#pragma once

#include "soverigngrad/cuda_mock.hpp"

namespace sovereigngrad {
namespace cuda {

// CUDA Kernels
__global__ void add_kernel(float* out, const float* a, const float* b, int N);
__global__ void relu_kernel(float* out, const float* in, int N);
__global__ void matmul_naive_kernel(float* C, const float* A, const float* B, int M, int N, int K);
__global__ void matmul_tiled_kernel(float* C, const float* A, const float* B, int M, int N, int K);
__global__ void conv2d_forward_kernel(float* out, const float* in, const float* weight, const float* bias,
                                       int B, int C, int H, int W, int O, int Kh, int Kw,
                                       int H_out, int W_out, int stride, int padding);
__global__ void conv2d_backward_input_kernel(float* d_in, const float* d_out, const float* weight,
                                              int B, int C, int H, int W, int O, int Kh, int Kw,
                                              int H_out, int W_out, int stride, int padding);
__global__ void conv2d_backward_weight_kernel(float* d_weight, const float* d_out, const float* in,
                                               int B, int C, int H, int W, int O, int Kh, int Kw,
                                               int H_out, int W_out, int stride, int padding);
__global__ void batchnorm_forward_kernel(float* out, const float* in, const float* gamma, const float* beta,
                                          const float* mean, const float* var,
                                          int B, int C, int H, int W, float eps);
__global__ void batchnorm_backward_kernel(float* d_in, float* d_gamma, float* d_beta,
                                           const float* d_out, const float* in, const float* gamma,
                                           const float* mean, const float* var,
                                           int B, int C, int H, int W, float eps);

// Host Launch Wrappers
void launch_add(float* out, const float* a, const float* b, int N);
void launch_relu(float* out, const float* in, int N);
void launch_matmul_naive(float* C, const float* A, const float* B, int M, int N, int K);
void launch_matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K);
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

} // namespace cuda
} // namespace sovereigngrad

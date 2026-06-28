#pragma once

#include "soverigngrad/tensor.hpp"
#include "soverigngrad/ops.hpp"
#include "soverigngrad/cuda_mock.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <cstring>

namespace sovereigngrad {

// Linear (Dense) Layer
class Linear {
public:
    Tensor W; // Weights
    Tensor b; // Biases

    Linear(int64_t in_features, int64_t out_features, Device device = Device::CPU) {
        // Xavier/Glorot Initialization: normal distribution with std = sqrt(2.0 / (in_features + out_features))
        float std = std::sqrt(2.0f / (in_features + out_features));
        W = Tensor::randn({in_features, out_features}, 0.0f, std, device, /*requires_grad=*/true);
        
        // Biases initialized to zero
        b = Tensor::zeros({1, out_features}, device, /*requires_grad=*/true);
    }

    Tensor forward(const Tensor& x) {
        // out = x * W + b
        return x.matmul(W) + b;
    }

    std::vector<Tensor> parameters() {
        return {W, b};
    }
};

// Adam Optimizer
class Adam {
public:
    std::vector<Tensor> params;
    std::vector<Tensor> m; // First moment
    std::vector<Tensor> v; // Second moment
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t; // Timestep

    Adam(const std::vector<Tensor>& parameters, float learning_rate = 0.01f, 
         float b1 = 0.9f, float b2 = 0.999f, float epsilon = 1e-8f)
        : params(parameters), lr(learning_rate), beta1(b1), beta2(b2), eps(epsilon), t(0) {
        
        for (const auto& p : params) {
            m.push_back(Tensor::zeros(p.shape(), p.device()));
            v.push_back(Tensor::zeros(p.shape(), p.device()));
        }
    }

    void zero_grad() {
        for (auto& p : params) {
            p.zero_grad();
        }
    }

    void step() {
        t++;
        float correction1 = 1.0f - std::pow(beta1, t);
        float correction2 = 1.0f - std::pow(beta2, t);

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor& p = params[i];
            Tensor grad = p.grad();
            if (!grad.impl) continue;

            // We do the Adam update element-wise on CPU for maximum safety and correctness.
            // Works flawlessly for both CPU and CUDA-device tensors.
            Device dev = p.device();
            Tensor cpu_p = p.to(Device::CPU);
            Tensor cpu_grad = grad.to(Device::CPU);
            Tensor cpu_m = m[i].to(Device::CPU);
            Tensor cpu_v = v[i].to(Device::CPU);

            float* p_data = cpu_p.impl->data();
            const float* g_data = cpu_grad.impl->data();
            float* m_data = cpu_m.impl->data();
            float* v_data = cpu_v.impl->data();

            int64_t total = cpu_p.numel();
            for (int64_t j = 0; j < total; ++j) {
                // Update biased first moment estimate
                m_data[j] = beta1 * m_data[j] + (1.0f - beta1) * g_data[j];
                // Update biased second raw moment estimate
                v_data[j] = beta2 * v_data[j] + (1.0f - beta2) * g_data[j] * g_data[j];

                // Compute bias-corrected first moment estimate
                float m_hat = m_data[j] / correction1;
                // Compute bias-corrected second raw moment estimate
                float v_hat = v_data[j] / correction2;

                // Update parameters
                p_data[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }

            // Write back to the target device
            if (dev == Device::CPU) {
                std::memcpy(p.impl->data(), cpu_p.impl->data(), total * sizeof(float));
            } else {
                cudaMemcpy(p.impl->data(), cpu_p.impl->data(), total * sizeof(float), cudaMemcpyHostToDevice);
            }

            m[i] = cpu_m.to(dev);
            v[i] = cpu_v.to(dev);
        }
    }
};

// Conv2d Layer
class Conv2d {
public:
    Tensor W; // Weights
    Tensor b; // Biases
    int stride;
    int padding;

    Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int stride = 1, int padding = 0, Device device = Device::CPU)
        : stride(stride), padding(padding) {
        
        // Xavier/Glorot Initialization for Convolution
        float fin = in_channels * kernel_size * kernel_size;
        float fout = out_channels * kernel_size * kernel_size;
        float std = std::sqrt(2.0f / (fin + fout));
        
        W = Tensor::randn({out_channels, in_channels, kernel_size, kernel_size}, 0.0f, std, device, /*requires_grad=*/true);
        b = Tensor::zeros({out_channels}, device, /*requires_grad=*/true);
    }

    Tensor forward(const Tensor& x) {
        return conv2d(x, W, b, stride, padding);
    }

    std::vector<Tensor> parameters() {
        return {W, b};
    }
};

// BatchNorm2d Layer
class BatchNorm2d {
public:
    Tensor gamma;
    Tensor beta;
    Tensor running_mean;
    Tensor running_var;
    float momentum;
    float eps;
    bool training;

    BatchNorm2d(int64_t num_features, float momentum = 0.1f, float eps = 1e-5f, Device device = Device::CPU)
        : momentum(momentum), eps(eps), training(true) {
        
        gamma = Tensor::ones({num_features}, device, /*requires_grad=*/true);
        beta = Tensor::zeros({num_features}, device, /*requires_grad=*/true);
        
        running_mean = Tensor::zeros({num_features}, device, /*requires_grad=*/false);
        running_var = Tensor::ones({num_features}, device, /*requires_grad=*/false);
    }

    Tensor forward(const Tensor& x) {
        return batch_norm2d(x, gamma, beta, running_mean, running_var, momentum, eps, training);
    }

    std::vector<Tensor> parameters() {
        return {gamma, beta};
    }
};

} // namespace sovereigngrad

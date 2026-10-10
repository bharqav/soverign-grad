#pragma once

#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/ops.hpp"
#include "sovereigngrad/module.hpp"
#include "sovereigngrad/cuda_kernels.cuh"
#include "sovereigngrad/quantization.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <cstring>
#include <memory>

namespace sovereigngrad {
namespace nn {

// 1. Linear (Dense) Layer
class Linear : public Module {
public:
    Tensor W; // Weights
    Tensor b; // Biases

    Linear(int64_t in_features, int64_t out_features, Device device = Device::CPU) {
        float std = std::sqrt(2.0f / (in_features + out_features));
        W = Tensor::randn({in_features, out_features}, 0.0f, std, device, /*requires_grad=*/true);
        b = Tensor::zeros({1, out_features}, device, /*requires_grad=*/true);

        register_parameter("weight", W);
        register_parameter("bias", b);
    }

    Tensor forward(const Tensor& x) {
        return x.matmul(W) + b;
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 2. LayerNorm
class LayerNorm : public Module {
public:
    Tensor gamma;
    Tensor beta;
    float eps;

    LayerNorm(int64_t normalized_shape, float eps = 1e-5f, Device device = Device::CPU)
        : eps(eps) {
        gamma = Tensor::ones({normalized_shape}, device, /*requires_grad=*/true);
        beta = Tensor::zeros({normalized_shape}, device, /*requires_grad=*/true);

        register_parameter("weight", gamma);
        register_parameter("bias", beta);
    }

    Tensor forward(const Tensor& x) {
        return layer_norm(x, gamma, beta, eps);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 2b. RMSNorm (Root Mean Square Normalization for modern LLMs)
class RMSNorm : public Module {
public:
    Tensor weight;
    float eps;

    RMSNorm(int64_t normalized_shape, float eps = 1e-5f, Device device = Device::CPU)
        : eps(eps) {
        weight = Tensor::ones({normalized_shape}, device, /*requires_grad=*/true);
        register_parameter("weight", weight);
    }

    Tensor forward(const Tensor& x) {
        return rms_norm(x, weight, eps);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 3. Dropout
class Dropout : public Module {
public:
    float p;

    Dropout(float p = 0.5f) : p(p) {}

    Tensor forward(const Tensor& x) {
        return dropout_op(x, p, training_);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 4. Activation Modules
class ReLU : public Module {
public:
    Tensor forward(const Tensor& x) { return relu(x); }
    Tensor operator()(const Tensor& x) { return forward(x); }
};

class GELU : public Module {
public:
    Tensor forward(const Tensor& x) { return gelu(x); }
    Tensor operator()(const Tensor& x) { return forward(x); }
};

class SiLU : public Module {
public:
    Tensor forward(const Tensor& x) { return silu(x); }
    Tensor operator()(const Tensor& x) { return forward(x); }
};

class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& x) { return sigmoid(x); }
    Tensor operator()(const Tensor& x) { return forward(x); }
};

class Tanh : public Module {
public:
    Tensor forward(const Tensor& x) { return tanh_op(x); }
    Tensor operator()(const Tensor& x) { return forward(x); }
};

// 4b. SwiGLU Module (LLaMA / Mistral Gated MLP Activation)
class SwiGLU : public Module {
public:
    std::shared_ptr<Linear> w_gate;
    std::shared_ptr<Linear> w_up;
    std::shared_ptr<Linear> w_down;

    SwiGLU(int64_t in_features, int64_t hidden_features, Device device = Device::CPU) {
        w_gate = std::make_shared<Linear>(in_features, hidden_features, device);
        w_up = std::make_shared<Linear>(in_features, hidden_features, device);
        w_down = std::make_shared<Linear>(hidden_features, in_features, device);

        register_module("w_gate", w_gate);
        register_module("w_up", w_up);
        register_module("w_down", w_down);
    }

    Tensor forward(const Tensor& x) {
        Tensor gate = w_gate->forward(x).silu();
        Tensor up = w_up->forward(x);
        Tensor hidden = gate * up;
        return w_down->forward(hidden);
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

// 5. Sequential Container
class Sequential : public Module {
public:
    std::vector<std::shared_ptr<Module>> layers;

    Sequential() = default;

    template <typename... Modules>
    Sequential(std::shared_ptr<Modules>... mods) {
        int idx = 0;
        ([&](std::shared_ptr<Module> m) {
            layers.push_back(m);
            register_module(std::to_string(idx++), m);
        }(mods), ...);
    }

    void add(std::shared_ptr<Module> module) {
        std::string name = std::to_string(layers.size());
        layers.push_back(module);
        register_module(name, module);
    }

    Tensor forward(const Tensor& input) override {
        Tensor out = input;
        for (auto& layer : layers) {
            out = layer->forward(out);
        }
        return out;
    }

    Tensor operator()(const Tensor& input) {
        return forward(input);
    }
};

// 6. Conv2d Layer
class Conv2d : public Module {
public:
    Tensor W; // Weights
    Tensor b; // Biases
    int stride;
    int padding;

    Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int stride = 1, int padding = 0, Device device = Device::CPU)
        : stride(stride), padding(padding) {
        
        float fin = (float)(in_channels * kernel_size * kernel_size);
        float fout = (float)(out_channels * kernel_size * kernel_size);
        float std = std::sqrt(2.0f / (fin + fout));
        
        W = Tensor::randn({out_channels, in_channels, kernel_size, kernel_size}, 0.0f, std, device, /*requires_grad=*/true);
        b = Tensor::zeros({out_channels}, device, /*requires_grad=*/true);

        register_parameter("weight", W);
        register_parameter("bias", b);
    }

    Tensor forward(const Tensor& x) {
        return conv2d(x, W, b, stride, padding);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 7. BatchNorm2d Layer
class BatchNorm2d : public Module {
public:
    Tensor gamma;
    Tensor beta;
    Tensor running_mean;
    Tensor running_var;
    float momentum;
    float eps;

    BatchNorm2d(int64_t num_features, float momentum = 0.1f, float eps = 1e-5f, Device device = Device::CPU)
        : momentum(momentum), eps(eps) {
        
        gamma = Tensor::ones({num_features}, device, /*requires_grad=*/true);
        beta = Tensor::zeros({num_features}, device, /*requires_grad=*/true);
        
        running_mean = Tensor::zeros({num_features}, device, /*requires_grad=*/false);
        running_var = Tensor::ones({num_features}, device, /*requires_grad=*/false);

        register_parameter("weight", gamma);
        register_parameter("bias", beta);
    }

    Tensor forward(const Tensor& x) {
        return batch_norm2d(x, gamma, beta, running_mean, running_var, momentum, eps, training_);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 8. BatchNorm1d Layer
class BatchNorm1d : public Module {
public:
    Tensor gamma;
    Tensor beta;
    Tensor running_mean;
    Tensor running_var;
    float momentum;
    float eps;

    BatchNorm1d(int64_t num_features, float momentum = 0.1f, float eps = 1e-5f, Device device = Device::CPU)
        : momentum(momentum), eps(eps) {
        
        gamma = Tensor::ones({num_features}, device, /*requires_grad=*/true);
        beta = Tensor::zeros({num_features}, device, /*requires_grad=*/true);
        
        running_mean = Tensor::zeros({num_features}, device, /*requires_grad=*/false);
        running_var = Tensor::ones({num_features}, device, /*requires_grad=*/false);

        register_parameter("weight", gamma);
        register_parameter("bias", beta);
    }

    Tensor forward(const Tensor& x) {
        return batch_norm1d(x, gamma, beta, running_mean, running_var, momentum, eps, training_);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 9. BatchNorm3d Layer
class BatchNorm3d : public Module {
public:
    Tensor gamma;
    Tensor beta;
    Tensor running_mean;
    Tensor running_var;
    float momentum;
    float eps;

    BatchNorm3d(int64_t num_features, float momentum = 0.1f, float eps = 1e-5f, Device device = Device::CPU)
        : momentum(momentum), eps(eps) {
        
        gamma = Tensor::ones({num_features}, device, /*requires_grad=*/true);
        beta = Tensor::zeros({num_features}, device, /*requires_grad=*/true);
        
        running_mean = Tensor::zeros({num_features}, device, /*requires_grad=*/false);
        running_var = Tensor::ones({num_features}, device, /*requires_grad=*/false);

        register_parameter("weight", gamma);
        register_parameter("bias", beta);
    }

    Tensor forward(const Tensor& x) {
        return batch_norm3d(x, gamma, beta, running_mean, running_var, momentum, eps, training_);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};



// 9. Pooling Layers
class MaxPool2d : public Module {
public:
    int kernel_size;
    int stride;
    int padding;

    MaxPool2d(int kernel_size = 2, int stride = 2, int padding = 0)
        : kernel_size(kernel_size), stride(stride), padding(padding) {}

    Tensor forward(const Tensor& x) {
        return max_pool2d(x, kernel_size, stride, padding);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

class AvgPool2d : public Module {
public:
    int kernel_size;
    int stride;
    int padding;

    AvgPool2d(int kernel_size = 2, int stride = 2, int padding = 0)
        : kernel_size(kernel_size), stride(stride), padding(padding) {}

    Tensor forward(const Tensor& x) {
        return avg_pool2d(x, kernel_size, stride, padding);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 10. Conv1d Layer
class Conv1d : public Module {
public:
    Tensor W;
    Tensor b;
    int stride;
    int padding;

    Conv1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int stride = 1, int padding = 0, Device device = Device::CPU)
        : stride(stride), padding(padding) {
        float std = std::sqrt(2.0f / (float)(in_channels * kernel_size + out_channels * kernel_size));
        W = Tensor::randn({out_channels, in_channels, kernel_size}, 0.0f, std, device, /*requires_grad=*/true);
        b = Tensor::zeros({out_channels}, device, /*requires_grad=*/true);

        register_parameter("weight", W);
        register_parameter("bias", b);
    }

    Tensor forward(const Tensor& x) {
        return conv1d(x, W, b, stride, padding);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 11. Embedding Layer
class Embedding : public Module {
public:
    Tensor weight;
    int64_t num_embeddings;
    int64_t embedding_dim;

    Embedding(int64_t num_embeddings, int64_t embedding_dim, Device device = Device::CPU)
        : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
        float std = std::sqrt(1.0f / (float)embedding_dim);
        weight = Tensor::randn({num_embeddings, embedding_dim}, 0.0f, std, device, /*requires_grad=*/true);
        register_parameter("weight", weight);
    }

    Tensor forward(const Tensor& indices) {
        return embedding(indices, weight);
    }

    Tensor operator()(const Tensor& indices) {
        return forward(indices);
    }
};

// 12. Structural & Normalization Modules
class Flatten : public Module {
public:
    int start_dim;
    int end_dim;

    Flatten(int start_dim = 1, int end_dim = -1) : start_dim(start_dim), end_dim(end_dim) {}

    Tensor forward(const Tensor& x) {
        return x.flatten(start_dim, end_dim);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

class Softmax : public Module {
public:
    int dim;

    Softmax(int dim = -1) : dim(dim) {}

    Tensor forward(const Tensor& x) {
        return softmax(x, dim);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 13. Multihead Self-Attention
class MultiheadAttention : public Module {
public:
    int64_t embed_dim;
    int64_t num_heads;
    int64_t head_dim;
    float scale;

    std::shared_ptr<Linear> q_proj;
    std::shared_ptr<Linear> k_proj;
    std::shared_ptr<Linear> v_proj;
    std::shared_ptr<Linear> out_proj;

    MultiheadAttention(int64_t embed_dim, int64_t num_heads, Device device = Device::CPU)
        : embed_dim(embed_dim), num_heads(num_heads), head_dim(embed_dim / num_heads),
          scale(1.0f / std::sqrt((float)(embed_dim / num_heads))) {
        
        if (embed_dim % num_heads != 0) {
            throw std::runtime_error("embed_dim must be divisible by num_heads!");
        }

        q_proj = std::make_shared<Linear>(embed_dim, embed_dim, device);
        k_proj = std::make_shared<Linear>(embed_dim, embed_dim, device);
        v_proj = std::make_shared<Linear>(embed_dim, embed_dim, device);
        out_proj = std::make_shared<Linear>(embed_dim, embed_dim, device);

        register_module("q_proj", q_proj);
        register_module("k_proj", k_proj);
        register_module("v_proj", v_proj);
        register_module("out_proj", out_proj);
    }

    // Forward pass for 2D [Seq_Len, Embed_Dim] sequence representation
    Tensor forward(const Tensor& x) {
        // x: [L, D]
        Tensor Q = q_proj->forward(x); // [L, D]
        Tensor K = k_proj->forward(x); // [L, D]
        Tensor V = v_proj->forward(x); // [L, D]

        // Scaled dot-product attention scores: [L, L]
        Tensor K_t = K.transpose(0, 1).contiguous(); // [D, L]
        Tensor scores = Q.matmul(K_t); // [L, L]
        scores = scores * scale;

        // Attention weights: [L, L]
        Tensor attn_weights = softmax(scores, /*dim=*/-1);

        // Context output: [L, D]
        Tensor context = attn_weights.matmul(V); // [L, D]

        return out_proj->forward(context);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 13b. Causal Self-Attention (Autoregressive Masking for LLMs)
class CausalSelfAttention : public Module {
public:
    int64_t embed_dim;
    int64_t num_heads;
    int64_t head_dim;
    float scale;

    std::shared_ptr<Linear> q_proj;
    std::shared_ptr<Linear> k_proj;
    std::shared_ptr<Linear> v_proj;
    std::shared_ptr<Linear> out_proj;
    bool use_flash_attn;

    CausalSelfAttention(int64_t embed_dim, int64_t num_heads, Device device = Device::CPU, bool use_flash_attn = false)
        : embed_dim(embed_dim), num_heads(num_heads), head_dim(embed_dim / num_heads),
          scale(1.0f / std::sqrt((float)(embed_dim / num_heads))), use_flash_attn(use_flash_attn) {
        
        if (embed_dim % num_heads != 0) {
            throw std::runtime_error("embed_dim must be divisible by num_heads!");
        }

        q_proj = std::make_shared<Linear>(embed_dim, embed_dim, device);
        k_proj = std::make_shared<Linear>(embed_dim, embed_dim, device);
        v_proj = std::make_shared<Linear>(embed_dim, embed_dim, device);
        out_proj = std::make_shared<Linear>(embed_dim, embed_dim, device);

        register_module("q_proj", q_proj);
        register_module("k_proj", k_proj);
        register_module("v_proj", v_proj);
        register_module("out_proj", out_proj);
    }

    Tensor forward(const Tensor& x) {
        // x: [L, D]
        int64_t L = x.shape()[0];
        Tensor Q = q_proj->forward(x);
        Tensor K = k_proj->forward(x);
        Tensor V = v_proj->forward(x);

        if (use_flash_attn) {
            Tensor context = flash_attention(Q, K, V, /*is_causal=*/true, scale);
            return out_proj->forward(context);
        }

        Tensor K_t = K.transpose(0, 1).contiguous();
        Tensor scores = Q.matmul(K_t) * scale; // [L, L]

        // Apply causal mask: upper triangle (col > row) masked to -1e9
        Tensor mask = Tensor::triu({L, L}, 1, x.device());
        scores = scores.masked_fill(mask, -1e9f);

        Tensor attn_weights = softmax(scores, -1);
        Tensor context = attn_weights.matmul(V);
        return out_proj->forward(context);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// 13d. FlashAttention-2 Module
class FlashAttention : public Module {
public:
    bool is_causal;
    float scale;

    FlashAttention(bool is_causal = false, float scale = -1.0f)
        : is_causal(is_causal), scale(scale) {}

    Tensor forward(const Tensor& Q, const Tensor& K, const Tensor& V) {
        return flash_attention(Q, K, V, is_causal, scale);
    }

    Tensor operator()(const Tensor& Q, const Tensor& K, const Tensor& V) {
        return forward(Q, K, V);
    }
};

// 13c. Rotary Position Embedding (RoPE)
class RotaryEmbedding : public Module {
public:
    int64_t dim;
    int64_t max_seq_len;
    Tensor cos_cached;
    Tensor sin_cached;

    RotaryEmbedding(int64_t dim, int64_t max_seq_len = 2048, float base = 10000.0f, Device device = Device::CPU)
        : dim(dim), max_seq_len(max_seq_len) {
        int64_t half = dim / 2;
        std::vector<float> inv_freq(half);
        for (int64_t i = 0; i < half; ++i) {
            inv_freq[i] = 1.0f / std::pow(base, (2.0f * (float)i) / (float)dim);
        }

        std::vector<float> cos_vals(max_seq_len * half);
        std::vector<float> sin_vals(max_seq_len * half);

        for (int64_t pos = 0; pos < max_seq_len; ++pos) {
            for (int64_t i = 0; i < half; ++i) {
                float theta = (float)pos * inv_freq[i];
                cos_vals[pos * half + i] = std::cos(theta);
                sin_vals[pos * half + i] = std::sin(theta);
            }
        }

        cos_cached = Tensor(cos_vals, {max_seq_len, half}, device, false);
        sin_cached = Tensor(sin_vals, {max_seq_len, half}, device, false);
    }

    Tensor forward(const Tensor& x) {
        return rotary_embedding(x, cos_cached, sin_cached);
    }
};

// 14. Loss Modules
class MSELoss {
public:
    Tensor forward(const Tensor& pred, const Tensor& target) {
        return mse_loss(pred, target);
    }
    Tensor operator()(const Tensor& pred, const Tensor& target) {
        return forward(pred, target);
    }
};

class CrossEntropyLoss {
public:
    Tensor forward(const Tensor& logits, const Tensor& target) {
        return cross_entropy_loss(logits, target);
    }
    Tensor operator()(const Tensor& logits, const Tensor& target) {
        return forward(logits, target);
    }
};

class BCEWithLogitsLoss {
public:
    Tensor forward(const Tensor& logits, const Tensor& target) {
        return bce_with_logits_loss(logits, target);
    }
    Tensor operator()(const Tensor& logits, const Tensor& target) {
        return forward(logits, target);
    }
};

class L1Loss {
public:
    Tensor forward(const Tensor& pred, const Tensor& target) {
        return l1_loss(pred, target);
    }
    Tensor operator()(const Tensor& pred, const Tensor& target) {
        return forward(pred, target);
    }
};

class SmoothL1Loss {
public:
    float beta;
    SmoothL1Loss(float beta = 1.0f) : beta(beta) {}
    Tensor operator()(const Tensor& pred, const Tensor& target) {
        return smooth_l1_loss(pred, target, beta);
    }
};

class KLDivLoss {
public:
    Tensor operator()(const Tensor& input, const Tensor& target) {
        return kl_div_loss(input, target);
    }
};

// -------------------------------------------------------------
// GRADIENT CLIPPING UTILITY
// -------------------------------------------------------------
inline float clip_grad_norm(const std::vector<Tensor>& params, float max_norm, float norm_type = 2.0f) {
    (void)norm_type;
    float total_sq = 0.0f;
    for (const auto& p : params) {
        Tensor g = p.grad();
        if (g.is_defined()) {
            float n = g.norm(2.0f);
            total_sq += n * n;
        }
    }
    float total_norm = std::sqrt(total_sq);
    if (total_norm > max_norm && total_norm > 0.0f) {
        float clip_coef = max_norm / (total_norm + 1e-6f);
        for (const auto& p : params) {
            Tensor g = p.grad();
            if (g.is_defined()) {
                g.mul_(clip_coef);
            }
        }
    }
    return total_norm;
}

// -------------------------------------------------------------
// OPTIMIZERS (Native GPU & Optimized CPU execution)
// -------------------------------------------------------------
class AdamW {
public:
    std::vector<Tensor> params;
    std::vector<Tensor> m;
    std::vector<Tensor> v;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int t;

    AdamW(const std::vector<Tensor>& parameters, float learning_rate = 0.001f, 
          float b1 = 0.9f, float b2 = 0.999f, float epsilon = 1e-8f, float weight_decay = 0.01f)
        : params(parameters), lr(learning_rate), beta1(b1), beta2(b2), eps(epsilon),
          weight_decay(weight_decay), t(0) {
        
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
        float correction1 = 1.0f - std::pow(beta1, (float)t);
        float correction2 = 1.0f - std::pow(beta2, (float)t);

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor& p = params[i];
            Tensor grad = p.grad();
            if (!grad.impl) continue;

            int64_t total = p.numel();

            if (p.device() == Device::CUDA) {
                // Native GPU optimizer step: Zero PCIe bounce!
                cuda::launch_adamw_step(p.impl->data(), grad.impl->data(),
                                        m[i].impl->data(), v[i].impl->data(),
                                        lr, beta1, beta2, eps, weight_decay,
                                        correction1, correction2, (int)total);
            } else {
                // Optimized CPU step
                float* p_data = p.impl->data();
                const float* g_data = grad.impl->data();
                float* m_data = m[i].impl->data();
                float* v_data = v[i].impl->data();

                for (int64_t j = 0; j < total; ++j) {
                    if (weight_decay != 0.0f) {
                        p_data[j] -= lr * weight_decay * p_data[j];
                    }

                    m_data[j] = beta1 * m_data[j] + (1.0f - beta1) * g_data[j];
                    v_data[j] = beta2 * v_data[j] + (1.0f - beta2) * g_data[j] * g_data[j];

                    float m_hat = m_data[j] / correction1;
                    float v_hat = v_data[j] / correction2;

                    p_data[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                }
            }
            p.impl->bump_version();
        }
    }
};

// Aliased Adam for standard API compatibility
class Adam : public AdamW {
public:
    Adam(const std::vector<Tensor>& parameters, float learning_rate = 0.01f, 
         float b1 = 0.9f, float b2 = 0.999f, float epsilon = 1e-8f)
        : AdamW(parameters, learning_rate, b1, b2, epsilon, /*weight_decay=*/0.0f) {}
};

// SGD Optimizer with Momentum & Nesterov
class SGD {
public:
    std::vector<Tensor> params;
    std::vector<Tensor> v;
    float lr;
    float momentum;
    float weight_decay;
    bool nesterov;

    SGD(const std::vector<Tensor>& parameters, float learning_rate = 0.01f,
        float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false)
        : params(parameters), lr(learning_rate), momentum(momentum),
          weight_decay(weight_decay), nesterov(nesterov) {
        
        for (const auto& p : params) {
            v.push_back(Tensor::zeros(p.shape(), p.device()));
        }
    }

    void zero_grad() {
        for (auto& p : params) {
            p.zero_grad();
        }
    }

    void step() {
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor& p = params[i];
            Tensor grad = p.grad();
            if (!grad.impl) continue;

            int64_t total = p.numel();

            if (p.device() == Device::CUDA) {
                cuda::launch_sgd_step(p.impl->data(), grad.impl->data(), v[i].impl->data(),
                                      lr, momentum, weight_decay, nesterov, (int)total);
            } else {
                float* p_data = p.impl->data();
                const float* g_data = grad.impl->data();
                float* v_data = v[i].impl->data();

                for (int64_t j = 0; j < total; ++j) {
                    float g_val = g_data[j];
                    if (weight_decay != 0.0f) {
                        g_val += weight_decay * p_data[j];
                    }

                    if (momentum != 0.0f) {
                        v_data[j] = momentum * v_data[j] + g_val;
                        if (nesterov) {
                            g_val = g_val + momentum * v_data[j];
                        } else {
                            g_val = v_data[j];
                        }
                    }

                    p_data[j] -= lr * g_val;
                }
            }
            p.impl->bump_version();
        }
    }
};

// -------------------------------------------------------------
// LEARNING RATE SCHEDULERS
// -------------------------------------------------------------
class LRScheduler {
public:
    float initial_lr;
    int last_epoch = 0;

    LRScheduler(float lr) : initial_lr(lr) {}
    virtual ~LRScheduler() = default;
    virtual float get_lr() const = 0;
    virtual void step(AdamW& optimizer) {
        last_epoch++;
        optimizer.lr = get_lr();
    }
    virtual void step(SGD& optimizer) {
        last_epoch++;
        optimizer.lr = get_lr();
    }
};

class StepLR : public LRScheduler {
public:
    int step_size;
    float gamma;

    StepLR(float lr, int step_size, float gamma = 0.1f)
        : LRScheduler(lr), step_size(step_size), gamma(gamma) {}

    float get_lr() const override {
        int steps = last_epoch / step_size;
        return initial_lr * std::pow(gamma, (float)steps);
    }
};

class CosineAnnealingLR : public LRScheduler {
public:
    int T_max;
    float eta_min;

    CosineAnnealingLR(float lr, int T_max, float eta_min = 0.0f)
        : LRScheduler(lr), T_max(T_max), eta_min(eta_min) {}

    float get_lr() const override {
        if (last_epoch >= T_max) return eta_min;
        const float pi = 3.14159265358979323846f;
        return eta_min + 0.5f * (initial_lr - eta_min) * (1.0f + std::cos(pi * (float)last_epoch / (float)T_max));
    }
};

class ExponentialLR : public LRScheduler {
public:
    float gamma;
    
    ExponentialLR(float lr, float gamma = 0.95f)
        : LRScheduler(lr), gamma(gamma) {}
    
    float get_lr() const override {
        return initial_lr * std::pow(gamma, (float)last_epoch);
    }
};

class PolynomialLR : public LRScheduler {
public:
    float power;
    int max_decay_steps;

    PolynomialLR(float lr, int max_decay_steps, float power = 1.0f)
        : LRScheduler(lr), power(power), max_decay_steps(max_decay_steps) {}

    float get_lr() const override {
        if (last_epoch >= max_decay_steps) return 0.0f;
        float decay_factor = std::pow(1.0f - (float)last_epoch / (float)max_decay_steps, power);
        return initial_lr * decay_factor;
    }
};

// 15. Quantized Linear Modules (INT8 & INT4)
class QuantizedLinear : public Module {
public:
    quantization::QuantizedTensor q_weight;
    Tensor bias;
    int64_t in_features;
    int64_t out_features;

    QuantizedLinear(int64_t in_features, int64_t out_features, Device device = Device::CPU)
        : in_features(in_features), out_features(out_features) {
        float std = std::sqrt(2.0f / (in_features + out_features));
        Tensor W = Tensor::randn({in_features, out_features}, 0.0f, std, device, false);
        q_weight = quantization::quantize_int8(W, quantization::QuantizationScheme::PerChannelSymmetric);
        bias = Tensor::zeros({1, out_features}, device, false);
        register_parameter("qdata", q_weight.qdata);
        register_parameter("scales", q_weight.scales);
        register_parameter("bias", bias);
    }

    static std::shared_ptr<QuantizedLinear> from_float(const Linear& float_linear) {
        auto ql = std::make_shared<QuantizedLinear>(float_linear.W.shape()[0], float_linear.W.shape()[1], float_linear.W.device());
        ql->q_weight = quantization::quantize_int8(float_linear.W, quantization::QuantizationScheme::PerChannelSymmetric);
        ql->bias = float_linear.b.clone();
        ql->register_parameter("qdata", ql->q_weight.qdata);
        ql->register_parameter("scales", ql->q_weight.scales);
        ql->register_parameter("bias", ql->bias);
        return ql;
    }

    Tensor forward(const Tensor& x) {
        return quantization::quantized_matmul_int8(x, q_weight, bias);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

class QuantizedLinear4Bit : public Module {
public:
    quantization::QuantizedTensor q_weight;
    Tensor bias;
    int64_t in_features;
    int64_t out_features;
    int64_t group_size;

    QuantizedLinear4Bit(int64_t in_features, int64_t out_features, int64_t group_size = 32, Device device = Device::CPU)
        : in_features(in_features), out_features(out_features), group_size(group_size) {
        float std = std::sqrt(2.0f / (in_features + out_features));
        Tensor W = Tensor::randn({in_features, out_features}, 0.0f, std, device, false);
        q_weight = quantization::quantize_int4(W, group_size);
        bias = Tensor::zeros({1, out_features}, device, false);
        register_parameter("qdata", q_weight.qdata);
        register_parameter("scales", q_weight.scales);
        register_parameter("bias", bias);
    }

    static std::shared_ptr<QuantizedLinear4Bit> from_float(const Linear& float_linear, int64_t group_size = 32) {
        auto ql = std::make_shared<QuantizedLinear4Bit>(float_linear.W.shape()[0], float_linear.W.shape()[1], group_size, float_linear.W.device());
        ql->q_weight = quantization::quantize_int4(float_linear.W, group_size);
        ql->bias = float_linear.b.clone();
        ql->register_parameter("qdata", ql->q_weight.qdata);
        ql->register_parameter("scales", ql->q_weight.scales);
        ql->register_parameter("bias", ql->bias);
        return ql;
    }

    Tensor forward(const Tensor& x) {
        return quantization::quantized_matmul_int4(x, q_weight, bias);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// --------------------------------------------------------------------------
// Multi-Threaded Data Parallelism Wrapper (nn::DataParallel)
// --------------------------------------------------------------------------
template <typename ModuleType>
class DataParallel : public Module {
public:
    std::shared_ptr<ModuleType> module;
    int num_devices;

    DataParallel(std::shared_ptr<ModuleType> mod, int num_devices = 2)
        : module(mod), num_devices(std::max(1, num_devices)) {
        register_module("module", module);
    }

    Tensor forward(const Tensor& x) override {
        int64_t total_batch = x.shape()[0];
        if (num_devices <= 1 || total_batch < num_devices) {
            return module->forward(x);
        }

        int64_t chunk_size = total_batch / num_devices;
        std::vector<Tensor> chunks;
        for (int i = 0; i < num_devices; ++i) {
            int64_t start = i * chunk_size;
            int64_t len = (i == num_devices - 1) ? (total_batch - start) : chunk_size;
            chunks.push_back(x.narrow(0, start, len));
        }

        std::vector<Tensor> outputs(num_devices);
        #pragma omp parallel for if(num_devices > 1)
        for (int i = 0; i < num_devices; ++i) {
            outputs[i] = module->forward(chunks[i]);
        }

        return cat(outputs, 0);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }

    void train(bool mode = true) override {
        Module::train(mode);
        if (module) module->train(mode);
    }
};

template <typename ModuleType>
inline std::shared_ptr<DataParallel<ModuleType>> make_data_parallel(std::shared_ptr<ModuleType> module, int num_devices = 2) {
    return std::make_shared<DataParallel<ModuleType>>(module, num_devices);
}

} // namespace nn
} // namespace sovereigngrad

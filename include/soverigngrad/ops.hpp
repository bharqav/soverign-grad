#pragma once

#include "soverigngrad/tensor.hpp"
#include "soverigngrad/autograd.hpp"
#include <vector>

namespace sovereigngrad {

// Broadcasting helpers
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape_a, const std::vector<int64_t>& shape_b);
Tensor reduce_gradient(const Tensor& grad, const std::vector<int64_t>& target_shape);

// Helper to accumulate gradients in the backward pass
void accumulate_grad(std::shared_ptr<TensorImpl> impl, const Tensor& grad);

// Autograd Node Subclasses representing mathematical operations
struct AddNode : public Node {
    AddNode(const Tensor& out, const Tensor& a, const Tensor& b);
    void backward(const Tensor& grad_output) override;
};

struct MulNode : public Node {
    Tensor saved_a;
    Tensor saved_b;

    MulNode(const Tensor& out, const Tensor& a, const Tensor& b);
    void backward(const Tensor& grad_output) override;
};

struct ReLUNode : public Node {
    Tensor saved_out; // Saved forward output for ReLU derivative check

    ReLUNode(const Tensor& out, const Tensor& in);
    void backward(const Tensor& grad_output) override;
};

struct MatMulNode : public Node {
    Tensor saved_a;
    Tensor saved_b;

    MatMulNode(const Tensor& out, const Tensor& a, const Tensor& b);
    void backward(const Tensor& grad_output) override;
};

// Loss function Node: Softmax Cross Entropy
struct SoftmaxCrossEntropyNode : public Node {
    Tensor saved_probs;
    Tensor saved_targets;

    SoftmaxCrossEntropyNode(const Tensor& out, const Tensor& logits, const Tensor& probs, const Tensor& targets);
    void backward(const Tensor& grad_output) override;
};

// Conv2D Node for autograd
struct Conv2DNode : public Node {
    Tensor saved_input;
    Tensor saved_weight;
    Tensor saved_bias;
    int stride;
    int padding;

    Conv2DNode(const Tensor& out, const Tensor& input, const Tensor& weight, const Tensor& bias, int stride, int padding);
    void backward(const Tensor& grad_output) override;
};

// BatchNorm2d Node for autograd
struct BatchNorm2dNode : public Node {
    Tensor saved_input;
    Tensor saved_gamma;
    Tensor saved_beta;
    Tensor saved_mean;
    Tensor saved_var;
    float eps;

    BatchNorm2dNode(const Tensor& out, const Tensor& input, const Tensor& gamma, const Tensor& beta,
                    const Tensor& mean, const Tensor& var, float eps);
    void backward(const Tensor& grad_output) override;
};

// Forward operations declarations
Tensor add_forward(const Tensor& a, const Tensor& b);
Tensor mul_forward(const Tensor& a, const Tensor& b);
Tensor relu_forward(const Tensor& in);
Tensor matmul_forward(const Tensor& a, const Tensor& b);
Tensor softmax_cross_entropy_forward(const Tensor& logits, const Tensor& targets);
Tensor softmax_cross_entropy(const Tensor& logits, const Tensor& targets);

Tensor conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, int stride, int padding);
Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias, int stride = 1, int padding = 0);

Tensor batch_norm2d_forward(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                            const Tensor& mean, const Tensor& var, float eps);
Tensor batch_norm2d(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                    Tensor& running_mean, Tensor& running_var, float momentum = 0.1f, float eps = 1e-5f, bool training = true);

} // namespace sovereigngrad

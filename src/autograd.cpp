#include "sovereigngrad/autograd.hpp"
#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/cuda_mock.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace sovereigngrad {

static thread_local bool g_autograd_enabled = true;

bool AutoGradMode::is_enabled() {
    return g_autograd_enabled;
}

void AutoGradMode::set_enabled(bool enabled) {
    g_autograd_enabled = enabled;
}

// Node constructor
Node::Node(const std::string& op_name, const std::vector<Tensor>& inputs) : op_name(op_name) {
    for (const auto& t : inputs) {
        if (t.impl) {
            this->inputs.push_back(t.impl);
            this->saved_versions.push_back(t.impl->version);
        }
    }
}

void Node::check_versions() const {
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i] && inputs[i]->version != saved_versions[i]) {
            throw std::runtime_error(
                "One of the variables needed for gradient computation has been modified by an in-place operation: "
                "Op [" + op_name + "] expected version " + std::to_string(saved_versions[i]) + 
                " but got version " + std::to_string(inputs[i]->version)
            );
        }
    }
}

void AutogradEngine::topological_sort(std::shared_ptr<TensorImpl> tensor, 
                                      std::vector<std::shared_ptr<TensorImpl>>& sorted, 
                                      std::unordered_set<TensorImpl*>& visited) {
    if (!tensor || visited.count(tensor.get())) return;
    visited.insert(tensor.get());

    if (tensor->creator) {
        for (const auto& input : tensor->creator->inputs) {
            topological_sort(input, sorted, visited);
        }
    }
    sorted.push_back(tensor);
}

void AutogradEngine::backward(Tensor& loss, bool retain_graph) {
    if (loss.numel() != 1) {
        throw std::runtime_error("backward() can only be called on scalar tensors (numel == 1)!");
    }

    // 1. Initialize loss gradient to 1.0
    if (!loss.impl->grad) {
        loss.impl->grad = std::make_shared<TensorImpl>(std::vector<int64_t>{1}, loss.device());
    }
    float* grad_ptr = loss.impl->grad->data();
    if (loss.device() == Device::CPU) {
        grad_ptr[0] = 1.0f;
    } else {
        float one = 1.0f;
        CUDA_CHECK(cudaMemcpy(grad_ptr, &one, sizeof(float), cudaMemcpyHostToDevice));
    }

    // 2. Perform Topological Sort
    std::vector<std::shared_ptr<TensorImpl>> sorted;
    std::unordered_set<TensorImpl*> visited;
    topological_sort(loss.impl, sorted, visited);

    // 3. Reverse topological sort to process from outputs to inputs
    std::reverse(sorted.begin(), sorted.end());

    // 4. Iterate backward through the computational graph
    for (const auto& tensor_impl : sorted) {
        if (tensor_impl->creator) {
            // Ensure the tensor has a valid gradient
            if (!tensor_impl->grad) {
                tensor_impl->grad = std::make_shared<TensorImpl>(tensor_impl->shape, tensor_impl->device);
                size_t bytes = tensor_impl->numel() * sizeof(float);
                if (tensor_impl->device == Device::CPU) {
                    std::memset(tensor_impl->grad->data(), 0, bytes);
                } else {
                    CUDA_CHECK(cudaMemset(tensor_impl->grad->data(), 0, bytes));
                }
            }

            // Verify in-place safety
            tensor_impl->creator->check_versions();

            Tensor grad_out(tensor_impl->grad);

            // Execute hooks if registered
            if (!tensor_impl->hooks.empty()) {
                for (const auto& hook : tensor_impl->hooks) {
                    grad_out = hook(grad_out);
                }
                tensor_impl->grad = grad_out.impl;
            }

            tensor_impl->creator->backward(grad_out);

            // If not retaining graph, clear the creator to free intermediate activation graph
            if (!retain_graph && tensor_impl != loss.impl) {
                tensor_impl->creator = nullptr;
            }
        }
    }
}

} // namespace sovereigngrad

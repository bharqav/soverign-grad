#include "soverigngrad/autograd.hpp"
#include "soverigngrad/tensor.hpp"
#include "soverigngrad/cuda_mock.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace sovereigngrad {

// Node constructor
Node::Node(const std::string& op_name, const std::vector<Tensor>& inputs) : op_name(op_name) {
    for (const auto& t : inputs) {
        if (t.impl) {
            this->inputs.push_back(t.impl);
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

void AutogradEngine::backward(Tensor& loss) {
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
        cudaMemcpy(grad_ptr, &one, sizeof(float), cudaMemcpyHostToDevice);
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
                // Initialize to zeros if gradient is empty (e.g. not accumulated yet)
                tensor_impl->grad = std::make_shared<TensorImpl>(tensor_impl->shape, tensor_impl->device);
                size_t bytes = tensor_impl->numel() * sizeof(float);
                if (tensor_impl->device == Device::CPU) {
                    std::memset(tensor_impl->grad->data(), 0, bytes);
                } else {
                    cudaMemset(tensor_impl->grad->data(), 0, bytes);
                }
            }

            Tensor grad_out(tensor_impl->grad);
            tensor_impl->creator->backward(grad_out);
        }
    }
}

} // namespace sovereigngrad

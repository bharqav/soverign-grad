#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_set>

namespace sovereigngrad {

class Tensor;
struct TensorImpl;

struct Node {
    std::string op_name;
    std::vector<std::shared_ptr<TensorImpl>> inputs;

    Node(const std::string& op_name, const std::vector<Tensor>& inputs);
    virtual ~Node() = default;

    // Virtual backward pass interface
    // Receives the output gradient and accumulates calculated input gradients
    virtual void backward(const Tensor& grad_output) = 0;
};

class AutogradEngine {
public:
    // Performs topological sort of the computational DAG and triggers the backward pass
    static void backward(Tensor& loss);

private:
    static void topological_sort(std::shared_ptr<TensorImpl> tensor, 
                                 std::vector<std::shared_ptr<TensorImpl>>& sorted, 
                                 std::unordered_set<TensorImpl*>& visited);
};

} // namespace sovereigngrad

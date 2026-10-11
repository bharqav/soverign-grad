#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_set>
#include <functional>
#include <cstdint>

namespace sovereigngrad {

class Tensor;
struct TensorImpl;

class AutoGradMode {
public:
    static bool is_enabled();
    static void set_enabled(bool enabled);
};

class NoGradGuard {
public:
    NoGradGuard() {
        prev_enabled_ = AutoGradMode::is_enabled();
        AutoGradMode::set_enabled(false);
    }

    ~NoGradGuard() {
        AutoGradMode::set_enabled(prev_enabled_);
    }

    // Non-copyable
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_enabled_;
};

struct Node {
    std::string op_name;
    std::vector<std::shared_ptr<TensorImpl>> inputs;
    std::vector<uint64_t> saved_versions;

    Node(const std::string& op_name, const std::vector<Tensor>& inputs);
    virtual ~Node() = default;

    // Check if any input was mutated in-place after graph construction
    void check_versions() const;

    // Virtual backward pass interface
    virtual void backward(const Tensor& grad_output) = 0;
};

class AutogradEngine {
public:
    // Performs topological sort of the computational DAG and triggers the backward pass
    static void backward(Tensor& loss, bool retain_graph = false);

private:
    static void topological_sort(std::shared_ptr<TensorImpl> tensor, 
                                 std::vector<std::shared_ptr<TensorImpl>>& sorted, 
                                 std::unordered_set<TensorImpl*>& visited);
};

} // namespace sovereigngrad

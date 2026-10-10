#pragma once

#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/ops.hpp"
#include "sovereigngrad/module.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>

namespace sovereigngrad {
namespace jit {

enum class OpType {
    Input,
    Constant,
    Matmul,
    Add,
    Sub,
    Mul,
    Div,
    GELU,
    ReLU,
    SiLU,
    LayerNorm,
    RMSNorm,
    // Fused Operations
    FusedLinearActivation,
    FusedResidualNorm,
    FusedScaleShift
};

enum class ActivationType {
    None,
    GELU,
    ReLU,
    SiLU
};

enum class NormType {
    LayerNorm,
    RMSNorm
};

struct Node {
    int id;
    std::string name;
    OpType op;
    std::vector<int> input_ids;
    Tensor value; // Weights / biases / constants

    // Fused parameters
    ActivationType act_type = ActivationType::None;
    NormType norm_type = NormType::LayerNorm;
    float eps = 1e-5f;
};

class Graph {
public:
    std::vector<Node> nodes;
    std::vector<int> output_ids;

    int add_node(const std::string& name, OpType op, const std::vector<int>& input_ids, const Tensor& value = Tensor()) {
        int id = (int)nodes.size();
        Node n;
        n.id = id;
        n.name = name;
        n.op = op;
        n.input_ids = input_ids;
        n.value = value;
        nodes.push_back(n);
        return id;
    }

    void optimize();
    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) const;
    void print() const;
};

class CompiledModule {
public:
    Graph graph;
    std::string name;

    CompiledModule(const Graph& g, const std::string& name = "CompiledModule")
        : graph(g), name(name) {}

    Tensor forward(const Tensor& x) {
        auto outs = graph.execute({x});
        if (outs.empty()) throw std::runtime_error("CompiledModule produced no output!");
        return outs[0];
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};

// Optimizer passes
void pass_fuse_linear_activation(Graph& g);
void pass_fuse_residual_norm(Graph& g);
void pass_fuse_scale_shift(Graph& g);

// Graph compilation helper
CompiledModule compile(const Graph& g);

} // namespace jit
} // namespace sovereigngrad

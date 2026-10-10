#include "sovereigngrad/jit.hpp"
#include "sovereigngrad/ops.hpp"
#include "sovereigngrad/simd.hpp"
#include <iostream>
#include <iomanip>
#include <unordered_map>

namespace sovereigngrad {
namespace jit {

void pass_fuse_linear_activation(Graph& g) {
    // Detect Pattern: (Matmul(x, W) + Bias) -> (GELU | ReLU | SiLU)
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        if (g.nodes[i].op == OpType::GELU || g.nodes[i].op == OpType::ReLU || g.nodes[i].op == OpType::SiLU) {
            if (g.nodes[i].input_ids.size() == 1) {
                int add_id = g.nodes[i].input_ids[0];
                if (add_id >= 0 && (size_t)add_id < g.nodes.size() && g.nodes[add_id].op == OpType::Add) {
                    // Check if add has input from Matmul
                    int matmul_id = -1;
                    int bias_id = -1;
                    for (int in_id : g.nodes[add_id].input_ids) {
                        if (in_id >= 0 && (size_t)in_id < g.nodes.size()) {
                            if (g.nodes[in_id].op == OpType::Matmul) matmul_id = in_id;
                            else bias_id = in_id;
                        }
                    }

                    if (matmul_id != -1 && bias_id != -1) {
                        // Create FusedLinearActivation node
                        Node fused;
                        fused.id = g.nodes[i].id;
                        fused.name = "FusedLinearActivation_" + std::to_string(fused.id);
                        fused.op = OpType::FusedLinearActivation;
                        fused.input_ids = {g.nodes[matmul_id].input_ids[0], g.nodes[matmul_id].input_ids[1], bias_id};
                        
                        if (g.nodes[i].op == OpType::GELU) fused.act_type = ActivationType::GELU;
                        else if (g.nodes[i].op == OpType::ReLU) fused.act_type = ActivationType::ReLU;
                        else fused.act_type = ActivationType::SiLU;

                        g.nodes[i] = fused;
                    }
                }
            }
        }
    }
}

void pass_fuse_residual_norm(Graph& g) {
    // Detect Pattern: (x + residual) -> (LayerNorm | RMSNorm)
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        if (g.nodes[i].op == OpType::LayerNorm || g.nodes[i].op == OpType::RMSNorm) {
            if (!g.nodes[i].input_ids.empty()) {
                int add_id = g.nodes[i].input_ids[0];
                if (add_id >= 0 && (size_t)add_id < g.nodes.size() && g.nodes[add_id].op == OpType::Add) {
                    if (g.nodes[add_id].input_ids.size() == 2) {
                        Node fused;
                        fused.id = g.nodes[i].id;
                        fused.name = "FusedResidualNorm_" + std::to_string(fused.id);
                        fused.op = OpType::FusedResidualNorm;
                        
                        // Inputs: x, residual, gamma, (beta)
                        fused.input_ids = {g.nodes[add_id].input_ids[0], g.nodes[add_id].input_ids[1]};
                        for (size_t k = 1; k < g.nodes[i].input_ids.size(); ++k) {
                            fused.input_ids.push_back(g.nodes[i].input_ids[k]);
                        }

                        fused.norm_type = (g.nodes[i].op == OpType::RMSNorm) ? NormType::RMSNorm : NormType::LayerNorm;
                        fused.eps = g.nodes[i].eps;

                        g.nodes[i] = fused;
                    }
                }
            }
        }
    }
}

void pass_fuse_scale_shift(Graph& g) {
    // Detect Pattern: (x * scalar) + bias -> FusedScaleShift (FMA)
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        if (g.nodes[i].op == OpType::Add && g.nodes[i].input_ids.size() == 2) {
            int mul_id = -1, other_id = -1;
            for (int in_id : g.nodes[i].input_ids) {
                if (in_id >= 0 && (size_t)in_id < g.nodes.size() && g.nodes[in_id].op == OpType::Mul) {
                    mul_id = in_id;
                } else {
                    other_id = in_id;
                }
            }
            if (mul_id != -1 && other_id != -1) {
                Node fused;
                fused.id = g.nodes[i].id;
                fused.name = "FusedScaleShift_" + std::to_string(fused.id);
                fused.op = OpType::FusedScaleShift;
                fused.input_ids = {g.nodes[mul_id].input_ids[0], g.nodes[mul_id].input_ids[1], other_id};
                g.nodes[i] = fused;
            }
        }
    }
}

void Graph::optimize() {
    pass_fuse_linear_activation(*this);
    pass_fuse_residual_norm(*this);
    pass_fuse_scale_shift(*this);
}

std::vector<Tensor> Graph::execute(const std::vector<Tensor>& inputs) const {
    std::unordered_map<int, Tensor> env;
    int input_idx = 0;

    for (const auto& node : nodes) {
        if (node.op == OpType::Input) {
            if (input_idx < (int)inputs.size()) {
                env[node.id] = inputs[input_idx++];
            } else {
                throw std::runtime_error("Not enough inputs provided for Graph execution!");
            }
        } else if (node.op == OpType::Constant) {
            env[node.id] = node.value;
        } else if (node.op == OpType::Matmul) {
            env[node.id] = env.at(node.input_ids[0]).matmul(env.at(node.input_ids[1]));
        } else if (node.op == OpType::Add) {
            env[node.id] = env.at(node.input_ids[0]) + env.at(node.input_ids[1]);
        } else if (node.op == OpType::Sub) {
            env[node.id] = env.at(node.input_ids[0]) - env.at(node.input_ids[1]);
        } else if (node.op == OpType::Mul) {
            env[node.id] = env.at(node.input_ids[0]) * env.at(node.input_ids[1]);
        } else if (node.op == OpType::Div) {
            env[node.id] = env.at(node.input_ids[0]) / env.at(node.input_ids[1]);
        } else if (node.op == OpType::GELU) {
            env[node.id] = env.at(node.input_ids[0]).gelu();
        } else if (node.op == OpType::ReLU) {
            env[node.id] = env.at(node.input_ids[0]).relu();
        } else if (node.op == OpType::SiLU) {
            env[node.id] = env.at(node.input_ids[0]).silu();
        } else if (node.op == OpType::LayerNorm) {
            const Tensor& x = env.at(node.input_ids[0]);
            const Tensor& g_val = env.at(node.input_ids[1]);
            const Tensor& b_val = (node.input_ids.size() > 2) ? env.at(node.input_ids[2]) : Tensor();
            env[node.id] = layer_norm(x, g_val, b_val, node.eps);
        } else if (node.op == OpType::RMSNorm) {
            const Tensor& x = env.at(node.input_ids[0]);
            const Tensor& g_val = env.at(node.input_ids[1]);
            env[node.id] = rms_norm(x, g_val, node.eps);
        } else if (node.op == OpType::FusedLinearActivation) {
            const Tensor& x = env.at(node.input_ids[0]);
            const Tensor& W = env.at(node.input_ids[1]);
            const Tensor& b_val = env.at(node.input_ids[2]);
            if (node.act_type == ActivationType::GELU) {
                env[node.id] = fused_linear_gelu(x, W, b_val);
            } else {
                Tensor lin = x.matmul(W) + b_val;
                env[node.id] = (node.act_type == ActivationType::ReLU) ? lin.relu() : lin.silu();
            }
        } else if (node.op == OpType::FusedResidualNorm) {
            const Tensor& x = env.at(node.input_ids[0]);
            const Tensor& res = env.at(node.input_ids[1]);
            const Tensor& gamma = env.at(node.input_ids[2]);
            if (node.norm_type == NormType::LayerNorm) {
                const Tensor& beta = (node.input_ids.size() > 3) ? env.at(node.input_ids[3]) : Tensor();
                env[node.id] = fused_add_layernorm(x, res, gamma, beta, node.eps);
            } else {
                env[node.id] = rms_norm(x + res, gamma, node.eps);
            }
        } else if (node.op == OpType::FusedScaleShift) {
            const Tensor& x = env.at(node.input_ids[0]);
            const Tensor& scale = env.at(node.input_ids[1]);
            const Tensor& shift = env.at(node.input_ids[2]);
            env[node.id] = (x * scale) + shift;
        }
    }

    std::vector<Tensor> results;
    if (output_ids.empty() && !nodes.empty()) {
        results.push_back(env.at(nodes.back().id));
    } else {
        for (int oid : output_ids) {
            results.push_back(env.at(oid));
        }
    }
    return results;
}

void Graph::print() const {
    std::cout << "\n=== SovereignGrad JIT Graph (Nodes: " << nodes.size() << ") ===\n";
    for (const auto& n : nodes) {
        std::cout << "  %" << n.id << " = " << n.name << " (Op: " << (int)n.op << ") [";
        for (size_t i = 0; i < n.input_ids.size(); ++i) {
            std::cout << "%" << n.input_ids[i] << (i + 1 < n.input_ids.size() ? ", " : "");
        }
        std::cout << "]\n";
    }
    std::cout << "===========================================\n";
}

CompiledModule compile(const Graph& g) {
    Graph opt_g = g;
    opt_g.optimize();
    return CompiledModule(opt_g);
}

} // namespace jit
} // namespace sovereigngrad

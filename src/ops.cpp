#include "sovereigngrad/ops.hpp"
#include "sovereigngrad/autograd.hpp"
#include "sovereigngrad/cuda_kernels.cuh"
#include "sovereigngrad/simd.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <iostream>
#include <random>

namespace sovereigngrad {

static inline bool op_requires_grad(bool req) {
    return req && AutoGradMode::is_enabled();
}

// Helper to convert flat index to coordinates
static std::vector<int64_t> get_coords(int64_t flat_idx, const std::vector<int64_t>& shape) {
    std::vector<int64_t> coords(shape.size());
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        coords[i] = flat_idx % shape[i];
        flat_idx /= shape[i];
    }
    return coords;
}

// Helper to map coordinates from a larger broadcasted output space to a smaller input space
static std::vector<int64_t> map_coords(const std::vector<int64_t>& out_coords, const std::vector<int64_t>& in_shape) {
    int rank_out = (int)out_coords.size();
    int rank_in = (int)in_shape.size();
    std::vector<int64_t> in_coords(rank_in);
    for (int i = 0; i < rank_in; ++i) {
        if (in_shape[i] == 1) {
            in_coords[i] = 0;
        } else {
            in_coords[i] = out_coords[rank_out - rank_in + i];
        }
    }
    return in_coords;
}

// Broadcasting shape calculation
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape_a, const std::vector<int64_t>& shape_b) {
    std::vector<int64_t> out_shape;
    int size_a = (int)shape_a.size();
    int size_b = (int)shape_b.size();
    int max_size = std::max(size_a, size_b);
    out_shape.resize(max_size);
    
    for (int i = 0; i < max_size; ++i) {
        int64_t dim_a = (i < size_a) ? shape_a[size_a - 1 - i] : 1;
        int64_t dim_b = (i < size_b) ? shape_b[size_b - 1 - i] : 1;
        
        if (dim_a == dim_b) {
            out_shape[max_size - 1 - i] = dim_a;
        } else if (dim_a == 1) {
            out_shape[max_size - 1 - i] = dim_b;
        } else if (dim_b == 1) {
            out_shape[max_size - 1 - i] = dim_a;
        } else {
            throw std::runtime_error("Incompatible shapes for broadcasting!");
        }
    }
    return out_shape;
}

void accumulate_grad(std::shared_ptr<TensorImpl> target, const Tensor& grad) {
    if (!target) return;
    if (!target->grad) {
        target->grad = grad.impl;
        return;
    }

    // Accumulate into existing gradient
    Tensor current_grad(target->grad);
    if (current_grad.device() != grad.device()) {
        throw std::runtime_error("Device mismatch during gradient accumulation!");
    }

    if (current_grad.device() == Device::CPU) {
        float* d1 = current_grad.impl->data();
        const float* d2 = grad.impl->data();
        int64_t n = current_grad.numel();
        for (int64_t i = 0; i < n; ++i) {
            d1[i] += d2[i];
        }
    } else {
        cuda::launch_add(current_grad.impl->data(), current_grad.impl->data(), grad.impl->data(), (int)current_grad.numel());
    }
}

// Helper to sum-reduce broadcasted gradients back to original input shape
static Tensor reduce_to_shape(const Tensor& grad, const std::vector<int64_t>& target_shape) {
    if (grad.shape() == target_shape) {
        return grad;
    }

    Tensor cpu_grad = grad.to(Device::CPU);
    Tensor cpu_res(target_shape, Device::CPU);
    std::memset(cpu_res.impl->data(), 0, cpu_res.numel() * sizeof(float));

    int64_t grad_numel = cpu_grad.numel();
    auto grad_shape = cpu_grad.shape();

    for (int64_t i = 0; i < grad_numel; ++i) {
        auto coords = get_coords(i, grad_shape);
        auto target_coords = map_coords(coords, target_shape);
        float current_val = cpu_res.get_element(target_coords);
        float add_val = cpu_grad.get_element(coords);
        cpu_res.set_element(target_coords, current_val + add_val);
    }

    return (grad.device() == Device::CUDA) ? cpu_res.to(Device::CUDA) : cpu_res;
}

// -------------------------------------------------------------
// ARITHMETIC NODES & OPS
// -------------------------------------------------------------
struct AddNode : public Node {
    std::vector<int64_t> shape_a, shape_b;
    AddNode(const Tensor& a, const Tensor& b) 
        : Node("Add", {a, b}), shape_a(a.shape()), shape_b(b.shape()) {}

    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], reduce_to_shape(grad_output, shape_a));
        accumulate_grad(inputs[1], reduce_to_shape(grad_output, shape_b));
    }
};

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        if (a.numel() == 1) return add(a.to(b.device()), b);
        if (b.numel() == 1) return add(a, b.to(a.device()));
        throw std::runtime_error("Device mismatch in add!");
    }
    auto out_shape = broadcast_shapes(a.shape(), b.shape());
    bool req_grad = op_requires_grad(a.requires_grad() || b.requires_grad());
    Tensor out(out_shape, a.device(), req_grad);

    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        int64_t n = a.numel();
        if (a.device() == Device::CPU) {
            simd::vec_add(out.impl->data(), a.impl->data(), b.impl->data(), n);
        } else {
            cuda::launch_add(out.impl->data(), a.impl->data(), b.impl->data(), (int)n);
        }
    } else {
        Tensor ca = a.to(Device::CPU);
        Tensor cb = b.to(Device::CPU);
        Tensor cout(out_shape, Device::CPU, req_grad);
        int64_t total = cout.numel();
        for (int64_t i = 0; i < total; ++i) {
            auto coords = get_coords(i, out_shape);
            float va = ca.get_element(map_coords(coords, ca.shape()));
            float vb = cb.get_element(map_coords(coords, cb.shape()));
            cout.set_element(coords, va + vb);
        }
        if (a.device() == Device::CUDA) {
            CUDA_CHECK(cudaMemcpy(out.impl->data(), cout.impl->data(), total * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            out = cout;
        }
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<AddNode>(a, b));
    }
    return out;
}

struct SubNode : public Node {
    std::vector<int64_t> shape_a, shape_b;
    SubNode(const Tensor& a, const Tensor& b) 
        : Node("Sub", {a, b}), shape_a(a.shape()), shape_b(b.shape()) {}

    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], reduce_to_shape(grad_output, shape_a));
        accumulate_grad(inputs[1], reduce_to_shape(-grad_output, shape_b));
    }
};

Tensor sub(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        if (a.numel() == 1) return sub(a.to(b.device()), b);
        if (b.numel() == 1) return sub(a, b.to(a.device()));
        throw std::runtime_error("Device mismatch in sub!");
    }
    auto out_shape = broadcast_shapes(a.shape(), b.shape());
    bool req_grad = op_requires_grad(a.requires_grad() || b.requires_grad());
    Tensor out(out_shape, a.device(), req_grad);

    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        int64_t n = a.numel();
        if (a.device() == Device::CPU) {
            const float* da = a.impl->data();
            const float* db = b.impl->data();
            float* dout = out.impl->data();
            for (int64_t i = 0; i < n; ++i) dout[i] = da[i] - db[i];
        } else {
            cuda::launch_sub(out.impl->data(), a.impl->data(), b.impl->data(), (int)n);
        }
    } else {
        Tensor ca = a.to(Device::CPU);
        Tensor cb = b.to(Device::CPU);
        Tensor cout(out_shape, Device::CPU, req_grad);
        int64_t total = cout.numel();
        for (int64_t i = 0; i < total; ++i) {
            auto coords = get_coords(i, out_shape);
            float va = ca.get_element(map_coords(coords, ca.shape()));
            float vb = cb.get_element(map_coords(coords, cb.shape()));
            cout.set_element(coords, va - vb);
        }
        if (a.device() == Device::CUDA) {
            CUDA_CHECK(cudaMemcpy(out.impl->data(), cout.impl->data(), total * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            out = cout;
        }
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<SubNode>(a, b));
    }
    return out;
}

struct MulNode : public Node {
    Tensor a_saved, b_saved;
    MulNode(const Tensor& a, const Tensor& b) 
        : Node("Mul", {a, b}), a_saved(a.detach()), b_saved(b.detach()) {}

    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], reduce_to_shape(grad_output * b_saved, a_saved.shape()));
        accumulate_grad(inputs[1], reduce_to_shape(grad_output * a_saved, b_saved.shape()));
    }
};

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        if (a.numel() == 1) return mul(a.to(b.device()), b);
        if (b.numel() == 1) return mul(a, b.to(a.device()));
        throw std::runtime_error("Device mismatch in mul!");
    }
    auto out_shape = broadcast_shapes(a.shape(), b.shape());
    bool req_grad = op_requires_grad(a.requires_grad() || b.requires_grad());
    Tensor out(out_shape, a.device(), req_grad);

    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        int64_t n = a.numel();
        if (a.device() == Device::CPU) {
            simd::vec_mul(out.impl->data(), a.impl->data(), b.impl->data(), n);
        } else {
            cuda::launch_mul(out.impl->data(), a.impl->data(), b.impl->data(), (int)n);
        }
    } else {
        Tensor ca = a.to(Device::CPU);
        Tensor cb = b.to(Device::CPU);
        Tensor cout(out_shape, Device::CPU, req_grad);
        int64_t total = cout.numel();
        for (int64_t i = 0; i < total; ++i) {
            auto coords = get_coords(i, out_shape);
            float va = ca.get_element(map_coords(coords, ca.shape()));
            float vb = cb.get_element(map_coords(coords, cb.shape()));
            cout.set_element(coords, va * vb);
        }
        if (a.device() == Device::CUDA) {
            CUDA_CHECK(cudaMemcpy(out.impl->data(), cout.impl->data(), total * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            out = cout;
        }
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<MulNode>(a, b));
    }
    return out;
}

struct DivNode : public Node {
    Tensor a_saved, b_saved;
    DivNode(const Tensor& a, const Tensor& b) 
        : Node("Div", {a, b}), a_saved(a.detach()), b_saved(b.detach()) {}

    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], reduce_to_shape(grad_output / b_saved, a_saved.shape()));
        accumulate_grad(inputs[1], reduce_to_shape(-grad_output * a_saved / (b_saved * b_saved), b_saved.shape()));
    }
};

Tensor div(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        if (a.numel() == 1) return div(a.to(b.device()), b);
        if (b.numel() == 1) return div(a, b.to(a.device()));
        throw std::runtime_error("Device mismatch in div!");
    }
    auto out_shape = broadcast_shapes(a.shape(), b.shape());
    bool req_grad = op_requires_grad(a.requires_grad() || b.requires_grad());
    Tensor out(out_shape, a.device(), req_grad);

    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        int64_t n = a.numel();
        if (a.device() == Device::CPU) {
            const float* da = a.impl->data();
            const float* db = b.impl->data();
            float* dout = out.impl->data();
            for (int64_t i = 0; i < n; ++i) dout[i] = da[i] / db[i];
        } else {
            cuda::launch_div(out.impl->data(), a.impl->data(), b.impl->data(), (int)n);
        }
    } else {
        Tensor ca = a.to(Device::CPU);
        Tensor cb = b.to(Device::CPU);
        Tensor cout(out_shape, Device::CPU, req_grad);
        int64_t total = cout.numel();
        for (int64_t i = 0; i < total; ++i) {
            auto coords = get_coords(i, out_shape);
            float va = ca.get_element(map_coords(coords, ca.shape()));
            float vb = cb.get_element(map_coords(coords, cb.shape()));
            cout.set_element(coords, va / vb);
        }
        if (a.device() == Device::CUDA) {
            CUDA_CHECK(cudaMemcpy(out.impl->data(), cout.impl->data(), total * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            out = cout;
        }
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<DivNode>(a, b));
    }
    return out;
}

struct NegNode : public Node {
    NegNode(const Tensor& a) : Node("Neg", {a}) {}
    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], -grad_output);
    }
};

Tensor neg(const Tensor& a) {
    Tensor out(a.shape(), a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        const float* da = a.impl->data();
        float* dout = out.impl->data();
        for (int64_t i = 0; i < n; ++i) dout[i] = -da[i];
    } else {
        cuda::launch_neg(out.impl->data(), a.impl->data(), (int)n);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<NegNode>(a));
    }
    return out;
}

// -------------------------------------------------------------
// MATRIX MULTIPLICATION
// -------------------------------------------------------------
struct MatmulNode : public Node {
    Tensor a_saved, b_saved;
    MatmulNode(const Tensor& a, const Tensor& b) 
        : Node("Matmul", {a, b}), a_saved(a.detach()), b_saved(b.detach()) {}

    void backward(const Tensor& grad_output) override {
        // dL/dA = grad_output * B^T
        Tensor b_t = b_saved.transpose(0, 1).contiguous();
        accumulate_grad(inputs[0], grad_output.matmul(b_t));

        // dL/dB = A^T * grad_output
        Tensor a_t = a_saved.transpose(0, 1).contiguous();
        accumulate_grad(inputs[1], a_t.matmul(grad_output));
    }
};

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) throw std::runtime_error("Device mismatch in matmul!");
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("matmul currently requires 2D tensors!");
    }
    int64_t M = a.shape()[0];
    int64_t K1 = a.shape()[1];
    int64_t K2 = b.shape()[0];
    int64_t N = b.shape()[1];
    if (K1 != K2) {
        throw std::runtime_error("Matrix inner dimensions must match for matmul!");
    }

    Tensor a_contig = a.contiguous();
    Tensor b_contig = b.contiguous();
    Tensor out({M, N}, a.device(), op_requires_grad(a.requires_grad() || b.requires_grad()));

    if (a.device() == Device::CPU) {
        const float* da = a_contig.impl->data();
        const float* db = b_contig.impl->data();
        float* dout = out.impl->data();
        std::memset(dout, 0, M * N * sizeof(float));

        const int64_t BLOCK_M = 32;
        const int64_t BLOCK_N = 64;
        const int64_t BLOCK_K = 128;
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(dynamic)
        #endif
        for (int64_t i0 = 0; i0 < M; i0 += BLOCK_M) {
            for (int64_t j0 = 0; j0 < N; j0 += BLOCK_N) {
                int64_t i_max = std::min(M, i0 + BLOCK_M);
                int64_t j_max = std::min(N, j0 + BLOCK_N);
                for (int64_t k0 = 0; k0 < K1; k0 += BLOCK_K) {
                    int64_t k_max = std::min(K1, k0 + BLOCK_K);
                    int64_t blk_k = k_max - k0;

                    int64_t i = i0;
                    for (; i + 8 <= i_max; i += 8) {
                        int64_t j = j0;
                        for (; j + 8 <= j_max; j += 8) {
                            simd::gemm_microkernel_8x8(
                                da + i * K1 + k0,
                                db + k0 * N + j,
                                dout + i * N + j,
                                K1, N, N, blk_k
                            );
                        }
                        for (; j < j_max; ++j) {
                            for (int64_t r = 0; r < 8; ++r) {
                                float acc = 0.0f;
                                for (int64_t k = k0; k < k_max; ++k) {
                                    acc += da[(i + r) * K1 + k] * db[k * N + j];
                                }
                                dout[(i + r) * N + j] += acc;
                            }
                        }
                    }
                    for (; i < i_max; ++i) {
                        for (int64_t k = k0; k < k_max; ++k) {
                            float a_ik = da[i * K1 + k];
                            for (int64_t j = j0; j < j_max; ++j) {
                                dout[i * N + j] += a_ik * db[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        cuda::launch_matmul_tiled(out.impl->data(), a_contig.impl->data(), b_contig.impl->data(), (int)M, (int)N, (int)K1);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<MatmulNode>(a, b));
    }
    return out;
}

// -------------------------------------------------------------
// ACTIVATIONS
// -------------------------------------------------------------
struct ReluNode : public Node {
    Tensor in_saved;
    ReluNode(const Tensor& in) : Node("ReLU", {in}), in_saved(in.detach()) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in(in_saved.shape(), in_saved.device());
        int64_t n = in_saved.numel();
        if (in_saved.device() == Device::CPU) {
            const float* din = in_saved.impl->data();
            const float* dout = grad_output.impl->data();
            float* dres = d_in.impl->data();
            for (int64_t i = 0; i < n; ++i) {
                dres[i] = (din[i] > 0.0f) ? dout[i] : 0.0f;
            }
        } else {
            cuda::launch_relu_backward(d_in.impl->data(), grad_output.impl->data(), in_saved.impl->data(), (int)n);
        }
        accumulate_grad(inputs[0], d_in);
    }
};

Tensor relu(const Tensor& a) {
    Tensor a_contig = a.contiguous();
    Tensor out(a.shape(), a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        simd::vec_relu(out.impl->data(), a_contig.impl->data(), n);
    } else {
        cuda::launch_relu(out.impl->data(), a_contig.impl->data(), (int)n);
    }
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<ReluNode>(a));
    }
    return out;
}

struct GeluNode : public Node {
    Tensor in_saved;
    GeluNode(const Tensor& in) : Node("GELU", {in}), in_saved(in.detach()) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in(in_saved.shape(), in_saved.device());
        int64_t n = in_saved.numel();
        if (in_saved.device() == Device::CPU) {
            const float* x = in_saved.impl->data();
            const float* dout = grad_output.impl->data();
            float* din = d_in.impl->data();
            float c = 0.79788456f;
            for (int64_t i = 0; i < n; ++i) {
                float xi = x[i];
                float u = c * (xi + 0.044715f * xi * xi * xi);
                float th = std::tanh(u);
                float du_dx = c * (1.0f + 3.0f * 0.044715f * xi * xi);
                float grad = 0.5f * (1.0f + th) + 0.5f * xi * (1.0f - th * th) * du_dx;
                din[i] = dout[i] * grad;
            }
        } else {
            cuda::launch_gelu_backward(d_in.impl->data(), grad_output.impl->data(), in_saved.impl->data(), (int)n);
        }
        accumulate_grad(inputs[0], d_in);
    }
};

Tensor gelu(const Tensor& a) {
    Tensor a_contig = a.contiguous();
    Tensor out(a.shape(), a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        simd::vec_gelu(out.impl->data(), a_contig.impl->data(), n);
    } else {
        cuda::launch_gelu(out.impl->data(), a_contig.impl->data(), (int)n);
    }
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<GeluNode>(a));
    }
    return out;
}

struct SiluNode : public Node {
    Tensor in_saved;
    SiluNode(const Tensor& in) : Node("SiLU", {in}), in_saved(in.detach()) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in(in_saved.shape(), in_saved.device());
        int64_t n = in_saved.numel();
        if (in_saved.device() == Device::CPU) {
            const float* x = in_saved.impl->data();
            const float* dout = grad_output.impl->data();
            float* din = d_in.impl->data();
            for (int64_t i = 0; i < n; ++i) {
                float sig = 1.0f / (1.0f + std::exp(-x[i]));
                float grad = sig * (1.0f + x[i] * (1.0f - sig));
                din[i] = dout[i] * grad;
            }
        } else {
            cuda::launch_silu_backward(d_in.impl->data(), grad_output.impl->data(), in_saved.impl->data(), (int)n);
        }
        accumulate_grad(inputs[0], d_in);
    }
};

Tensor silu(const Tensor& a) {
    Tensor a_contig = a.contiguous();
    Tensor out(a.shape(), a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        simd::vec_silu(out.impl->data(), a_contig.impl->data(), n);
    } else {
        cuda::launch_silu(out.impl->data(), a_contig.impl->data(), (int)n);
    }
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<SiluNode>(a));
    }
    return out;
}

struct SigmoidNode : public Node {
    Tensor out_saved;
    SigmoidNode(const Tensor& in, const Tensor& out) : Node("Sigmoid", {in}), out_saved(out.detach()) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in(out_saved.shape(), out_saved.device());
        int64_t n = out_saved.numel();
        if (out_saved.device() == Device::CPU) {
            const float* y = out_saved.impl->data();
            const float* dout = grad_output.impl->data();
            float* din = d_in.impl->data();
            for (int64_t i = 0; i < n; ++i) {
                din[i] = dout[i] * y[i] * (1.0f - y[i]);
            }
        } else {
            cuda::launch_sigmoid_backward(d_in.impl->data(), grad_output.impl->data(), out_saved.impl->data(), (int)n);
        }
        accumulate_grad(inputs[0], d_in);
    }
};

Tensor sigmoid(const Tensor& a) {
    Tensor a_contig = a.contiguous();
    Tensor out(a.shape(), a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        const float* da = a_contig.impl->data();
        float* dout = out.impl->data();
        for (int64_t i = 0; i < n; ++i) {
            dout[i] = 1.0f / (1.0f + std::exp(-da[i]));
        }
    } else {
        cuda::launch_sigmoid(out.impl->data(), a_contig.impl->data(), (int)n);
    }
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<SigmoidNode>(a, out));
    }
    return out;
}

struct TanhNode : public Node {
    Tensor out_saved;
    TanhNode(const Tensor& in, const Tensor& out) : Node("Tanh", {in}), out_saved(out.detach()) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in(out_saved.shape(), out_saved.device());
        int64_t n = out_saved.numel();
        if (out_saved.device() == Device::CPU) {
            const float* y = out_saved.impl->data();
            const float* dout = grad_output.impl->data();
            float* din = d_in.impl->data();
            for (int64_t i = 0; i < n; ++i) {
                din[i] = dout[i] * (1.0f - y[i] * y[i]);
            }
        } else {
            cuda::launch_tanh_backward(d_in.impl->data(), grad_output.impl->data(), out_saved.impl->data(), (int)n);
        }
        accumulate_grad(inputs[0], d_in);
    }
};

Tensor tanh_op(const Tensor& a) {
    Tensor a_contig = a.contiguous();
    Tensor out(a.shape(), a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        const float* da = a_contig.impl->data();
        float* dout = out.impl->data();
        for (int64_t i = 0; i < n; ++i) dout[i] = std::tanh(da[i]);
    } else {
        cuda::launch_tanh(out.impl->data(), a_contig.impl->data(), (int)n);
    }
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<TanhNode>(a, out));
    }
    return out;
}

struct ExpNode : public Node {
    Tensor out_saved;
    ExpNode(const Tensor& in, const Tensor& out) : Node("Exp", {in}), out_saved(out.detach()) {}

    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], grad_output * out_saved);
    }
};

Tensor exp_op(const Tensor& a) {
    Tensor a_contig = a.contiguous();
    Tensor out(a.shape(), a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        const float* da = a_contig.impl->data();
        float* dout = out.impl->data();
        for (int64_t i = 0; i < n; ++i) dout[i] = std::exp(da[i]);
    } else {
        cuda::launch_exp(out.impl->data(), a_contig.impl->data(), (int)n);
    }
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<ExpNode>(a, out));
    }
    return out;
}

struct LogNode : public Node {
    Tensor in_saved;
    LogNode(const Tensor& in) : Node("Log", {in}), in_saved(in.detach()) {}

    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], grad_output / in_saved);
    }
};

Tensor log_op(const Tensor& a) {
    Tensor a_contig = a.contiguous();
    Tensor out(a.shape(), a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        const float* da = a_contig.impl->data();
        float* dout = out.impl->data();
        for (int64_t i = 0; i < n; ++i) {
            dout[i] = std::log(std::max(da[i], 1e-12f));
        }
    } else {
        cuda::launch_log(out.impl->data(), a_contig.impl->data(), (int)n);
    }
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<LogNode>(a));
    }
    return out;
}

struct SqrtNode : public Node {
    Tensor out_saved;
    SqrtNode(const Tensor& in, const Tensor& out) : Node("Sqrt", {in}), out_saved(out.detach()) {}

    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], grad_output / (out_saved * Tensor(2.0f, out_saved.device())));
    }
};

Tensor sqrt_op(const Tensor& a) {
    Tensor a_contig = a.contiguous();
    Tensor out(a.shape(), a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        const float* da = a_contig.impl->data();
        float* dout = out.impl->data();
        for (int64_t i = 0; i < n; ++i) dout[i] = std::sqrt(std::max(da[i], 0.0f));
    } else {
        cuda::launch_sqrt(out.impl->data(), a_contig.impl->data(), (int)n);
    }
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<SqrtNode>(a, out));
    }
    return out;
}

struct PowNode : public Node {
    Tensor in_saved;
    float exponent;
    PowNode(const Tensor& in, float exponent) : Node("Pow", {in}), in_saved(in.detach()), exponent(exponent) {}

    void backward(const Tensor& grad_output) override {
        Tensor pow_term = pow_op(in_saved, exponent - 1.0f);
        accumulate_grad(inputs[0], grad_output * pow_term * Tensor(exponent, in_saved.device()));
    }
};

Tensor pow_op(const Tensor& a, float exponent) {
    Tensor a_contig = a.contiguous();
    Tensor out(a.shape(), a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        const float* da = a_contig.impl->data();
        float* dout = out.impl->data();
        for (int64_t i = 0; i < n; ++i) dout[i] = std::pow(da[i], exponent);
    } else {
        cuda::launch_pow(out.impl->data(), a_contig.impl->data(), exponent, (int)n);
    }
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<PowNode>(a, exponent));
    }
    return out;
}

struct ClampNode : public Node {
    Tensor in_saved;
    float min_val, max_val;
    ClampNode(const Tensor& in, float min_val, float max_val) 
        : Node("Clamp", {in}), in_saved(in.detach()), min_val(min_val), max_val(max_val) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in(in_saved.shape(), in_saved.device());
        int64_t n = in_saved.numel();
        if (in_saved.device() == Device::CPU) {
            const float* in = in_saved.impl->data();
            const float* dout = grad_output.impl->data();
            float* din = d_in.impl->data();
            for (int64_t i = 0; i < n; ++i) {
                din[i] = (in[i] >= min_val && in[i] <= max_val) ? dout[i] : 0.0f;
            }
        } else {
            Tensor cpu_in = in_saved.to(Device::CPU);
            Tensor cpu_dout = grad_output.to(Device::CPU);
            Tensor cpu_din(in_saved.shape(), Device::CPU);
            const float* in = cpu_in.impl->data();
            const float* dout = cpu_dout.impl->data();
            float* din = cpu_din.impl->data();
            for (int64_t i = 0; i < n; ++i) {
                din[i] = (in[i] >= min_val && in[i] <= max_val) ? dout[i] : 0.0f;
            }
            d_in = cpu_din.to(in_saved.device());
        }
        accumulate_grad(inputs[0], d_in);
    }
};

Tensor clamp_op(const Tensor& a, float min_val, float max_val) {
    Tensor a_contig = a.contiguous();
    Tensor out(a.shape(), a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        const float* da = a_contig.impl->data();
        float* dout = out.impl->data();
        for (int64_t i = 0; i < n; ++i) {
            dout[i] = std::max(min_val, std::min(max_val, da[i]));
        }
    } else {
        cuda::launch_clamp(out.impl->data(), a_contig.impl->data(), min_val, max_val, (int)n);
    }
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<ClampNode>(a, min_val, max_val));
    }
    return out;
}

struct DropoutNode : public Node {
    Tensor mask;
    float scale;
    DropoutNode(const Tensor& in, const Tensor& mask, float scale) 
        : Node("Dropout", {in}), mask(mask), scale(scale) {}

    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], grad_output * mask * Tensor(scale, mask.device()));
    }
};

Tensor dropout_op(const Tensor& a, float p, bool training) {
    if (!training || p <= 0.0f) {
        return a;
    }
    if (p >= 1.0f) {
        return Tensor::zeros(a.shape(), a.device(), a.requires_grad());
    }

    float scale = 1.0f / (1.0f - p);
    Tensor rand_u = Tensor::uniform(a.shape(), 0.0f, 1.0f, Device::CPU);
    Tensor mask(a.shape(), Device::CPU);
    int64_t n = a.numel();
    float* m_data = mask.impl->data();
    const float* u_data = rand_u.impl->data();
    for (int64_t i = 0; i < n; ++i) {
        m_data[i] = (u_data[i] >= p) ? 1.0f : 0.0f;
    }

    Tensor device_mask = mask.to(a.device());
    Tensor out = a * device_mask * Tensor(scale, a.device());

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<DropoutNode>(a, device_mask, scale));
    }
    return out;
}

// -------------------------------------------------------------
// REDUCTIONS & LOSSES
// -------------------------------------------------------------
struct SumNode : public Node {
    std::vector<int64_t> in_shape;
    int dim;
    bool keepdim;
    bool sum_all;

    SumNode(const Tensor& in, int dim, bool keepdim)
        : Node("Sum", {in}), in_shape(in.shape()), dim(dim), keepdim(keepdim), sum_all(false) {}

    SumNode(const Tensor& in)
        : Node("SumAll", {in}), in_shape(in.shape()), dim(0), keepdim(false), sum_all(true) {}

    void backward(const Tensor& grad_output) override {
        Tensor cpu_grad = grad_output.to(Device::CPU);
        Tensor cpu_res(in_shape, Device::CPU);
        int64_t total = cpu_res.numel();

        if (sum_all) {
            float val = cpu_grad.item();
            float* d = cpu_res.impl->data();
            for (int64_t i = 0; i < total; ++i) d[i] = val;
        } else {
            int rank = (int)in_shape.size();
            for (int64_t i = 0; i < total; ++i) {
                auto coords = get_coords(i, in_shape);
                std::vector<int64_t> out_coords;
                for (int j = 0; j < rank; ++j) {
                    if (j == dim) {
                        if (keepdim) out_coords.push_back(0);
                    } else {
                        out_coords.push_back(coords[j]);
                    }
                }
                if (out_coords.empty()) out_coords.push_back(0);
                cpu_res.set_element(coords, cpu_grad.get_element(out_coords));
            }
        }

        accumulate_grad(inputs[0], cpu_res.to(grad_output.device()));
    }
};

Tensor sum(const Tensor& a) {
    Tensor out(std::vector<int64_t>{1}, a.device(), op_requires_grad(a.requires_grad()));
    int64_t n = a.numel();
    if (a.device() == Device::CPU) {
        const float* da = a.impl->data();
        float total = 0.0f;
        for (int64_t i = 0; i < n; ++i) total += da[i];
        out.impl->data()[0] = total;
    } else {
        cuda::launch_sum_all(out.impl->data(), a.impl->data(), (int)n);
    }
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<SumNode>(a));
    }
    return out;
}

Tensor sum(const Tensor& a, int dim, bool keepdim) {
    int rank = (int)a.shape().size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) throw std::runtime_error("Invalid dimension for sum reduction!");

    std::vector<int64_t> out_shape;
    for (int i = 0; i < rank; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(a.shape()[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor cpu_a = a.to(Device::CPU);
    Tensor cpu_out = Tensor::zeros(out_shape, Device::CPU, op_requires_grad(a.requires_grad()));
    int64_t in_total = cpu_a.numel();

    for (int64_t i = 0; i < in_total; ++i) {
        auto coords = get_coords(i, cpu_a.shape());
        std::vector<int64_t> out_coords;
        for (int j = 0; j < rank; ++j) {
            if (j == dim) {
                if (keepdim) out_coords.push_back(0);
            } else {
                out_coords.push_back(coords[j]);
            }
        }
        if (out_coords.empty()) out_coords.push_back(0);
        float cur = cpu_out.get_element(out_coords);
        cpu_out.set_element(out_coords, cur + cpu_a.get_element(coords));
    }

    Tensor out = (a.device() == Device::CUDA) ? cpu_out.to(Device::CUDA) : cpu_out;
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<SumNode>(a, dim, keepdim));
    }
    return out;
}

Tensor mean(const Tensor& a) {
    Tensor s = sum(a);
    float n = (float)a.numel();
    return s / Tensor(n, a.device());
}

Tensor mean(const Tensor& a, int dim, bool keepdim) {
    Tensor s = sum(a, dim, keepdim);
    int rank = (int)a.shape().size();
    if (dim < 0) dim += rank;
    float n = (float)a.shape()[dim];
    return s / Tensor(n, a.device());
}

Tensor max_op(const Tensor& a, int dim, bool keepdim) {
    int rank = (int)a.shape().size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) throw std::runtime_error("Invalid dimension for max!");

    std::vector<int64_t> out_shape;
    for (int i = 0; i < rank; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(a.shape()[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor cpu_a = a.to(Device::CPU);
    Tensor cpu_out = Tensor::full(out_shape, -std::numeric_limits<float>::infinity(), Device::CPU, false);
    int64_t in_total = cpu_a.numel();

    for (int64_t i = 0; i < in_total; ++i) {
        auto coords = get_coords(i, cpu_a.shape());
        std::vector<int64_t> out_coords;
        for (int j = 0; j < rank; ++j) {
            if (j == dim) {
                if (keepdim) out_coords.push_back(0);
            } else {
                out_coords.push_back(coords[j]);
            }
        }
        if (out_coords.empty()) out_coords.push_back(0);
        float cur = cpu_out.get_element(out_coords);
        float v = cpu_a.get_element(coords);
        if (v > cur) cpu_out.set_element(out_coords, v);
    }

    return (a.device() == Device::CUDA) ? cpu_out.to(Device::CUDA) : cpu_out;
}

// -------------------------------------------------------------
// LOSSES
// -------------------------------------------------------------
struct MSELossNode : public Node {
    Tensor pred_saved, target_saved;
    MSELossNode(const Tensor& pred, const Tensor& target) 
        : Node("MSELoss", {pred}), pred_saved(pred.detach()), target_saved(target.detach()) {}

    void backward(const Tensor& grad_output) override {
        float scale = grad_output.item();
        int64_t n = pred_saved.numel();
        Tensor d_pred(pred_saved.shape(), pred_saved.device());

        if (pred_saved.device() == Device::CPU) {
            const float* p = pred_saved.impl->data();
            const float* t = target_saved.impl->data();
            float* dp = d_pred.impl->data();
            float factor = scale * 2.0f / (float)n;
            for (int64_t i = 0; i < n; ++i) {
                dp[i] = factor * (p[i] - t[i]);
            }
        } else {
            cuda::launch_mse_loss_backward(d_pred.impl->data(), pred_saved.impl->data(), target_saved.impl->data(), scale, (int)n);
        }

        accumulate_grad(inputs[0], d_pred);
    }
};

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape() || pred.device() != target.device()) {
        throw std::runtime_error("Shape or device mismatch in mse_loss!");
    }
    Tensor out(std::vector<int64_t>{1}, pred.device(), op_requires_grad(pred.requires_grad()));
    int64_t n = pred.numel();

    if (pred.device() == Device::CPU) {
        const float* p = pred.impl->data();
        const float* t = target.impl->data();
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            float diff = p[i] - t[i];
            sum_sq += diff * diff;
        }
        out.impl->data()[0] = sum_sq / (float)n;
    } else {
        cuda::launch_mse_loss_forward(out.impl->data(), pred.impl->data(), target.impl->data(), (int)n);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<MSELossNode>(pred, target));
    }
    return out;
}

struct SoftmaxCrossEntropyNode : public Node {
    Tensor logits_saved, targets_saved;
    int B, C;
    SoftmaxCrossEntropyNode(const Tensor& logits, const Tensor& targets, int B, int C)
        : Node("SoftmaxCrossEntropy", {logits}), logits_saved(logits.detach()), targets_saved(targets.detach()), B(B), C(C) {}

    void backward(const Tensor& grad_output) override {
        (void)grad_output;
        Tensor d_logits(logits_saved.shape(), logits_saved.device());
        if (logits_saved.device() == Device::CPU) {
            const float* log_data = logits_saved.impl->data();
            const float* tgt_data = targets_saved.impl->data();
            float* d_data = d_logits.impl->data();

            for (int b = 0; b < B; ++b) {
                float max_val = log_data[b * C];
                for (int c = 1; c < C; ++c) {
                    if (log_data[b * C + c] > max_val) max_val = log_data[b * C + c];
                }
                float sum_exp = 0.0f;
                for (int c = 0; c < C; ++c) sum_exp += std::exp(log_data[b * C + c] - max_val);
                for (int c = 0; c < C; ++c) {
                    float prob = std::exp(log_data[b * C + c] - max_val) / sum_exp;
                    d_data[b * C + c] = (prob - tgt_data[b * C + c]) / (float)B;
                }
            }
        } else {
            cuda::launch_softmax_cross_entropy_backward(d_logits.impl->data(), logits_saved.impl->data(), targets_saved.impl->data(), B, C);
        }
        accumulate_grad(inputs[0], d_logits);
    }
};

Tensor softmax_cross_entropy(const Tensor& logits, const Tensor& targets) {
    if (logits.shape().size() != 2 || targets.shape().size() != 2 || logits.shape() != targets.shape()) {
        throw std::runtime_error("softmax_cross_entropy requires 2D tensors with matching shapes!");
    }
    int B = (int)logits.shape()[0];
    int C = (int)logits.shape()[1];

    Tensor loss(std::vector<int64_t>{1}, logits.device(), op_requires_grad(logits.requires_grad()));

    if (logits.device() == Device::CPU) {
        const float* log_data = logits.impl->data();
        const float* tgt_data = targets.impl->data();
        float total_loss = 0.0f;

        for (int b = 0; b < B; ++b) {
            float max_val = log_data[b * C];
            for (int c = 1; c < C; ++c) {
                if (log_data[b * C + c] > max_val) max_val = log_data[b * C + c];
            }
            float sum_exp = 0.0f;
            for (int c = 0; c < C; ++c) sum_exp += std::exp(log_data[b * C + c] - max_val);
            float log_sum_exp = max_val + std::log(sum_exp);

            for (int c = 0; c < C; ++c) {
                float target_val = tgt_data[b * C + c];
                if (target_val > 0.0f) {
                    total_loss -= target_val * (log_data[b * C + c] - log_sum_exp);
                }
            }
        }
        loss.impl->data()[0] = total_loss / (float)B;
    } else {
        cuda::launch_softmax_cross_entropy_forward(loss.impl->data(), logits.impl->data(), targets.impl->data(), B, C);
    }

    if (AutoGradMode::is_enabled() && loss.requires_grad()) {
        loss.set_creator(std::make_shared<SoftmaxCrossEntropyNode>(logits, targets, B, C));
    }
    return loss;
}

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    return softmax_cross_entropy(logits, targets);
}

struct BCEWithLogitsNode : public Node {
    Tensor logits_saved, targets_saved;
    int64_t N;
    BCEWithLogitsNode(const Tensor& logits, const Tensor& targets)
        : Node("BCEWithLogits", {logits}), logits_saved(logits.detach()), targets_saved(targets.detach()), N(logits.numel()) {}

    void backward(const Tensor& grad_output) override {
        (void)grad_output;
        Tensor d_logits(logits_saved.shape(), logits_saved.device());
        if (logits_saved.device() == Device::CPU) {
            const float* z = logits_saved.impl->data();
            const float* y = targets_saved.impl->data();
            float* dz = d_logits.impl->data();
            for (int64_t i = 0; i < N; ++i) {
                float sig = 1.0f / (1.0f + std::exp(-z[i]));
                dz[i] = (sig - y[i]) / (float)N;
            }
        } else {
            cuda::launch_bce_with_logits_backward(d_logits.impl->data(), logits_saved.impl->data(), targets_saved.impl->data(), (int)N);
        }
        accumulate_grad(inputs[0], d_logits);
    }
};

Tensor bce_with_logits_loss(const Tensor& logits, const Tensor& targets) {
    if (logits.shape() != targets.shape()) {
        throw std::runtime_error("bce_with_logits_loss requires matching shapes!");
    }
    int64_t N = logits.numel();
    Tensor loss(std::vector<int64_t>{1}, logits.device(), op_requires_grad(logits.requires_grad()));

    if (logits.device() == Device::CPU) {
        const float* z = logits.impl->data();
        const float* y = targets.impl->data();
        float total = 0.0f;
        for (int64_t i = 0; i < N; ++i) {
            float zi = z[i];
            float yi = y[i];
            float max_val = zi > 0.0f ? zi : 0.0f;
            float abs_z = std::abs(zi);
            total += max_val - zi * yi + std::log1p(std::exp(-abs_z));
        }
        loss.impl->data()[0] = total / (float)N;
    } else {
        cuda::launch_bce_with_logits_forward(loss.impl->data(), logits.impl->data(), targets.impl->data(), (int)N);
    }

    if (AutoGradMode::is_enabled() && loss.requires_grad()) {
        loss.set_creator(std::make_shared<BCEWithLogitsNode>(logits, targets));
    }
    return loss;
}

struct L1LossNode : public Node {
    Tensor pred_saved, target_saved;
    int64_t N;
    L1LossNode(const Tensor& pred, const Tensor& target)
        : Node("L1Loss", {pred}), pred_saved(pred.detach()), target_saved(target.detach()), N(pred.numel()) {}

    void backward(const Tensor& grad_output) override {
        (void)grad_output;
        Tensor d_pred(pred_saved.shape(), pred_saved.device());
        Tensor cpu_p = pred_saved.to(Device::CPU);
        Tensor cpu_t = target_saved.to(Device::CPU);
        Tensor cpu_dp(pred_saved.shape(), Device::CPU);

        const float* p = cpu_p.impl->data();
        const float* t = cpu_t.impl->data();
        float* dp = cpu_dp.impl->data();

        for (int64_t i = 0; i < N; ++i) {
            float diff = p[i] - t[i];
            dp[i] = (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f)) / (float)N;
        }
        accumulate_grad(inputs[0], cpu_dp.to(pred_saved.device()));
    }
};

// -------------------------------------------------------------
// SMOOTH L1 LOSS
// -------------------------------------------------------------
struct SmoothL1LossNode : public Node {
    Tensor pred_saved, target_saved;
    float beta;
    SmoothL1LossNode(const Tensor& pred, const Tensor& target, float beta) 
        : Node("SmoothL1Loss", {pred}), pred_saved(pred.detach()), target_saved(target.detach()), beta(beta) {}

    void backward(const Tensor& grad_output) override {
        float scale = grad_output.item();
        int64_t n = pred_saved.numel();
        Tensor d_pred(pred_saved.shape(), pred_saved.device());

        if (pred_saved.device() == Device::CPU) {
            const float* p = pred_saved.impl->data();
            const float* t = target_saved.impl->data();
            float* dp = d_pred.impl->data();
            float factor = scale / (float)n;
            for (int64_t i = 0; i < n; ++i) {
                float diff = p[i] - t[i];
                if (std::abs(diff) < beta) {
                    dp[i] = factor * diff / beta;
                } else {
                    dp[i] = factor * (diff > 0 ? 1.0f : -1.0f);
                }
            }
        } else {
            // Fallback for CUDA
            // For now, if CUDA is requested, we will fall back to CPU or we could just implement the kernel.
            // Assuming we added the kernel to cuda_kernels
            cuda::launch_smooth_l1_loss_backward(d_pred.impl->data(), pred_saved.impl->data(), target_saved.impl->data(), beta, scale, (int)n);
        }

        accumulate_grad(inputs[0], d_pred);
    }
};

Tensor smooth_l1_loss(const Tensor& pred, const Tensor& target, float beta) {
    if (pred.shape() != target.shape() || pred.device() != target.device()) {
        throw std::runtime_error("Shape or device mismatch in smooth_l1_loss!");
    }
    Tensor out(std::vector<int64_t>{1}, pred.device(), op_requires_grad(pred.requires_grad()));
    int64_t n = pred.numel();

    if (pred.device() == Device::CPU) {
        const float* p = pred.impl->data();
        const float* t = target.impl->data();
        float sum_loss = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            float diff = std::abs(p[i] - t[i]);
            if (diff < beta) {
                sum_loss += 0.5f * diff * diff / beta;
            } else {
                sum_loss += diff - 0.5f * beta;
            }
        }
        out.impl->data()[0] = sum_loss / (float)n;
    } else {
        cuda::launch_smooth_l1_loss_forward(out.impl->data(), pred.impl->data(), target.impl->data(), beta, (int)n);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<SmoothL1LossNode>(pred, target, beta));
    }
    return out;
}

// -------------------------------------------------------------
// KLDIV LOSS
// -------------------------------------------------------------
struct KLDivLossNode : public Node {
    Tensor pred_saved, target_saved;
    KLDivLossNode(const Tensor& pred, const Tensor& target) 
        : Node("KLDivLoss", {pred}), pred_saved(pred.detach()), target_saved(target.detach()) {}

    void backward(const Tensor& grad_output) override {
        float scale = grad_output.item();
        int64_t n = pred_saved.numel();
        Tensor d_pred(pred_saved.shape(), pred_saved.device());

        if (pred_saved.device() == Device::CPU) {
            const float* t = target_saved.impl->data();
            float* dp = d_pred.impl->data();
            float factor = scale / (float)n;
            for (int64_t i = 0; i < n; ++i) {
                dp[i] = -factor * t[i];
            }
        } else {
            cuda::launch_kl_div_loss_backward(d_pred.impl->data(), target_saved.impl->data(), scale, (int)n);
        }

        accumulate_grad(inputs[0], d_pred);
    }
};

Tensor kl_div_loss(const Tensor& input, const Tensor& target) {
    if (input.shape() != target.shape() || input.device() != target.device()) {
        throw std::runtime_error("Shape or device mismatch in kl_div_loss!");
    }
    Tensor out(std::vector<int64_t>{1}, input.device(), op_requires_grad(input.requires_grad()));
    int64_t n = input.numel();

    if (input.device() == Device::CPU) {
        const float* p = input.impl->data();
        const float* t = target.impl->data();
        float sum_loss = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            if (t[i] > 0.0f) {
                sum_loss += t[i] * (std::log(t[i]) - p[i]);
            }
        }
        out.impl->data()[0] = sum_loss / (float)n;
    } else {
        cuda::launch_kl_div_loss_forward(out.impl->data(), input.impl->data(), target.impl->data(), (int)n);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<KLDivLossNode>(input, target));
    }
    return out;
}

Tensor l1_loss(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        throw std::runtime_error("l1_loss requires matching shapes!");
    }
    int64_t N = pred.numel();
    Tensor loss(std::vector<int64_t>{1}, pred.device(), op_requires_grad(pred.requires_grad()));

    Tensor cpu_p = pred.to(Device::CPU);
    Tensor cpu_t = target.to(Device::CPU);
    const float* p = cpu_p.impl->data();
    const float* t = cpu_t.impl->data();
    float total = 0.0f;
    for (int64_t i = 0; i < N; ++i) {
        total += std::abs(p[i] - t[i]);
    }
    if (pred.device() == Device::CPU) {
        loss.impl->data()[0] = total / (float)N;
    } else {
        float val = total / (float)N;
        CUDA_CHECK(cudaMemcpy(loss.impl->data(), &val, sizeof(float), cudaMemcpyHostToDevice));
    }

    if (AutoGradMode::is_enabled() && loss.requires_grad()) {
        loss.set_creator(std::make_shared<L1LossNode>(pred, target));
    }
    return loss;
}

// -------------------------------------------------------------
// NORMALIZATION & CONV
// -------------------------------------------------------------
struct LayerNormNode : public Node {
    Tensor x_saved, gamma_saved, beta_saved;
    int B, D;
    float eps;

    LayerNormNode(const Tensor& x, const Tensor& gamma, const Tensor& beta, int B, int D, float eps)
        : Node("LayerNorm", {x, gamma, beta}), x_saved(x.detach()), gamma_saved(gamma.detach()), beta_saved(beta.detach()), B(B), D(D), eps(eps) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_x(x_saved.shape(), x_saved.device());
        Tensor d_gamma = Tensor::zeros(gamma_saved.shape(), gamma_saved.device());
        Tensor d_beta = Tensor::zeros(beta_saved.shape(), beta_saved.device());

        Tensor cpu_x = x_saved.to(Device::CPU);
        Tensor cpu_gamma = gamma_saved.to(Device::CPU);
        Tensor cpu_dout = grad_output.to(Device::CPU);

        Tensor cpu_dx(x_saved.shape(), Device::CPU);
        Tensor cpu_dgamma = Tensor::zeros(gamma_saved.shape(), Device::CPU);
        Tensor cpu_dbeta = Tensor::zeros(beta_saved.shape(), Device::CPU);

        const float* x_ptr = cpu_x.impl->data();
        const float* g_ptr = cpu_gamma.impl->data();
        const float* dout_ptr = cpu_dout.impl->data();
        float* dx_ptr = cpu_dx.impl->data();
        float* dg_ptr = cpu_dgamma.impl->data();
        float* db_ptr = cpu_dbeta.impl->data();

        for (int b = 0; b < B; ++b) {
            float mean = 0.0f;
            for (int d = 0; d < D; ++d) mean += x_ptr[b * D + d];
            mean /= (float)D;

            float var = 0.0f;
            for (int d = 0; d < D; ++d) {
                float diff = x_ptr[b * D + d] - mean;
                var += diff * diff;
            }
            var /= (float)D;
            float inv_std = 1.0f / std::sqrt(var + eps);

            float dot_p = 0.0f;
            float sum_dout = 0.0f;
            for (int d = 0; d < D; ++d) {
                float x_hat = (x_ptr[b * D + d] - mean) * inv_std;
                float g = g_ptr[d];
                float dout = dout_ptr[b * D + d];
                dot_p += dout * g * x_hat;
                sum_dout += dout * g;
                dg_ptr[d] += dout * x_hat;
                db_ptr[d] += dout;
            }

            for (int d = 0; d < D; ++d) {
                float x_hat = (x_ptr[b * D + d] - mean) * inv_std;
                float g = g_ptr[d];
                float dout = dout_ptr[b * D + d];
                dx_ptr[b * D + d] = (inv_std / (float)D) * ((float)D * dout * g - sum_dout - x_hat * dot_p);
            }
        }

        accumulate_grad(inputs[0], cpu_dx.to(x_saved.device()));
        accumulate_grad(inputs[1], cpu_dgamma.to(gamma_saved.device()));
        accumulate_grad(inputs[2], cpu_dbeta.to(beta_saved.device()));
    }
};

Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
    if (x.shape().size() < 2) throw std::runtime_error("LayerNorm requires at least 2D tensor!");
    int D = (int)x.shape().back();
    int B = (int)(x.numel() / D);

    Tensor out(x.shape(), x.device(), op_requires_grad(x.requires_grad() || gamma.requires_grad() || beta.requires_grad()));

    if (x.device() == Device::CPU) {
        const float* in = x.impl->data();
        const float* g = gamma.impl->data();
        const float* b = beta.impl->data();
        float* dout = out.impl->data();

        for (int i = 0; i < B; ++i) {
            float mean = 0.0f;
            for (int d = 0; d < D; ++d) mean += in[i * D + d];
            mean /= (float)D;

            float var = 0.0f;
            for (int d = 0; d < D; ++d) {
                float diff = in[i * D + d] - mean;
                var += diff * diff;
            }
            var /= (float)D;
            float inv_std = 1.0f / std::sqrt(var + eps);

            for (int d = 0; d < D; ++d) {
                float x_hat = (in[i * D + d] - mean) * inv_std;
                dout[i * D + d] = g[d] * x_hat + b[d];
            }
        }
    } else {
        cuda::launch_layernorm_forward(out.impl->data(), x.impl->data(), gamma.impl->data(), beta.impl->data(), B, D, eps);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<LayerNormNode>(x, gamma, beta, B, D, eps));
    }
    return out;
}

// RMSNorm
struct RMSNormNode : public Node {
    Tensor x_saved, gamma_saved;
    int B, D;
    float eps;

    RMSNormNode(const Tensor& x, const Tensor& gamma, int B, int D, float eps)
        : Node("RMSNorm", {x, gamma}), x_saved(x.detach()), gamma_saved(gamma.detach()), B(B), D(D), eps(eps) {}

    void backward(const Tensor& grad_output) override {
        Tensor cpu_x = x_saved.to(Device::CPU);
        Tensor cpu_gamma = gamma_saved.to(Device::CPU);
        Tensor cpu_dout = grad_output.to(Device::CPU);

        Tensor cpu_dx(x_saved.shape(), Device::CPU);
        Tensor cpu_dgamma = Tensor::zeros(gamma_saved.shape(), Device::CPU);

        const float* x_ptr = cpu_x.impl->data();
        const float* g_ptr = cpu_gamma.impl->data();
        const float* dout_ptr = cpu_dout.impl->data();
        float* dx_ptr = cpu_dx.impl->data();
        float* dg_ptr = cpu_dgamma.impl->data();

        for (int b = 0; b < B; ++b) {
            float sum_sq = 0.0f;
            for (int d = 0; d < D; ++d) {
                float v = x_ptr[b * D + d];
                sum_sq += v * v;
            }
            float inv_rms = 1.0f / std::sqrt(sum_sq / (float)D + eps);

            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                float dy = dout_ptr[b * D + d];
                float g = g_ptr[d];
                float x = x_ptr[b * D + d];
                dot += dy * g * x;
                dg_ptr[d] += dy * x * inv_rms;
            }

            for (int d = 0; d < D; ++d) {
                float dy = dout_ptr[b * D + d];
                float g = g_ptr[d];
                float x = x_ptr[b * D + d];
                dx_ptr[b * D + d] = inv_rms * (dy * g - (x * inv_rms * inv_rms / (float)D) * dot);
            }
        }

        accumulate_grad(inputs[0], cpu_dx.to(x_saved.device()));
        accumulate_grad(inputs[1], cpu_dgamma.to(gamma_saved.device()));
    }
};

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps) {
    if (x.shape().size() < 2) throw std::runtime_error("RMSNorm requires at least 2D tensor!");
    int D = (int)x.shape().back();
    int B = (int)(x.numel() / D);

    Tensor out(x.shape(), x.device(), op_requires_grad(x.requires_grad() || gamma.requires_grad()));

    if (x.device() == Device::CPU) {
        const float* in = x.impl->data();
        const float* g = gamma.impl->data();
        float* dout = out.impl->data();

        for (int i = 0; i < B; ++i) {
            float sum_sq = 0.0f;
            for (int d = 0; d < D; ++d) {
                float v = in[i * D + d];
                sum_sq += v * v;
            }
            float inv_rms = 1.0f / std::sqrt(sum_sq / (float)D + eps);

            for (int d = 0; d < D; ++d) {
                dout[i * D + d] = in[i * D + d] * inv_rms * g[d];
            }
        }
    } else {
        cuda::launch_rmsnorm_forward(out.impl->data(), x.impl->data(), gamma.impl->data(), B, D, eps);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<RMSNormNode>(x, gamma, B, D, eps));
    }
    return out;
}

// Masked Fill
struct MaskedFillNode : public Node {
    Tensor mask_saved;
    MaskedFillNode(const Tensor& x, const Tensor& mask)
        : Node("MaskedFill", {x}), mask_saved(mask.detach()) {}

    void backward(const Tensor& grad_output) override {
        int64_t n = grad_output.numel();
        Tensor cpu_dout = grad_output.to(Device::CPU);
        Tensor cpu_mask = mask_saved.to(Device::CPU);
        Tensor cpu_dx(grad_output.shape(), Device::CPU);

        const float* dout = cpu_dout.impl->data();
        const float* mask = cpu_mask.impl->data();
        float* dx = cpu_dx.impl->data();

        for (int64_t i = 0; i < n; ++i) {
            dx[i] = (mask[i] != 0.0f) ? 0.0f : dout[i];
        }
        accumulate_grad(inputs[0], cpu_dx.to(grad_output.device()));
    }
};

Tensor masked_fill(const Tensor& x, const Tensor& mask, float value) {
    if (x.shape() != mask.shape()) {
        throw std::runtime_error("masked_fill requires matching shapes for x and mask!");
    }
    Tensor out(x.shape(), x.device(), op_requires_grad(x.requires_grad()));
    int64_t n = x.numel();

    if (x.device() == Device::CPU) {
        const float* in = x.impl->data();
        const float* m = mask.impl->data();
        float* dout = out.impl->data();
        for (int64_t i = 0; i < n; ++i) {
            dout[i] = (m[i] != 0.0f) ? value : in[i];
        }
    } else {
        cuda::launch_masked_fill(out.impl->data(), x.impl->data(), mask.impl->data(), value, (int)n);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<MaskedFillNode>(x, mask));
    }
    return out;
}

// Cat / Concat
struct CatNode : public Node {
    int dim;
    std::vector<int64_t> split_sizes;
    CatNode(const std::vector<Tensor>& inputs, int dim, const std::vector<int64_t>& split_sizes)
        : Node("Cat", inputs), dim(dim), split_sizes(split_sizes) {}

    void backward(const Tensor& grad_output) override {
        Tensor cpu_g = grad_output.to(Device::CPU);
        auto shape = grad_output.shape();
        int64_t outer_size = 1;
        for (int i = 0; i < dim; ++i) outer_size *= shape[i];
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (size_t i = dim + 1; i < shape.size(); ++i) inner_size *= shape[i];

        int64_t offset = 0;
        for (size_t k = 0; k < inputs.size(); ++k) {
            int64_t cur_len = split_sizes[k];
            Tensor sub_grad(inputs[k]->shape, Device::CPU);
            float* sub_ptr = sub_grad.impl->data();
            const float* g_ptr = cpu_g.impl->data();

            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t c = 0; c < cur_len; ++c) {
                    for (int64_t in = 0; in < inner_size; ++in) {
                        int64_t g_idx = (o * dim_size + offset + c) * inner_size + in;
                        int64_t s_idx = (o * cur_len + c) * inner_size + in;
                        sub_ptr[s_idx] = g_ptr[g_idx];
                    }
                }
            }
            offset += cur_len;
            accumulate_grad(inputs[k], sub_grad.to(grad_output.device()));
        }
    }
};

Tensor cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) throw std::runtime_error("cat requires non-empty tensor list!");
    int rank = (int)tensors[0].shape().size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) throw std::runtime_error("cat dim out of bounds!");

    std::vector<int64_t> out_shape = tensors[0].shape();
    int64_t total_dim = 0;
    std::vector<int64_t> split_sizes;
    bool any_requires_grad = false;

    for (const auto& t : tensors) {
        if (t.shape().size() != (size_t)rank) throw std::runtime_error("cat requires all tensors to have matching rank!");
        for (int i = 0; i < rank; ++i) {
            if (i != dim && t.shape()[i] != out_shape[i]) {
                throw std::runtime_error("cat shape mismatch on non-cat dimension!");
            }
        }
        total_dim += t.shape()[dim];
        split_sizes.push_back(t.shape()[dim]);
        if (t.requires_grad()) any_requires_grad = true;
    }
    out_shape[dim] = total_dim;

    Tensor out(out_shape, tensors[0].device(), op_requires_grad(any_requires_grad));
    Tensor cpu_out(out_shape, Device::CPU, op_requires_grad(any_requires_grad));
    float* out_ptr = cpu_out.impl->data();

    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= out_shape[i];
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < out_shape.size(); ++i) inner_size *= out_shape[i];

    int64_t offset = 0;
    for (const auto& t : tensors) {
        Tensor cpu_t = t.to(Device::CPU);
        const float* t_ptr = cpu_t.impl->data();
        int64_t cur_len = t.shape()[dim];

        for (int64_t o = 0; o < outer_size; ++o) {
            for (int64_t c = 0; c < cur_len; ++c) {
                for (int64_t in = 0; in < inner_size; ++in) {
                    int64_t o_idx = (o * total_dim + offset + c) * inner_size + in;
                    int64_t t_idx = (o * cur_len + c) * inner_size + in;
                    out_ptr[o_idx] = t_ptr[t_idx];
                }
            }
        }
        offset += cur_len;
    }

    if (out.device() == Device::CUDA) {
        out = cpu_out.to(Device::CUDA);
    } else {
        out = cpu_out;
    }

    if (AutoGradMode::is_enabled() && any_requires_grad) {
        out.set_creator(std::make_shared<CatNode>(tensors, dim, split_sizes));
    }
    return out;
}

Tensor rotary_embedding(const Tensor& x, const Tensor& cos, const Tensor& sin) {
    // x: [..., D] where D is head dim
    int64_t D = x.shape().back();
    if (D % 2 != 0) throw std::runtime_error("Rotary embedding requires even head dimension!");
    int64_t half = D / 2;
    int64_t total_vecs = x.numel() / D;

    Tensor cpu_x = x.to(Device::CPU);
    Tensor cpu_cos = cos.to(Device::CPU);
    Tensor cpu_sin = sin.to(Device::CPU);
    Tensor cpu_out(x.shape(), Device::CPU);

    const float* xp = cpu_x.impl->data();
    const float* cp = cpu_cos.impl->data();
    const float* sp = cpu_sin.impl->data();
    float* op = cpu_out.impl->data();

    int64_t cos_numel = cos.numel();

    for (int64_t v = 0; v < total_vecs; ++v) {
        for (int64_t d = 0; d < half; ++d) {
            float x1 = xp[v * D + d];
            float x2 = xp[v * D + half + d];
            float c = cp[(v * half + d) % cos_numel];
            float s = sp[(v * half + d) % cos_numel];
            op[v * D + d] = x1 * c - x2 * s;
            op[v * D + half + d] = x2 * c + x1 * s;
        }
    }

    return (x.device() == Device::CUDA) ? cpu_out.to(Device::CUDA) : cpu_out;
}

// -------------------------------------------------------------
// CONV2D & BATCHNORM2D
// -------------------------------------------------------------
struct Conv2DNode : public Node {
    Tensor x_saved, weight_saved, bias_saved;
    int B, C, H, W, O, Kh, Kw, H_out, W_out, stride, padding;

    Conv2DNode(const Tensor& x, const Tensor& weight, const Tensor& bias,
               int B, int C, int H, int W, int O, int Kh, int Kw,
               int H_out, int W_out, int stride, int padding)
        : Node("Conv2D", {x, weight, bias}),
          x_saved(x.detach()), weight_saved(weight.detach()), bias_saved(bias.detach()),
          B(B), C(C), H(H), W(W), O(O), Kh(Kh), Kw(Kw),
          H_out(H_out), W_out(W_out), stride(stride), padding(padding) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in = Tensor::zeros(x_saved.shape(), x_saved.device());
        Tensor d_weight = Tensor::zeros(weight_saved.shape(), weight_saved.device());
        Tensor d_bias = Tensor::zeros(bias_saved.shape(), bias_saved.device());

        if (x_saved.device() == Device::CPU) {
            const float* d_out = grad_output.impl->data();
            const float* in = x_saved.impl->data();
            const float* weight = weight_saved.impl->data();
            float* din_ptr = d_in.impl->data();
            float* dw_ptr = d_weight.impl->data();
            float* db_ptr = d_bias.impl->data();

            // Compute d_weight and d_bias
            for (int b = 0; b < B; ++b) {
                for (int o = 0; o < O; ++o) {
                    for (int h_out = 0; h_out < H_out; ++h_out) {
                        for (int w_out = 0; w_out < W_out; ++w_out) {
                            float dout_val = d_out[((b * O + o) * H_out + h_out) * W_out + w_out];
                            db_ptr[o] += dout_val;

                            for (int c = 0; c < C; ++c) {
                                for (int ky = 0; ky < Kh; ++ky) {
                                    for (int kx = 0; kx < Kw; ++kx) {
                                        int in_y = h_out * stride - padding + ky;
                                        int in_x = w_out * stride - padding + kx;
                                        if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                            int in_idx = ((b * C + c) * H + in_y) * W + in_x;
                                            int w_idx = ((o * C + c) * Kh + ky) * Kw + kx;
                                            dw_ptr[w_idx] += dout_val * in[in_idx];
                                            din_ptr[in_idx] += dout_val * weight[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            cuda::launch_conv2d_backward_input(d_in.impl->data(), grad_output.impl->data(), weight_saved.impl->data(),
                                               B, C, H, W, O, Kh, Kw, H_out, W_out, stride, padding);
            cuda::launch_conv2d_backward_weight(d_weight.impl->data(), grad_output.impl->data(), x_saved.impl->data(),
                                                B, C, H, W, O, Kh, Kw, H_out, W_out, stride, padding);
            // Sum bias on CPU
            Tensor cpu_dout = grad_output.to(Device::CPU);
            const float* dout = cpu_dout.impl->data();
            Tensor cpu_db = Tensor::zeros(bias_saved.shape(), Device::CPU);
            float* db = cpu_db.impl->data();
            for (int b = 0; b < B; ++b) {
                for (int o = 0; o < O; ++o) {
                    for (int hw = 0; hw < H_out * W_out; ++hw) {
                        db[o] += dout[(b * O + o) * H_out * W_out + hw];
                    }
                }
            }
            d_bias = cpu_db.to(Device::CUDA);
        }

        accumulate_grad(inputs[0], d_in);
        accumulate_grad(inputs[1], d_weight);
        accumulate_grad(inputs[2], d_bias);
    }
};

Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias, int stride, int padding) {
    if (x.shape().size() != 4 || weight.shape().size() != 4) {
        throw std::runtime_error("conv2d requires 4D tensors: input (B,C,H,W) and weight (O,C,Kh,Kw)!");
    }
    int B = (int)x.shape()[0];
    int C = (int)x.shape()[1];
    int H = (int)x.shape()[2];
    int W = (int)x.shape()[3];

    int O = (int)weight.shape()[0];
    int C_in = (int)weight.shape()[1];
    int Kh = (int)weight.shape()[2];
    int Kw = (int)weight.shape()[3];

    if (C != C_in) throw std::runtime_error("Input channel mismatch in conv2d!");

    int H_out = (H + 2 * padding - Kh) / stride + 1;
    int W_out = (W + 2 * padding - Kw) / stride + 1;

    Tensor out({B, O, H_out, W_out}, x.device(), op_requires_grad(x.requires_grad() || weight.requires_grad() || bias.requires_grad()));

    if (x.device() == Device::CPU) {
        const float* in = x.impl->data();
        const float* w = weight.impl->data();
        const float* b = bias.impl->data();
        float* dout = out.impl->data();

        for (int b_idx = 0; b_idx < B; ++b_idx) {
            for (int o = 0; o < O; ++o) {
                for (int h_out = 0; h_out < H_out; ++h_out) {
                    for (int w_out = 0; w_out < W_out; ++w_out) {
                        float val = b ? b[o] : 0.0f;
                        for (int c = 0; c < C; ++c) {
                            for (int ky = 0; ky < Kh; ++ky) {
                                for (int kx = 0; kx < Kw; ++kx) {
                                    int in_y = h_out * stride - padding + ky;
                                    int in_x = w_out * stride - padding + kx;
                                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                        int in_idx = ((b_idx * C + c) * H + in_y) * W + in_x;
                                        int w_idx = ((o * C + c) * Kh + ky) * Kw + kx;
                                        val += in[in_idx] * w[w_idx];
                                    }
                                }
                            }
                        }
                        int out_idx = ((b_idx * O + o) * H_out + h_out) * W_out + w_out;
                        dout[out_idx] = val;
                    }
                }
            }
        }
    } else {
        cuda::launch_conv2d_forward(out.impl->data(), x.impl->data(), weight.impl->data(), bias.impl->data(),
                                    B, C, H, W, O, Kh, Kw, H_out, W_out, stride, padding);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<Conv2DNode>(x, weight, bias, B, C, H, W, O, Kh, Kw, H_out, W_out, stride, padding));
    }
    return out;
}

struct BatchNorm2DNode : public Node {
    Tensor x_saved, gamma_saved, mean_saved, var_saved;
    int B, C, H, W;
    float eps;

    BatchNorm2DNode(const Tensor& x, const Tensor& gamma, const Tensor& beta, const Tensor& mean, const Tensor& var,
                    int B, int C, int H, int W, float eps)
        : Node("BatchNorm2D", {x, gamma, beta}),
          x_saved(x.detach()), gamma_saved(gamma.detach()), mean_saved(mean.detach()), var_saved(var.detach()),
          B(B), C(C), H(H), W(W), eps(eps) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_x = Tensor::zeros(x_saved.shape(), Device::CPU);
        Tensor d_gamma = Tensor::zeros(gamma_saved.shape(), Device::CPU);
        Tensor d_beta = Tensor::zeros(gamma_saved.shape(), Device::CPU);

        Tensor cpu_dout = grad_output.to(Device::CPU);
        Tensor cpu_x = x_saved.to(Device::CPU);
        Tensor cpu_gamma = gamma_saved.to(Device::CPU);
        Tensor cpu_mean = mean_saved.to(Device::CPU);
        Tensor cpu_var = var_saved.to(Device::CPU);

        const float* dout = cpu_dout.impl->data();
        const float* x = cpu_x.impl->data();
        const float* gamma = cpu_gamma.impl->data();
        const float* mean = cpu_mean.impl->data();
        const float* var = cpu_var.impl->data();

        float* dx = d_x.impl->data();
        float* dg = d_gamma.impl->data();
        float* db = d_beta.impl->data();

        int N = B * H * W;

        for (int c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(var[c] + eps);
            float d_gamma_c = 0.0f;
            float d_beta_c = 0.0f;

            for (int b = 0; b < B; ++b) {
                for (int hw = 0; hw < H * W; ++hw) {
                    int idx = (b * C + c) * H * W + hw;
                    float x_hat = (x[idx] - mean[c]) * inv_std;
                    d_gamma_c += dout[idx] * x_hat;
                    d_beta_c += dout[idx];
                }
            }
            dg[c] = d_gamma_c;
            db[c] = d_beta_c;

            // dL/dx = (gamma / (N * sqrt(var + eps))) * (N * dout - sum(dout) - x_hat * sum(dout * x_hat))
            for (int b = 0; b < B; ++b) {
                for (int hw = 0; hw < H * W; ++hw) {
                    int idx = (b * C + c) * H * W + hw;
                    float x_hat = (x[idx] - mean[c]) * inv_std;
                    dx[idx] = (gamma[c] * inv_std / (float)N) * ((float)N * dout[idx] - d_beta_c - x_hat * d_gamma_c);
                }
            }
        }

        accumulate_grad(inputs[0], d_x.to(x_saved.device()));
        accumulate_grad(inputs[1], d_gamma.to(gamma_saved.device()));
        accumulate_grad(inputs[2], d_beta.to(gamma_saved.device()));
    }
};

Tensor batch_norm3d(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                    Tensor& running_mean, Tensor& running_var,
                    float momentum, float eps, bool training) {
    if (x.shape().size() != 5) throw std::runtime_error("batch_norm3d requires 5D tensor (B,C,D,H,W)!");
    int64_t B = x.shape()[0];
    int64_t C = x.shape()[1];
    int64_t D = x.shape()[2];
    int64_t H = x.shape()[3];
    int64_t W = x.shape()[4];
    Tensor x_2d = x.reshape({B, C, D * H, W});
    Tensor out_2d = batch_norm2d(x_2d, gamma, beta, running_mean, running_var, momentum, eps, training);
    return out_2d.reshape({B, C, D, H, W});
}

Tensor batch_norm2d(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                    Tensor& running_mean, Tensor& running_var,
                    float momentum, float eps, bool training) {
    if (x.shape().size() != 4) throw std::runtime_error("batch_norm2d requires 4D tensor (B,C,H,W)!");
    int B = (int)x.shape()[0];
    int C = (int)x.shape()[1];
    int H = (int)x.shape()[2];
    int W = (int)x.shape()[3];

    Tensor mean(std::vector<int64_t>{(int64_t)C}, x.device());
    Tensor var(std::vector<int64_t>{(int64_t)C}, x.device());

    if (training) {
        Tensor cpu_x = x.to(Device::CPU);
        Tensor cpu_mean(std::vector<int64_t>{(int64_t)C}, Device::CPU);
        Tensor cpu_var(std::vector<int64_t>{(int64_t)C}, Device::CPU);

        const float* x_ptr = cpu_x.impl->data();
        float* m_ptr = cpu_mean.impl->data();
        float* v_ptr = cpu_var.impl->data();
        int N = B * H * W;

        for (int c = 0; c < C; ++c) {
            float sum_val = 0.0f;
            for (int b = 0; b < B; ++b) {
                for (int hw = 0; hw < H * W; ++hw) {
                    sum_val += x_ptr[(b * C + c) * H * W + hw];
                }
            }
            float cur_mean = sum_val / (float)N;
            m_ptr[c] = cur_mean;

            float sum_sq = 0.0f;
            for (int b = 0; b < B; ++b) {
                for (int hw = 0; hw < H * W; ++hw) {
                    float diff = x_ptr[(b * C + c) * H * W + hw] - cur_mean;
                    sum_sq += diff * diff;
                }
            }
            v_ptr[c] = sum_sq / (float)N;
        }

        // Update running stats
        Tensor cpu_rm = running_mean.to(Device::CPU);
        Tensor cpu_rv = running_var.to(Device::CPU);
        for (int c = 0; c < C; ++c) {
            cpu_rm.impl->data()[c] = (1.0f - momentum) * cpu_rm.impl->data()[c] + momentum * m_ptr[c];
            cpu_rv.impl->data()[c] = (1.0f - momentum) * cpu_rv.impl->data()[c] + momentum * v_ptr[c];
        }
        running_mean = cpu_rm.to(running_mean.device());
        running_var = cpu_rv.to(running_var.device());

        mean = cpu_mean.to(x.device());
        var = cpu_var.to(x.device());
    } else {
        mean = running_mean;
        var = running_var;
    }

    Tensor out(x.shape(), x.device(), op_requires_grad(x.requires_grad() || gamma.requires_grad() || beta.requires_grad()));

    if (x.device() == Device::CPU) {
        const float* in = x.impl->data();
        const float* g = gamma.impl->data();
        const float* b = beta.impl->data();
        const float* m = mean.impl->data();
        const float* v = var.impl->data();
        float* dout = out.impl->data();

        for (int b_idx = 0; b_idx < B; ++b_idx) {
            for (int c = 0; c < C; ++c) {
                float inv_std = 1.0f / std::sqrt(v[c] + eps);
                for (int hw = 0; hw < H * W; ++hw) {
                    int idx = (b_idx * C + c) * H * W + hw;
                    float x_hat = (in[idx] - m[c]) * inv_std;
                    dout[idx] = g[c] * x_hat + b[c];
                }
            }
        }
    } else {
        cuda::launch_batchnorm_forward(out.impl->data(), x.impl->data(), gamma.impl->data(), beta.impl->data(),
                                       mean.impl->data(), var.impl->data(), B, C, H, W, eps);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<BatchNorm2DNode>(x, gamma, beta, mean, var, B, C, H, W, eps));
    }
    return out;
}

// -------------------------------------------------------------
// SOFTMAX (ARBITRARY DIMENSION)
// -------------------------------------------------------------
struct SoftmaxNode : public Node {
    Tensor out_saved;
    int outer_size, dim_size, inner_size;

    SoftmaxNode(const Tensor& in, const Tensor& out, int outer, int dim_s, int inner)
        : Node("Softmax", {in}), out_saved(out.detach()), outer_size(outer), dim_size(dim_s), inner_size(inner) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in = Tensor::zeros(out_saved.shape(), out_saved.device());

        if (out_saved.device() == Device::CPU) {
            const float* dout = grad_output.impl->data();
            const float* s = out_saved.impl->data();
            float* din = d_in.impl->data();

            #ifdef _OPENMP
            #pragma omp parallel for collapse(2) schedule(static)
            #endif
            for (int o = 0; o < outer_size; ++o) {
                for (int i = 0; i < inner_size; ++i) {
                    float dot_sum = 0.0f;
                    for (int d = 0; d < dim_size; ++d) {
                        int idx = (o * dim_size + d) * inner_size + i;
                        dot_sum += dout[idx] * s[idx];
                    }
                    for (int d = 0; d < dim_size; ++d) {
                        int idx = (o * dim_size + d) * inner_size + i;
                        din[idx] = s[idx] * (dout[idx] - dot_sum);
                    }
                }
            }
        } else {
            cuda::launch_softmax_backward(d_in.impl->data(), grad_output.impl->data(), out_saved.impl->data(),
                                          outer_size, dim_size, inner_size);
        }
        accumulate_grad(inputs[0], d_in);
    }
};

Tensor softmax(const Tensor& x, int dim) {
    if (x.numel() == 0) return x;
    int ndim = (int)x.shape().size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::runtime_error("Invalid dimension for softmax!");

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= (int)x.shape()[i];
    int dim_size = (int)x.shape()[dim];
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_size *= (int)x.shape()[i];

    Tensor out(x.shape(), x.device(), op_requires_grad(x.requires_grad()));

    if (x.device() == Device::CPU) {
        const float* in = x.impl->data();
        float* dout = out.impl->data();

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int o = 0; o < outer_size; ++o) {
            for (int i = 0; i < inner_size; ++i) {
                float max_val = in[(o * dim_size + 0) * inner_size + i];
                for (int d = 1; d < dim_size; ++d) {
                    float v = in[(o * dim_size + d) * inner_size + i];
                    if (v > max_val) max_val = v;
                }

                float sum_exp = 0.0f;
                for (int d = 0; d < dim_size; ++d) {
                    sum_exp += std::exp(in[(o * dim_size + d) * inner_size + i] - max_val);
                }

                for (int d = 0; d < dim_size; ++d) {
                    int idx = (o * dim_size + d) * inner_size + i;
                    dout[idx] = std::exp(in[idx] - max_val) / sum_exp;
                }
            }
        }
    } else {
        cuda::launch_softmax_forward(out.impl->data(), x.impl->data(), outer_size, dim_size, inner_size);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<SoftmaxNode>(x, out, outer_size, dim_size, inner_size));
    }
    return out;
}

// -------------------------------------------------------------
// MAXPOOL2D & AVGPOOL2D
// -------------------------------------------------------------
struct MaxPool2DNode : public Node {
    Tensor argmax_saved;
    std::vector<int64_t> in_shape;

    MaxPool2DNode(const Tensor& in, const Tensor& argmax)
        : Node("MaxPool2D", {in}), argmax_saved(argmax.detach()), in_shape(in.shape()) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in = Tensor::zeros(in_shape, grad_output.device());
        if (grad_output.device() == Device::CPU) {
            const float* dout = grad_output.impl->data();
            const float* argmax = argmax_saved.impl->data();
            float* din = d_in.impl->data();
            int64_t total = grad_output.numel();
            for (int64_t i = 0; i < total; ++i) {
                int in_idx = (int)argmax[i];
                if (in_idx >= 0) din[in_idx] += dout[i];
            }
        } else {
            cuda::launch_max_pool2d_backward(d_in.impl->data(), grad_output.impl->data(), argmax_saved.impl->data(),
                                             (int)in_shape[0], (int)in_shape[1], (int)in_shape[2], (int)in_shape[3],
                                             (int)grad_output.shape()[2], (int)grad_output.shape()[3]);
        }
        accumulate_grad(inputs[0], d_in);
    }
};

Tensor max_pool2d(const Tensor& x, int kernel_size, int stride, int padding) {
    if (x.shape().size() != 4) throw std::runtime_error("max_pool2d requires 4D tensor (B,C,H,W)!");
    int B = (int)x.shape()[0];
    int C = (int)x.shape()[1];
    int H = (int)x.shape()[2];
    int W = (int)x.shape()[3];

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    Tensor out({B, C, H_out, W_out}, x.device(), op_requires_grad(x.requires_grad()));
    Tensor argmax({B, C, H_out, W_out}, x.device(), false);

    if (x.device() == Device::CPU) {
        const float* in = x.impl->data();
        float* dout = out.impl->data();
        float* dargmax = argmax.impl->data();

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int h_out = 0; h_out < H_out; ++h_out) {
                    for (int w_out = 0; w_out < W_out; ++w_out) {
                        float max_val = -1e38f;
                        int max_idx = -1;
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int in_y = h_out * stride - padding + ky;
                                int in_x = w_out * stride - padding + kx;
                                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                    int in_idx = ((b * C + c) * H + in_y) * W + in_x;
                                    float val = in[in_idx];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_idx = in_idx;
                                    }
                                }
                            }
                        }
                        int out_idx = ((b * C + c) * H_out + h_out) * W_out + w_out;
                        dout[out_idx] = max_val;
                        dargmax[out_idx] = (float)max_idx;
                    }
                }
            }
        }
    } else {
        cuda::launch_max_pool2d_forward(out.impl->data(), argmax.impl->data(), x.impl->data(),
                                        B, C, H, W, kernel_size, kernel_size, H_out, W_out, stride, padding);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<MaxPool2DNode>(x, argmax));
    }
    return out;
}

struct AvgPool2DNode : public Node {
    std::vector<int64_t> in_shape;
    int kernel_size, stride, padding;

    AvgPool2DNode(const Tensor& in, int k, int s, int p)
        : Node("AvgPool2D", {in}), in_shape(in.shape()), kernel_size(k), stride(s), padding(p) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in = Tensor::zeros(in_shape, grad_output.device());
        int B = (int)in_shape[0], C = (int)in_shape[1], H = (int)in_shape[2], W = (int)in_shape[3];
        int H_out = (int)grad_output.shape()[2], W_out = (int)grad_output.shape()[3];
        float scale = 1.0f / (float)(kernel_size * kernel_size);

        if (grad_output.device() == Device::CPU) {
            const float* dout = grad_output.impl->data();
            float* din = d_in.impl->data();

            for (int b = 0; b < B; ++b) {
                for (int c = 0; c < C; ++c) {
                    for (int h_out = 0; h_out < H_out; ++h_out) {
                        for (int w_out = 0; w_out < W_out; ++w_out) {
                            float grad_val = dout[((b * C + c) * H_out + h_out) * W_out + w_out] * scale;
                            for (int ky = 0; ky < kernel_size; ++ky) {
                                for (int kx = 0; kx < kernel_size; ++kx) {
                                    int in_y = h_out * stride - padding + ky;
                                    int in_x = w_out * stride - padding + kx;
                                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                        din[((b * C + c) * H + in_y) * W + in_x] += grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            cuda::launch_avg_pool2d_backward(d_in.impl->data(), grad_output.impl->data(),
                                             B, C, H, W, kernel_size, kernel_size, H_out, W_out, stride, padding);
        }
        accumulate_grad(inputs[0], d_in);
    }
};

Tensor avg_pool2d(const Tensor& x, int kernel_size, int stride, int padding) {
    if (x.shape().size() != 4) throw std::runtime_error("avg_pool2d requires 4D tensor (B,C,H,W)!");
    int B = (int)x.shape()[0];
    int C = (int)x.shape()[1];
    int H = (int)x.shape()[2];
    int W = (int)x.shape()[3];

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    Tensor out({B, C, H_out, W_out}, x.device(), op_requires_grad(x.requires_grad()));

    if (x.device() == Device::CPU) {
        const float* in = x.impl->data();
        float* dout = out.impl->data();
        float scale = 1.0f / (float)(kernel_size * kernel_size);

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int h_out = 0; h_out < H_out; ++h_out) {
                    for (int w_out = 0; w_out < W_out; ++w_out) {
                        float sum_val = 0.0f;
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int in_y = h_out * stride - padding + ky;
                                int in_x = w_out * stride - padding + kx;
                                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                    sum_val += in[((b * C + c) * H + in_y) * W + in_x];
                                }
                            }
                        }
                        dout[((b * C + c) * H_out + h_out) * W_out + w_out] = sum_val * scale;
                    }
                }
            }
        }
    } else {
        cuda::launch_avg_pool2d_forward(out.impl->data(), x.impl->data(),
                                        B, C, H, W, kernel_size, kernel_size, H_out, W_out, stride, padding);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<AvgPool2DNode>(x, kernel_size, stride, padding));
    }
    return out;
}

// -------------------------------------------------------------
// BATCHNORM1D
// -------------------------------------------------------------
struct BatchNorm1DNode : public Node {
    Tensor x_saved, gamma_saved, mean_saved, var_saved;
    int B, D;
    float eps;

    BatchNorm1DNode(const Tensor& x, const Tensor& gamma, const Tensor& beta, const Tensor& mean, const Tensor& var,
                    int B, int D, float eps)
        : Node("BatchNorm1D", {x, gamma, beta}),
          x_saved(x.detach()), gamma_saved(gamma.detach()), mean_saved(mean.detach()), var_saved(var.detach()),
          B(B), D(D), eps(eps) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_x = Tensor::zeros(x_saved.shape(), Device::CPU);
        Tensor d_gamma = Tensor::zeros(gamma_saved.shape(), Device::CPU);
        Tensor d_beta = Tensor::zeros(gamma_saved.shape(), Device::CPU);

        Tensor cpu_dout = grad_output.to(Device::CPU);
        Tensor cpu_x = x_saved.to(Device::CPU);
        Tensor cpu_gamma = gamma_saved.to(Device::CPU);
        Tensor cpu_mean = mean_saved.to(Device::CPU);
        Tensor cpu_var = var_saved.to(Device::CPU);

        const float* dout = cpu_dout.impl->data();
        const float* x = cpu_x.impl->data();
        const float* gamma = cpu_gamma.impl->data();
        const float* mean = cpu_mean.impl->data();
        const float* var = cpu_var.impl->data();

        float* dx = d_x.impl->data();
        float* dg = d_gamma.impl->data();
        float* db = d_beta.impl->data();

        for (int d = 0; d < D; ++d) {
            float inv_std = 1.0f / std::sqrt(var[d] + eps);
            float d_gamma_d = 0.0f;
            float d_beta_d = 0.0f;

            for (int b = 0; b < B; ++b) {
                float x_hat = (x[b * D + d] - mean[d]) * inv_std;
                d_gamma_d += dout[b * D + d] * x_hat;
                d_beta_d += dout[b * D + d];
            }
            dg[d] = d_gamma_d;
            db[d] = d_beta_d;

            for (int b = 0; b < B; ++b) {
                float x_hat = (x[b * D + d] - mean[d]) * inv_std;
                dx[b * D + d] = (gamma[d] * inv_std / (float)B) * ((float)B * dout[b * D + d] - d_beta_d - x_hat * d_gamma_d);
            }
        }

        accumulate_grad(inputs[0], d_x.to(x_saved.device()));
        accumulate_grad(inputs[1], d_gamma.to(gamma_saved.device()));
        accumulate_grad(inputs[2], d_beta.to(gamma_saved.device()));
    }
};

Tensor batch_norm1d(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                    Tensor& running_mean, Tensor& running_var,
                    float momentum, float eps, bool training) {
    if (x.shape().size() != 2) throw std::runtime_error("batch_norm1d currently expects 2D tensor (B, D)!");
    int B = (int)x.shape()[0];
    int D = (int)x.shape()[1];

    Tensor mean(std::vector<int64_t>{(int64_t)D}, x.device());
    Tensor var(std::vector<int64_t>{(int64_t)D}, x.device());

    if (training) {
        Tensor cpu_x = x.to(Device::CPU);
        Tensor cpu_mean(std::vector<int64_t>{(int64_t)D}, Device::CPU);
        Tensor cpu_var(std::vector<int64_t>{(int64_t)D}, Device::CPU);

        const float* x_ptr = cpu_x.impl->data();
        float* m_ptr = cpu_mean.impl->data();
        float* v_ptr = cpu_var.impl->data();

        for (int d = 0; d < D; ++d) {
            float sum_val = 0.0f;
            for (int b = 0; b < B; ++b) sum_val += x_ptr[b * D + d];
            float cur_mean = sum_val / (float)B;
            m_ptr[d] = cur_mean;

            float sum_sq = 0.0f;
            for (int b = 0; b < B; ++b) {
                float diff = x_ptr[b * D + d] - cur_mean;
                sum_sq += diff * diff;
            }
            v_ptr[d] = sum_sq / (float)B;
        }

        // Update running stats
        Tensor cpu_rm = running_mean.to(Device::CPU);
        Tensor cpu_rv = running_var.to(Device::CPU);
        for (int d = 0; d < D; ++d) {
            cpu_rm.impl->data()[d] = (1.0f - momentum) * cpu_rm.impl->data()[d] + momentum * m_ptr[d];
            cpu_rv.impl->data()[d] = (1.0f - momentum) * cpu_rv.impl->data()[d] + momentum * v_ptr[d];
        }
        running_mean = cpu_rm.to(running_mean.device());
        running_var = cpu_rv.to(running_var.device());

        mean = cpu_mean.to(x.device());
        var = cpu_var.to(x.device());
    } else {
        mean = running_mean;
        var = running_var;
    }

    Tensor out(x.shape(), x.device(), op_requires_grad(x.requires_grad() || gamma.requires_grad() || beta.requires_grad()));
    Tensor cpu_out(x.shape(), Device::CPU);
    Tensor cpu_in = x.to(Device::CPU);
    Tensor cpu_g = gamma.to(Device::CPU);
    Tensor cpu_b = beta.to(Device::CPU);
    Tensor cpu_m = mean.to(Device::CPU);
    Tensor cpu_v = var.to(Device::CPU);

    const float* in = cpu_in.impl->data();
    const float* g = cpu_g.impl->data();
    const float* b = cpu_b.impl->data();
    const float* m = cpu_m.impl->data();
    const float* v = cpu_v.impl->data();
    float* dout = cpu_out.impl->data();

    for (int b_idx = 0; b_idx < B; ++b_idx) {
        for (int d = 0; d < D; ++d) {
            float inv_std = 1.0f / std::sqrt(v[d] + eps);
            float x_hat = (in[b_idx * D + d] - m[d]) * inv_std;
            dout[b_idx * D + d] = g[d] * x_hat + b[d];
        }
    }
    out = (x.device() == Device::CUDA) ? cpu_out.to(Device::CUDA) : cpu_out;

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<BatchNorm1DNode>(x, gamma, beta, mean, var, B, D, eps));
    }
    return out;
}

// -------------------------------------------------------------
// CONV1D
// -------------------------------------------------------------
struct Conv1DNode : public Node {
    Tensor x_saved, weight_saved, bias_saved;
    int B, C, L, O, K, L_out, stride, padding;

    Conv1DNode(const Tensor& x, const Tensor& weight, const Tensor& bias,
               int B, int C, int L, int O, int K, int L_out, int stride, int padding)
        : Node("Conv1D", {x, weight, bias}),
          x_saved(x.detach()), weight_saved(weight.detach()), bias_saved(bias.detach()),
          B(B), C(C), L(L), O(O), K(K), L_out(L_out), stride(stride), padding(padding) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in = Tensor::zeros(x_saved.shape(), Device::CPU);
        Tensor d_weight = Tensor::zeros(weight_saved.shape(), Device::CPU);
        Tensor d_bias = Tensor::zeros(bias_saved.shape(), Device::CPU);

        Tensor cpu_dout = grad_output.to(Device::CPU);
        Tensor cpu_x = x_saved.to(Device::CPU);
        Tensor cpu_w = weight_saved.to(Device::CPU);

        const float* dout = cpu_dout.impl->data();
        const float* in = cpu_x.impl->data();
        const float* weight = cpu_w.impl->data();

        float* din = d_in.impl->data();
        float* dw = d_weight.impl->data();
        float* db = d_bias.impl->data();

        for (int b = 0; b < B; ++b) {
            for (int o = 0; o < O; ++o) {
                for (int l_out = 0; l_out < L_out; ++l_out) {
                    float dout_val = dout[(b * O + o) * L_out + l_out];
                    db[o] += dout_val;

                    for (int c = 0; c < C; ++c) {
                        for (int k = 0; k < K; ++k) {
                            int in_l = l_out * stride - padding + k;
                            if (in_l >= 0 && in_l < L) {
                                int in_idx = (b * C + c) * L + in_l;
                                int w_idx = (o * C + c) * K + k;
                                dw[w_idx] += dout_val * in[in_idx];
                                din[in_idx] += dout_val * weight[w_idx];
                            }
                        }
                    }
                }
            }
        }

        accumulate_grad(inputs[0], d_in.to(x_saved.device()));
        accumulate_grad(inputs[1], d_weight.to(weight_saved.device()));
        accumulate_grad(inputs[2], d_bias.to(bias_saved.device()));
    }
};

Tensor conv1d(const Tensor& x, const Tensor& weight, const Tensor& bias, int stride, int padding) {
    if (x.shape().size() != 3 || weight.shape().size() != 3) {
        throw std::runtime_error("conv1d requires 3D tensors: input (B,C,L) and weight (O,C,K)!");
    }
    int B = (int)x.shape()[0];
    int C = (int)x.shape()[1];
    int L = (int)x.shape()[2];

    int O = (int)weight.shape()[0];
    int C_in = (int)weight.shape()[1];
    int K = (int)weight.shape()[2];

    if (C != C_in) throw std::runtime_error("Input channel mismatch in conv1d!");

    int L_out = (L + 2 * padding - K) / stride + 1;
    Tensor out({B, O, L_out}, x.device(), op_requires_grad(x.requires_grad() || weight.requires_grad() || bias.requires_grad()));

    if (x.device() == Device::CPU) {
        const float* in = x.impl->data();
        const float* w = weight.impl->data();
        const float* b = bias.impl->data();
        float* dout = out.impl->data();

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int b_idx = 0; b_idx < B; ++b_idx) {
            for (int o = 0; o < O; ++o) {
                for (int l_out = 0; l_out < L_out; ++l_out) {
                    float val = b ? b[o] : 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int k = 0; k < K; ++k) {
                            int in_l = l_out * stride - padding + k;
                            if (in_l >= 0 && in_l < L) {
                                val += in[(b_idx * C + c) * L + in_l] * w[(o * C + c) * K + k];
                            }
                        }
                    }
                    dout[(b_idx * O + o) * L_out + l_out] = val;
                }
            }
        }
    } else {
        cuda::launch_conv1d_forward(out.impl->data(), x.impl->data(), weight.impl->data(), bias.impl->data(),
                                    B, C, L, O, K, L_out, stride, padding);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<Conv1DNode>(x, weight, bias, B, C, L, O, K, L_out, stride, padding));
    }
    return out;
}

// -------------------------------------------------------------
// EMBEDDING LOOKUP
// -------------------------------------------------------------
struct EmbeddingNode : public Node {
    Tensor indices_saved;
    std::vector<int64_t> weight_shape;

    EmbeddingNode(const Tensor& indices, const Tensor& weight)
        : Node("Embedding", {weight}), indices_saved(indices.detach()), weight_shape(weight.shape()) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_weight = Tensor::zeros(weight_shape, grad_output.device());
        int64_t num_indices = indices_saved.numel();
        int64_t emb_dim = weight_shape.back();

        if (grad_output.device() == Device::CPU) {
            const float* dout = grad_output.impl->data();
            const float* idx_ptr = indices_saved.impl->data();
            float* dw = d_weight.impl->data();

            for (int64_t i = 0; i < num_indices; ++i) {
                int token_id = (int)idx_ptr[i];
                if (token_id >= 0 && token_id < weight_shape[0]) {
                    for (int64_t d = 0; d < emb_dim; ++d) {
                        dw[token_id * emb_dim + d] += dout[i * emb_dim + d];
                    }
                }
            }
        } else {
            cuda::launch_embedding_backward(d_weight.impl->data(), grad_output.impl->data(),
                                            indices_saved.impl->data(), (int)num_indices, (int)emb_dim);
        }
        accumulate_grad(inputs[0], d_weight);
    }
};

Tensor embedding(const Tensor& indices, const Tensor& weight) {
    if (weight.shape().size() != 2) throw std::runtime_error("Embedding weight must be 2D tensor [vocab_size, embedding_dim]!");
    int64_t num_indices = indices.numel();
    int64_t emb_dim = weight.shape()[1];

    auto out_shape = indices.shape();
    out_shape.push_back(emb_dim);

    Tensor out(out_shape, weight.device(), op_requires_grad(weight.requires_grad()));

    if (weight.device() == Device::CPU) {
        const float* idx_ptr = indices.impl->data();
        const float* w_ptr = weight.impl->data();
        float* dout = out.impl->data();

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t i = 0; i < num_indices; ++i) {
            int token_id = (int)idx_ptr[i];
            if (token_id >= 0 && token_id < weight.shape()[0]) {
                std::memcpy(dout + i * emb_dim, w_ptr + token_id * emb_dim, emb_dim * sizeof(float));
            } else {
                std::memset(dout + i * emb_dim, 0, emb_dim * sizeof(float));
            }
        }
    } else {
        cuda::launch_embedding_forward(out.impl->data(), indices.impl->data(), weight.impl->data(),
                                       (int)num_indices, (int)emb_dim);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<EmbeddingNode>(indices, weight));
    }
    return out;
}

// -------------------------------------------------------------
// TENSOR MEMBER OPERATOR DISPATCHERS
// -------------------------------------------------------------
Tensor Tensor::operator+(const Tensor& other) const { return add(*this, other); }
Tensor Tensor::operator-(const Tensor& other) const { return sub(*this, other); }
Tensor Tensor::operator*(const Tensor& other) const { return mul(*this, other); }
Tensor Tensor::operator/(const Tensor& other) const { return div(*this, other); }
Tensor Tensor::operator-() const { return neg(*this); }
Tensor Tensor::matmul(const Tensor& other) const { return sovereigngrad::matmul(*this, other); }
Tensor Tensor::relu() const { return sovereigngrad::relu(*this); }
Tensor Tensor::gelu() const { return sovereigngrad::gelu(*this); }
Tensor Tensor::silu() const { return sovereigngrad::silu(*this); }
Tensor Tensor::sigmoid() const { return sovereigngrad::sigmoid(*this); }
Tensor Tensor::tanh() const { return sovereigngrad::tanh_op(*this); }
Tensor Tensor::exp() const { return sovereigngrad::exp_op(*this); }
Tensor Tensor::log() const { return sovereigngrad::log_op(*this); }
Tensor Tensor::sqrt() const { return sovereigngrad::sqrt_op(*this); }
Tensor Tensor::pow(float exponent) const { return sovereigngrad::pow_op(*this, exponent); }
Tensor Tensor::clamp(float min_val, float max_val) const { return sovereigngrad::clamp_op(*this, min_val, max_val); }
Tensor Tensor::dropout(float p, bool training) const { return sovereigngrad::dropout_op(*this, p, training); }
Tensor Tensor::sum(int dim, bool keepdim) const { return sovereigngrad::sum(*this, dim, keepdim); }
Tensor Tensor::sum() const { return sovereigngrad::sum(*this); }
Tensor Tensor::mean(int dim, bool keepdim) const { return sovereigngrad::mean(*this, dim, keepdim); }
Tensor Tensor::mean() const { return sovereigngrad::mean(*this); }
Tensor Tensor::max(int dim, bool keepdim) const { return sovereigngrad::max_op(*this, dim, keepdim); }
Tensor Tensor::softmax(int dim) const { return sovereigngrad::softmax(*this, dim); }
Tensor Tensor::max_pool2d(int kernel_size, int stride, int padding) const { return sovereigngrad::max_pool2d(*this, kernel_size, stride, padding); }
Tensor Tensor::avg_pool2d(int kernel_size, int stride, int padding) const { return sovereigngrad::avg_pool2d(*this, kernel_size, stride, padding); }
Tensor Tensor::masked_fill(const Tensor& mask, float value) const { return sovereigngrad::masked_fill(*this, mask, value); }
Tensor Tensor::rmsnorm(const Tensor& gamma, float eps) const { return sovereigngrad::rms_norm(*this, gamma, eps); }

// -------------------------------------------------------------
// OPERATOR & KERNEL FUSION
// -------------------------------------------------------------
struct FusedLinearGeluNode : public Node {
    Tensor x_saved, w_saved, z_saved;

    FusedLinearGeluNode(const Tensor& x, const Tensor& w, const Tensor& b, const Tensor& z)
        : Node("FusedLinearGELU", {x, w, b}), x_saved(x.detach()), w_saved(w.detach()), z_saved(z.detach()) {}

    void backward(const Tensor& grad_output) override {
        // 1. Derivative of GELU: d_z = grad_output * dgelu(z)
        Tensor d_z(z_saved.shape(), z_saved.device());
        int64_t n = z_saved.numel();
        const float* z_data = z_saved.impl->data();
        const float* dout_data = grad_output.impl->data();
        float* dz_data = d_z.impl->data();
        float c = 0.79788456f;

        for (int64_t i = 0; i < n; ++i) {
            float xi = z_data[i];
            float u = c * (xi + 0.044715f * xi * xi * xi);
            float th = std::tanh(u);
            float du_dx = c * (1.0f + 3.0f * 0.044715f * xi * xi);
            float grad = 0.5f * (1.0f + th) + 0.5f * xi * (1.0f - th * th) * du_dx;
            dz_data[i] = dout_data[i] * grad;
        }

        // 2. Backpropagation to x: d_x = d_z @ W^T
        Tensor wt = w_saved.transpose(0, 1).contiguous();
        accumulate_grad(inputs[0], d_z.matmul(wt));

        // 3. Backpropagation to W: d_W = x^T @ d_z
        Tensor xt = x_saved.transpose(0, 1).contiguous();
        accumulate_grad(inputs[1], xt.matmul(d_z));

        // 4. Backpropagation to bias b: d_b = sum(d_z, 0)
        accumulate_grad(inputs[2], reduce_to_shape(d_z, inputs[2]->shape));
    }
};

Tensor fused_linear_gelu(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    Tensor z = x.matmul(weight) + bias;
    Tensor out = gelu(z);

    if (AutoGradMode::is_enabled() && (x.requires_grad() || weight.requires_grad() || bias.requires_grad())) {
        out.set_creator(std::make_shared<FusedLinearGeluNode>(x, weight, bias, z));
    }
    return out;
}

struct FusedAddLayerNormNode : public Node {
    Tensor sum_x_saved, gamma_saved, beta_saved;
    float eps;

    FusedAddLayerNormNode(const Tensor& x, const Tensor& residual, const Tensor& gamma, const Tensor& beta,
                          const Tensor& sum_x, float eps)
        : Node("FusedAddLayerNorm", {x, residual, gamma, beta}),
          sum_x_saved(sum_x.detach()), gamma_saved(gamma.detach()), beta_saved(beta.detach()), eps(eps) {}

    void backward(const Tensor& grad_output) override {
        // Backprop through LayerNorm with respect to sum_x, gamma, beta
        int64_t B = sum_x_saved.shape()[0];
        int64_t D = sum_x_saved.shape()[1];

        Tensor d_sum_x(sum_x_saved.shape(), sum_x_saved.device());
        Tensor d_gamma(gamma_saved.shape(), gamma_saved.device());
        Tensor d_beta(beta_saved.shape(), beta_saved.device());

        if (sum_x_saved.device() == Device::CPU) {
            const float* in_ptr = sum_x_saved.impl->data();
            const float* dout_ptr = grad_output.impl->data();
            const float* g_ptr = gamma_saved.impl->data();
            float* din_ptr = d_sum_x.impl->data();
            float* dg_ptr = d_gamma.impl->data();
            float* db_ptr = d_beta.impl->data();

            std::memset(dg_ptr, 0, D * sizeof(float));
            std::memset(db_ptr, 0, D * sizeof(float));

            for (int64_t b = 0; b < B; ++b) {
                float mean = 0.0f;
                for (int64_t d = 0; d < D; ++d) mean += in_ptr[b * D + d];
                mean /= (float)D;

                float var = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    float diff = in_ptr[b * D + d] - mean;
                    var += diff * diff;
                }
                var /= (float)D;
                float inv_std = 1.0f / std::sqrt(var + eps);

                float sum_dout = 0.0f;
                float sum_dout_xhat = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    float x_hat = (in_ptr[b * D + d] - mean) * inv_std;
                    float dy = dout_ptr[b * D + d];
                    sum_dout += dy * g_ptr[d];
                    sum_dout_xhat += dy * g_ptr[d] * x_hat;
                    dg_ptr[d] += dy * x_hat;
                    db_ptr[d] += dy;
                }

                for (int64_t d = 0; d < D; ++d) {
                    float x_hat = (in_ptr[b * D + d] - mean) * inv_std;
                    float term1 = (float)D * dout_ptr[b * D + d] * g_ptr[d];
                    float term2 = sum_dout;
                    float term3 = x_hat * sum_dout_xhat;
                    din_ptr[b * D + d] = (inv_std / (float)D) * (term1 - term2 - term3);
                }
            }
        } else {
            cuda::launch_layernorm_backward(d_sum_x.impl->data(), d_gamma.impl->data(), d_beta.impl->data(),
                                           grad_output.impl->data(), sum_x_saved.impl->data(), gamma_saved.impl->data(),
                                           (int)B, (int)D, eps);
        }

        // Distribute d_sum_x to both x and residual branches
        accumulate_grad(inputs[0], reduce_to_shape(d_sum_x, inputs[0]->shape));
        accumulate_grad(inputs[1], reduce_to_shape(d_sum_x, inputs[1]->shape));
        accumulate_grad(inputs[2], d_gamma);
        accumulate_grad(inputs[3], d_beta);
    }
};

Tensor fused_add_layernorm(const Tensor& x, const Tensor& residual, const Tensor& gamma, const Tensor& beta, float eps) {
    Tensor sum_x = x + residual;
    Tensor out = layer_norm(sum_x, gamma, beta, eps);

    if (AutoGradMode::is_enabled() && (x.requires_grad() || residual.requires_grad() || gamma.requires_grad() || beta.requires_grad())) {
        out.set_creator(std::make_shared<FusedAddLayerNormNode>(x, residual, gamma, beta, sum_x, eps));
    }
    return out;
}

Tensor fused_scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V, float scale) {
    Tensor K_t = K.transpose(0, 1).contiguous();
    Tensor scores = Q.matmul(K_t) * scale;
    Tensor attn_weights = softmax(scores, /*dim=*/-1);
    return attn_weights.matmul(V);
}

// -------------------------------------------------------------
// DYNAMIC SLICING & INDEXING
// -------------------------------------------------------------
struct SliceNode : public Node {
    int dim;
    int64_t start, end, step;
    std::vector<int64_t> orig_shape;

    SliceNode(const Tensor& in, int dim, int64_t start, int64_t end, int64_t step)
        : Node("Slice", {in}), dim(dim), start(start), end(end), step(step), orig_shape(in.shape()) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in = Tensor::zeros(orig_shape, grad_output.device());
        int rank = (int)orig_shape.size();
        auto in_strides = default_strides(orig_shape);
        auto out_strides = default_strides(grad_output.shape());

        if (grad_output.device() == Device::CPU) {
            const float* dout = grad_output.impl->data();
            float* din = d_in.impl->data();
            int64_t total = grad_output.numel();

            for (int64_t idx = 0; idx < total; ++idx) {
                int64_t in_offset = 0;
                int64_t rem = idx;
                for (int d = 0; d < rank; ++d) {
                    int64_t out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    int64_t in_coord = (d == dim) ? (start + out_coord * step) : out_coord;
                    in_offset += in_coord * in_strides[d];
                }
                din[in_offset] += dout[idx];
            }
        } else {
            cuda::launch_slice_backward(d_in.impl->data(), grad_output.impl->data(),
                                        orig_shape.data(), in_strides.data(),
                                        grad_output.shape().data(), out_strides.data(),
                                        rank, dim, start, step, (int)grad_output.numel());
        }
        accumulate_grad(inputs[0], d_in);
    }
};

Tensor slice_op(const Tensor& x, int dim, int64_t start, int64_t end, int64_t step) {
    int rank = (int)x.shape().size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) throw std::runtime_error("slice dim out of bounds!");

    int64_t dim_size = x.shape()[dim];
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;
    start = std::max(0LL, std::min(start, dim_size));
    end = std::max(start, std::min(end, dim_size));
    step = std::max(1LL, step);

    int64_t out_dim_size = (end - start + step - 1) / step;
    auto out_shape = x.shape();
    out_shape[dim] = out_dim_size;

    Tensor out(out_shape, x.device(), op_requires_grad(x.requires_grad()));
    auto in_strides = default_strides(x.shape());
    auto out_strides = default_strides(out_shape);

    if (x.device() == Device::CPU) {
        const float* in_ptr = x.impl->data();
        float* out_ptr = out.impl->data();
        int64_t total = out.numel();

        for (int64_t idx = 0; idx < total; ++idx) {
            int64_t in_offset = 0;
            int64_t rem = idx;
            for (int d = 0; d < rank; ++d) {
                int64_t out_coord = rem / out_strides[d];
                rem %= out_strides[d];
                int64_t in_coord = (d == dim) ? (start + out_coord * step) : out_coord;
                in_offset += in_coord * in_strides[d];
            }
            out_ptr[idx] = in_ptr[in_offset];
        }
    } else {
        cuda::launch_slice_forward(out.impl->data(), x.impl->data(),
                                   x.shape().data(), in_strides.data(),
                                   out_shape.data(), out_strides.data(),
                                   rank, dim, start, step, (int)out.numel());
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<SliceNode>(x, dim, start, end, step));
    }
    return out;
}

struct SelectNode : public Node {
    int dim;
    int64_t index;
    std::vector<int64_t> orig_shape;

    SelectNode(const Tensor& in, int dim, int64_t index)
        : Node("Select", {in}), dim(dim), index(index), orig_shape(in.shape()) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in = Tensor::zeros(orig_shape, grad_output.device());
        int rank = (int)orig_shape.size();
        auto in_strides = default_strides(orig_shape);

        std::vector<int64_t> expanded_shape = orig_shape;
        expanded_shape[dim] = 1;
        Tensor reshaped_grad = grad_output.reshape(expanded_shape);
        auto out_strides = default_strides(expanded_shape);

        if (grad_output.device() == Device::CPU) {
            const float* dout = reshaped_grad.impl->data();
            float* din = d_in.impl->data();
            int64_t total = reshaped_grad.numel();

            for (int64_t idx = 0; idx < total; ++idx) {
                int64_t in_offset = 0;
                int64_t rem = idx;
                for (int d = 0; d < rank; ++d) {
                    int64_t out_coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    int64_t in_coord = (d == dim) ? index : out_coord;
                    in_offset += in_coord * in_strides[d];
                }
                din[in_offset] += dout[idx];
            }
        }
        accumulate_grad(inputs[0], d_in);
    }
};

Tensor select_op(const Tensor& x, int dim, int64_t index) {
    int rank = (int)x.shape().size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) throw std::runtime_error("select dim out of bounds!");

    int64_t dim_size = x.shape()[dim];
    if (index < 0) index += dim_size;
    if (index < 0 || index >= dim_size) throw std::runtime_error("select index out of range!");

    Tensor sliced = slice_op(x, dim, index, index + 1, 1);
    std::vector<int64_t> out_shape;
    for (int i = 0; i < rank; ++i) {
        if (i != dim) out_shape.push_back(x.shape()[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor out = sliced.reshape(out_shape);
    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<SelectNode>(x, dim, index));
    }
    return out;
}

struct IndexSelectNode : public Node {
    int dim;
    Tensor index_saved;
    std::vector<int64_t> orig_shape;

    IndexSelectNode(const Tensor& in, int dim, const Tensor& index)
        : Node("IndexSelect", {in}), dim(dim), index_saved(index.detach()), orig_shape(in.shape()) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in = Tensor::zeros(orig_shape, grad_output.device());
        int rank = (int)orig_shape.size();
        int64_t outer_size = 1;
        for (int i = 0; i < dim; ++i) outer_size *= orig_shape[i];
        int64_t in_dim_size = orig_shape[dim];
        int64_t out_dim_size = grad_output.shape()[dim];
        int64_t inner_size = 1;
        for (int i = dim + 1; i < rank; ++i) inner_size *= orig_shape[i];

        Tensor cpu_dout = grad_output.to(Device::CPU);
        Tensor cpu_idx = index_saved.to(Device::CPU);
        Tensor cpu_din = d_in.to(Device::CPU);

        const float* dout = cpu_dout.impl->data();
        const float* indices = cpu_idx.impl->data();
        float* din = cpu_din.impl->data();

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int64_t o = 0; o < outer_size; ++o) {
            for (int64_t i = 0; i < out_dim_size; ++i) {
                int64_t idx_val = (int64_t)indices[i];
                if (idx_val >= 0 && idx_val < in_dim_size) {
                    const float* src = dout + (o * out_dim_size + i) * inner_size;
                    float* dst = din + (o * in_dim_size + idx_val) * inner_size;
                    for (int64_t in_idx = 0; in_idx < inner_size; ++in_idx) {
                        #pragma omp atomic
                        dst[in_idx] += src[in_idx];
                    }
                }
            }
        }

        accumulate_grad(inputs[0], cpu_din.to(grad_output.device()));
    }
};

Tensor index_select_op(const Tensor& x, int dim, const Tensor& index) {
    int rank = (int)x.shape().size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) throw std::runtime_error("index_select dim out of bounds!");

    std::vector<int64_t> out_shape = x.shape();
    out_shape[dim] = index.numel();

    Tensor out(out_shape, x.device(), op_requires_grad(x.requires_grad()));
    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= x.shape()[i];
    int64_t in_dim_size = x.shape()[dim];
    int64_t out_dim_size = index.numel();
    int64_t inner_size = 1;
    for (int i = dim + 1; i < rank; ++i) inner_size *= x.shape()[i];

    Tensor cpu_x = x.to(Device::CPU);
    Tensor cpu_idx = index.to(Device::CPU);
    Tensor cpu_out(out_shape, Device::CPU, op_requires_grad(x.requires_grad()));

    const float* in = cpu_x.impl->data();
    const float* idx_ptr = cpu_idx.impl->data();
    float* out_ptr = cpu_out.impl->data();

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t i = 0; i < out_dim_size; ++i) {
            int64_t idx_val = (int64_t)idx_ptr[i];
            if (idx_val >= 0 && idx_val < in_dim_size) {
                const float* src = in + (o * in_dim_size + idx_val) * inner_size;
                float* dst = out_ptr + (o * out_dim_size + i) * inner_size;
                for (int64_t in_idx = 0; in_idx < inner_size; ++in_idx) {
                    dst[in_idx] = src[in_idx];
                }
            } else {
                float* dst = out_ptr + (o * out_dim_size + i) * inner_size;
                std::memset(dst, 0, inner_size * sizeof(float));
            }
        }
    }

    if (x.device() == Device::CUDA) {
        out = cpu_out.to(Device::CUDA);
    } else {
        out = cpu_out;
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<IndexSelectNode>(x, dim, index));
    }
    return out;
}

// -------------------------------------------------------------
// FLASHATTENTION-2 TILED ONLINE SOFTMAX
// -------------------------------------------------------------
struct FlashAttentionNode : public Node {
    Tensor q_saved, k_saved, v_saved, out_saved, lse_saved;
    bool is_causal;
    float scale;
    int B, H, L, S, D;
    std::vector<int64_t> orig_q_shape, orig_k_shape, orig_v_shape;

    FlashAttentionNode(const Tensor& Q, const Tensor& K, const Tensor& V,
                       const Tensor& out, const Tensor& lse,
                       bool is_causal, float scale,
                       int B, int H, int L, int S, int D,
                       const std::vector<int64_t>& orig_q,
                       const std::vector<int64_t>& orig_k,
                       const std::vector<int64_t>& orig_v)
        : Node("FlashAttention", {Q, K, V}),
          q_saved(Q.detach()), k_saved(K.detach()), v_saved(V.detach()),
          out_saved(out.detach()), lse_saved(lse.detach()),
          is_causal(is_causal), scale(scale),
          B(B), H(H), L(L), S(S), D(D),
          orig_q_shape(orig_q), orig_k_shape(orig_k), orig_v_shape(orig_v) {}

    void backward(const Tensor& grad_output) override {
        Tensor dQ = Tensor::zeros(q_saved.shape(), q_saved.device());
        Tensor dK = Tensor::zeros(k_saved.shape(), k_saved.device());
        Tensor dV = Tensor::zeros(v_saved.shape(), v_saved.device());

        Tensor cpu_dO = grad_output.to(Device::CPU).contiguous();
        Tensor cpu_Q = q_saved.to(Device::CPU).contiguous();
        Tensor cpu_K = k_saved.to(Device::CPU).contiguous();
        Tensor cpu_V = v_saved.to(Device::CPU).contiguous();
        Tensor cpu_O = out_saved.to(Device::CPU).contiguous();
        Tensor cpu_LSE = lse_saved.to(Device::CPU).contiguous();

        Tensor cpu_dQ(q_saved.shape(), Device::CPU);
        Tensor cpu_dK(k_saved.shape(), Device::CPU);
        Tensor cpu_dV(v_saved.shape(), Device::CPU);

        cuda::launch_flash_attention_backward(
            cpu_dQ.impl->data(), cpu_dK.impl->data(), cpu_dV.impl->data(),
            cpu_dO.impl->data(), cpu_Q.impl->data(), cpu_K.impl->data(),
            cpu_V.impl->data(), cpu_O.impl->data(), cpu_LSE.impl->data(),
            B, H, L, S, D, scale, is_causal
        );

        accumulate_grad(inputs[0], cpu_dQ.to(q_saved.device()).reshape(orig_q_shape));
        accumulate_grad(inputs[1], cpu_dK.to(k_saved.device()).reshape(orig_k_shape));
        accumulate_grad(inputs[2], cpu_dV.to(v_saved.device()).reshape(orig_v_shape));
    }
};

Tensor flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V, bool is_causal, float scale) {
    int64_t q_rank = Q.shape().size();
    if (q_rank < 2 || K.shape().size() < 2 || V.shape().size() < 2) {
        throw std::runtime_error("FlashAttention requires at least 2D tensors [..., L, D]!");
    }

    int64_t D = Q.shape().back();
    if (scale <= 0.0f) {
        scale = 1.0f / std::sqrt((float)D);
    }

    int B = 1, H = 1, L = 1, S = 1;
    if (q_rank == 2) {
        B = 1; H = 1; L = (int)Q.shape()[0]; S = (int)K.shape()[0];
    } else if (q_rank == 3) {
        B = (int)Q.shape()[0]; H = 1; L = (int)Q.shape()[1]; S = (int)K.shape()[1];
    } else if (q_rank == 4) {
        B = (int)Q.shape()[0]; H = (int)Q.shape()[1]; L = (int)Q.shape()[2]; S = (int)K.shape()[2];
    } else {
        throw std::runtime_error("FlashAttention supports up to 4D tensors [B, H, L, D]!");
    }

    bool req_grad = op_requires_grad(Q.requires_grad() || K.requires_grad() || V.requires_grad());
    Tensor out(Q.shape(), Q.device(), req_grad);
    Tensor lse({B * H, L}, Device::CPU, false);

    Tensor cpu_Q = Q.to(Device::CPU).contiguous();
    Tensor cpu_K = K.to(Device::CPU).contiguous();
    Tensor cpu_V = V.to(Device::CPU).contiguous();
    Tensor cpu_out(Q.shape(), Device::CPU, req_grad);

    cuda::launch_flash_attention_forward(
        cpu_out.impl->data(), lse.impl->data(),
        cpu_Q.impl->data(), cpu_K.impl->data(), cpu_V.impl->data(),
        B, H, L, S, (int)D, scale, is_causal
    );

    if (Q.device() == Device::CUDA) {
        out = cpu_out.to(Device::CUDA);
    } else {
        out = cpu_out;
    }

    if (AutoGradMode::is_enabled() && req_grad) {
        out.set_creator(std::make_shared<FlashAttentionNode>(
            Q, K, V, out, lse, is_causal, scale, B, H, L, S, (int)D,
            Q.shape(), K.shape(), V.shape()
        ));
    }
    return out;
}

} // namespace sovereigngrad

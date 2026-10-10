#pragma once

#include "sovereigngrad/allocator.hpp"
#include <vector>
#include <memory>
#include <string>
#include <initializer_list>
#include <functional>
#include <cstdint>

namespace sovereigngrad {

enum class DType {
    Float32,
    Float16,
    Int32,
    Int64
};

inline std::string dtype_to_string(DType d) {
    switch (d) {
        case DType::Float32: return "Float32";
        case DType::Float16: return "Float16";
        case DType::Int32:   return "Int32";
        case DType::Int64:   return "Int64";
        default: return "Unknown";
    }
}

struct Node;
struct TensorImpl;
class Tensor;

struct Storage {
    void* ptr = nullptr;
    size_t size = 0; // size in bytes
    Device device = Device::CPU;

    Storage(size_t size, Device device);
    ~Storage();

    // Disable copying
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
};

struct TensorImpl {
    std::shared_ptr<Storage> storage;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t offset = 0; // offset in elements (floats)
    Device device = Device::CPU;
    DType dtype = DType::Float32;

    // Autograd properties
    std::shared_ptr<Node> creator;
    std::shared_ptr<TensorImpl> grad;
    bool requires_grad = false;
    uint64_t version = 0; // In-place modification tracker

    // Gradient hooks
    std::vector<std::function<Tensor(const Tensor&)>> hooks;

    TensorImpl(const std::vector<int64_t>& shape, Device device, bool requires_grad = false);
    TensorImpl(const std::vector<int64_t>& shape, Device device, DType dtype, bool requires_grad = false);
    TensorImpl(std::shared_ptr<Storage> storage, const std::vector<int64_t>& shape, 
               const std::vector<int64_t>& strides, int64_t offset, Device device, bool requires_grad = false);
    ~TensorImpl();

    float* data() { 
        if (!storage || !storage->ptr) return nullptr;
        return static_cast<float*>(storage->ptr) + offset; 
    }
    
    const float* data() const { 
        if (!storage || !storage->ptr) return nullptr;
        return static_cast<const float*>(storage->ptr) + offset; 
    }

    int64_t numel() const;
    bool is_contiguous() const;
    void bump_version() { version++; }
};

class Tensor {
public:
    std::shared_ptr<TensorImpl> impl;

    Tensor() : impl(nullptr) {}
    Tensor(std::shared_ptr<TensorImpl> impl) : impl(impl) {}
    Tensor(const std::vector<int64_t>& shape, Device device = Device::CPU, bool requires_grad = false);
    Tensor(const std::vector<int64_t>& shape, Device device, DType dtype, bool requires_grad = false);
    Tensor(float scalar, Device device = Device::CPU, bool requires_grad = false);
    Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape, Device device = Device::CPU, bool requires_grad = false);

    // Factory methods
    static Tensor zeros(const std::vector<int64_t>& shape, Device device = Device::CPU, bool requires_grad = false);
    static Tensor ones(const std::vector<int64_t>& shape, Device device = Device::CPU, bool requires_grad = false);
    static Tensor full(const std::vector<int64_t>& shape, float value, Device device = Device::CPU, bool requires_grad = false);
    static Tensor randn(const std::vector<int64_t>& shape, float mean = 0.0f, float std = 1.0f, Device device = Device::CPU, bool requires_grad = false);
    static Tensor uniform(const std::vector<int64_t>& shape, float low = 0.0f, float high = 1.0f, Device device = Device::CPU, bool requires_grad = false);
    static Tensor arange(float start, float end, float step = 1.0f, Device device = Device::CPU, bool requires_grad = false);
    static Tensor tril(const std::vector<int64_t>& shape, int diagonal = 0, Device device = Device::CPU);
    static Tensor triu(const std::vector<int64_t>& shape, int diagonal = 0, Device device = Device::CPU);

    // Metadata
    const std::vector<int64_t>& shape() const { return impl->shape; }
    const std::vector<int64_t>& strides() const { return impl->strides; }
    int64_t offset() const { return impl->offset; }
    Device device() const { return impl->device; }
    DType dtype() const { return impl ? impl->dtype : DType::Float32; }
    int64_t numel() const { return impl ? impl->numel() : 0; }
    bool is_contiguous() const { return impl ? impl->is_contiguous() : true; }
    uint64_t version() const { return impl ? impl->version : 0; }
    bool is_defined() const { return impl != nullptr; }
    bool defined() const { return impl != nullptr; }
    float* data() { return impl ? impl->data() : nullptr; }
    const float* data() const { return impl ? impl->data() : nullptr; }

    // Autograd controls
    bool requires_grad() const { return impl ? impl->requires_grad : false; }
    void set_requires_grad(bool req) { if (impl) impl->requires_grad = req; }
    Tensor grad() const;
    void set_grad(const Tensor& g);
    void zero_grad();
    std::shared_ptr<Node> creator() const { return impl ? impl->creator : nullptr; }
    void set_creator(std::shared_ptr<Node> node) { if (impl) impl->creator = node; }
    void register_hook(std::function<Tensor(const Tensor&)> hook);

    // Backward pass
    void backward(bool retain_graph = false);

    // Detach & Clone
    Tensor detach() const;
    Tensor clone() const;

    // Strided manipulations
    Tensor transpose(int dim0, int dim1) const;
    Tensor view(const std::vector<int64_t>& new_shape) const;
    Tensor reshape(const std::vector<int64_t>& new_shape) const;
    Tensor contiguous() const;
    Tensor slice(int dim, int64_t start, int64_t end, int64_t step = 1) const;
    Tensor select(int dim, int64_t index) const;
    Tensor narrow(int dim, int64_t start, int64_t length) const;
    Tensor index_select(int dim, const Tensor& index) const;

    // PCIe Device Transfer & DType Cast
    Tensor to(Device device) const;
    Tensor to(DType dtype) const;
    Tensor cpu() const { return to(Device::CPU); }
    Tensor cuda() const { return to(Device::CUDA); }

    // Forward Math Operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator+(float scalar) const { return *this + Tensor(scalar, device()); }
    Tensor operator-(float scalar) const { return *this - Tensor(scalar, device()); }
    Tensor operator*(float scalar) const { return *this * Tensor(scalar, device()); }
    Tensor operator/(float scalar) const { return *this / Tensor(scalar, device()); }
    Tensor operator-() const; // Unary negation

    // In-place operations (bumps version!)
    Tensor& add_(const Tensor& other);
    Tensor& mul_(float scalar);

    // Core Neural Ops
    Tensor matmul(const Tensor& other) const;
    Tensor relu() const;
    Tensor gelu() const;
    Tensor silu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor sqrt() const;
    Tensor pow(float exponent) const;
    Tensor clamp(float min_val, float max_val) const;
    Tensor dropout(float p, bool training = true) const;
    Tensor softmax(int dim = -1) const;
    Tensor flatten(int start_dim = 0, int end_dim = -1) const;
    Tensor max_pool2d(int kernel_size = 2, int stride = 2, int padding = 0) const;
    Tensor avg_pool2d(int kernel_size = 2, int stride = 2, int padding = 0) const;
    Tensor masked_fill(const Tensor& mask, float value) const;
    Tensor rmsnorm(const Tensor& gamma, float eps = 1e-5f) const;

    // Reductions
    Tensor sum(int dim, bool keepdim = false) const;
    Tensor sum() const;
    Tensor mean(int dim, bool keepdim = false) const;
    Tensor mean() const;
    Tensor max(int dim, bool keepdim = false) const;
    float norm(float p = 2.0f) const;

    // Scalar Extraction
    float item() const;
    
    // Element Access
    float get_element(const std::vector<int64_t>& indices) const;
    void set_element(const std::vector<int64_t>& indices, float value);

    // Formatting & printing
    void print() const;
    std::string summary() const;
};

// Utilities for shapes
std::vector<int64_t> default_strides(const std::vector<int64_t>& shape);
int64_t compute_numel(const std::vector<int64_t>& shape);

} // namespace sovereigngrad

#pragma once

#include "soverigngrad/allocator.hpp"
#include <vector>
#include <memory>
#include <string>
#include <initializer_list>

namespace sovereigngrad {

struct Node;
struct TensorImpl;

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

    // Autograd properties
    std::shared_ptr<Node> creator;
    std::shared_ptr<TensorImpl> grad;
    bool requires_grad = false;

    TensorImpl(const std::vector<int64_t>& shape, Device device, bool requires_grad = false);
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
};

class Tensor {
public:
    std::shared_ptr<TensorImpl> impl;

    Tensor() : impl(nullptr) {}
    Tensor(std::shared_ptr<TensorImpl> impl) : impl(impl) {}
    Tensor(const std::vector<int64_t>& shape, Device device = Device::CPU, bool requires_grad = false);
    Tensor(float scalar, Device device = Device::CPU, bool requires_grad = false);
    Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape, Device device = Device::CPU, bool requires_grad = false);

    // Factory methods
    static Tensor zeros(const std::vector<int64_t>& shape, Device device = Device::CPU, bool requires_grad = false);
    static Tensor ones(const std::vector<int64_t>& shape, Device device = Device::CPU, bool requires_grad = false);
    static Tensor randn(const std::vector<int64_t>& shape, float mean = 0.0f, float std = 1.0f, Device device = Device::CPU, bool requires_grad = false);

    // Shape, strides, size
    const std::vector<int64_t>& shape() const { return impl->shape; }
    const std::vector<int64_t>& strides() const { return impl->strides; }
    int64_t offset() const { return impl->offset; }
    Device device() const { return impl->device; }
    int64_t numel() const { return impl->numel(); }
    bool is_contiguous() const { return impl->is_contiguous(); }

    // Autograd controls
    bool requires_grad() const { return impl->requires_grad; }
    void set_requires_grad(bool req) { impl->requires_grad = req; }
    Tensor grad() const;
    void set_grad(const Tensor& g);
    void zero_grad();
    std::shared_ptr<Node> creator() const { return impl->creator; }
    void set_creator(std::shared_ptr<Node> node) { impl->creator = node; }

    // Backward pass
    void backward();

    // Strided manipulations
    Tensor transpose(int dim0, int dim1) const;
    Tensor view(const std::vector<int64_t>& new_shape) const;
    Tensor contiguous() const;

    // PCIe Device Transfer
    Tensor to(Device device) const;

    // Basic operators (Forward passes implemented in ops.hpp/ops.cpp)
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor relu() const;
    Tensor matmul(const Tensor& other) const;

    // Helper operations
    Tensor sum(int dim, bool keepdim = false) const;
    Tensor sum() const; // sum all elements

    // Get scalar value (only for 1-element tensors)
    float item() const;
    
    // Set/Get elements on CPU
    float get_element(const std::vector<int64_t>& indices) const;
    void set_element(const std::vector<int64_t>& indices, float value);

    // Formatting & printing
    void print() const;
};

// Utilities for shapes
std::vector<int64_t> default_strides(const std::vector<int64_t>& shape);
int64_t compute_numel(const std::vector<int64_t>& shape);

} // namespace sovereigngrad

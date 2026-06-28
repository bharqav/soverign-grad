#include "soverigngrad/tensor.hpp"
#include "soverigngrad/autograd.hpp"
#include "soverigngrad/cuda_mock.hpp"
#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>
#include <sstream>

namespace sovereigngrad {

// Helper to compute default strides
std::vector<int64_t> default_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

int64_t compute_numel(const std::vector<int64_t>& shape) {
    int64_t num = 1;
    for (auto d : shape) num *= d;
    return num;
}

// Storage Implementation
Storage::Storage(size_t size, Device device) : size(size), device(device) {
    ptr = MemoryAllocator::instance().allocate(size, device);
}

Storage::~Storage() {
    MemoryAllocator::instance().free(ptr, device);
}

TensorImpl::~TensorImpl() {}

// TensorImpl Implementation
TensorImpl::TensorImpl(const std::vector<int64_t>& shape, Device device, bool requires_grad)
    : shape(shape), strides(default_strides(shape)), offset(0), device(device), requires_grad(requires_grad) {
    size_t bytes = compute_numel(shape) * sizeof(float);
    storage = std::make_shared<Storage>(bytes, device);
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage, const std::vector<int64_t>& shape, 
                       const std::vector<int64_t>& strides, int64_t offset, Device device, bool requires_grad)
    : storage(storage), shape(shape), strides(strides), offset(offset), device(device), requires_grad(requires_grad) {}

int64_t TensorImpl::numel() const {
    return compute_numel(shape);
}

bool TensorImpl::is_contiguous() const {
    if (shape.empty()) return true;
    auto expected = default_strides(shape);
    return strides == expected;
}

// Tensor Implementation
Tensor::Tensor(const std::vector<int64_t>& shape, Device device, bool requires_grad) {
    impl = std::make_shared<TensorImpl>(shape, device, requires_grad);
}

Tensor::Tensor(float scalar, Device device, bool requires_grad) {
    impl = std::make_shared<TensorImpl>(std::vector<int64_t>{1}, device, requires_grad);
    set_element({0}, scalar);
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape, Device device, bool requires_grad) {
    int64_t total = compute_numel(shape);
    if (data.size() != static_cast<size_t>(total)) {
        throw std::runtime_error("Data size does not match shape!");
    }
    impl = std::make_shared<TensorImpl>(shape, Device::CPU, requires_grad);
    std::memcpy(impl->data(), data.data(), total * sizeof(float));
    if (device == Device::CUDA) {
        *this = this->to(Device::CUDA);
    }
}

// Factories
Tensor Tensor::zeros(const std::vector<int64_t>& shape, Device device, bool requires_grad) {
    Tensor t(shape, device, requires_grad);
    size_t bytes = t.numel() * sizeof(float);
    if (device == Device::CPU) {
        std::memset(t.impl->data(), 0, bytes);
    } else {
        cudaMemset(t.impl->data(), 0, bytes);
    }
    return t;
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, Device device, bool requires_grad) {
    Tensor t(shape, device, requires_grad);
    int64_t total = t.numel();
    if (device == Device::CPU) {
        float* data = t.impl->data();
        for (int64_t i = 0; i < total; ++i) data[i] = 1.0f;
    } else {
        // Copy CPU ones to GPU
        std::vector<float> cpu_ones(total, 1.0f);
        cudaMemcpy(t.impl->data(), cpu_ones.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    }
    return t;
}

Tensor Tensor::randn(const std::vector<int64_t>& shape, float mean, float std, Device device, bool requires_grad) {
    Tensor t(shape, Device::CPU, requires_grad);
    int64_t total = t.numel();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    
    float* data = t.impl->data();
    for (int64_t i = 0; i < total; ++i) {
        data[i] = dist(gen);
    }

    if (device == Device::CUDA) {
        t = t.to(Device::CUDA);
    }
    return t;
}

// Autograd Gradients
Tensor Tensor::grad() const {
    if (!impl->grad) return Tensor();
    return Tensor(impl->grad);
}

void Tensor::set_grad(const Tensor& g) {
    if (!g.impl) {
        impl->grad = nullptr;
    } else {
        impl->grad = g.impl;
    }
}

void Tensor::zero_grad() {
    if (impl->grad) {
        size_t bytes = numel() * sizeof(float);
        if (device() == Device::CPU) {
            std::memset(impl->grad->data(), 0, bytes);
        } else {
            cudaMemset(impl->grad->data(), 0, bytes);
        }
    }
}

// PCIe transfers
Tensor Tensor::to(Device target_device) const {
    if (device() == target_device) {
        return *this;
    }

    // Force to contiguous layout first to simplify transfers
    Tensor contig = this->contiguous();
    Tensor out(shape(), target_device, requires_grad());

    size_t bytes = numel() * sizeof(float);
    if (device() == Device::CPU && target_device == Device::CUDA) {
        cudaMemcpy(out.impl->data(), contig.impl->data(), bytes, cudaMemcpyHostToDevice);
    } else if (device() == Device::CUDA && target_device == Device::CPU) {
        cudaMemcpy(out.impl->data(), contig.impl->data(), bytes, cudaMemcpyDeviceToHost);
    }

    return out;
}

// Strided views
Tensor Tensor::transpose(int dim0, int dim1) const {
    int dims = (int)shape().size();
    if (dim0 < 0 || dim0 >= dims || dim1 < 0 || dim1 >= dims) {
        throw std::runtime_error("Invalid dimensions for transpose!");
    }

    // Swap shapes and strides for dimensions dim0 and dim1
    auto new_shape = shape();
    auto new_strides = strides();
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    // O(1) operation: shares the exact same storage but with new shape/strides!
    return Tensor(std::make_shared<TensorImpl>(
        impl->storage, new_shape, new_strides, impl->offset, device(), requires_grad()
    ));
}

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    int64_t old_numel = numel();
    int64_t new_numel = 1;
    std::vector<int64_t> inferred_shape = new_shape;
    int inferred_idx = -1;

    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (inferred_idx != -1) {
                throw std::runtime_error("Only one dimension can be inferred (-1)!");
            }
            inferred_idx = (int)i;
        } else {
            new_numel *= new_shape[i];
        }
    }

    if (inferred_idx != -1) {
        if (old_numel % new_numel != 0) {
            throw std::runtime_error("Cannot reshape: size is not divisible by dimensions!");
        }
        inferred_shape[inferred_idx] = old_numel / new_numel;
        new_numel = old_numel;
    }

    if (old_numel != new_numel) {
        throw std::runtime_error("Reshape changes the number of elements!");
    }

    // If contiguous, we can reshape in O(1) time
    if (is_contiguous()) {
        auto new_strides = default_strides(inferred_shape);
        return Tensor(std::make_shared<TensorImpl>(
            impl->storage, inferred_shape, new_strides, impl->offset, device(), requires_grad()
        ));
    } else {
        // If not contiguous, make it contiguous first
        return this->contiguous().view(inferred_shape);
    }
}

// Strided copier helper
static void recursive_strided_copy(const float* src, float* dst, 
                                   const std::vector<int64_t>& shape, 
                                   const std::vector<int64_t>& strides, 
                                   int64_t src_offset, 
                                   int dim, 
                                   int64_t& dst_idx) {
    if (dim == (int)shape.size()) {
        dst[dst_idx++] = src[src_offset];
        return;
    }
    for (int64_t i = 0; i < shape[dim]; ++i) {
        recursive_strided_copy(src, dst, shape, strides, src_offset + i * strides[dim], dim + 1, dst_idx);
    }
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }

    Tensor out(shape(), device(), requires_grad());
    if (device() == Device::CPU) {
        int64_t dst_idx = 0;
        recursive_strided_copy(impl->data(), out.impl->data(), shape(), strides(), 0, 0, dst_idx);
    } else {
        // Safe GPU fallback for non-contiguous:
        // 1. Create a CPU storage matching the GPU storage size
        auto cpu_storage = std::make_shared<Storage>(impl->storage->size, Device::CPU);
        // 2. Copy the entire GPU memory block to CPU
        cudaMemcpy(cpu_storage->ptr, impl->storage->ptr, impl->storage->size, cudaMemcpyDeviceToHost);
        // 3. Create a CPU tensor view with the same shape, strides, and offset
        Tensor cpu_view(std::make_shared<TensorImpl>(
            cpu_storage, shape(), strides(), impl->offset, Device::CPU, requires_grad()
        ));
        // 4. Make it contiguous on the CPU (uses CPU recursive copy)
        Tensor cpu_contig = cpu_view.contiguous();
        // 5. Copy the contiguous CPU tensor back to CUDA
        out = cpu_contig.to(Device::CUDA);
    }
    return out;
}

// Element access (CPU only)
float Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item() can only be called on 1-element tensors!");
    }
    if (device() == Device::CUDA) {
        return this->to(Device::CPU).item();
    }
    return impl->data()[0];
}

float Tensor::get_element(const std::vector<int64_t>& indices) const {
    if (device() == Device::CUDA) {
        return this->to(Device::CPU).get_element(indices);
    }
    if (indices.size() != shape().size()) {
        throw std::runtime_error("Indices dimension mismatch!");
    }
    int64_t idx = impl->offset;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape()[i]) {
            throw std::runtime_error("Index out of bounds!");
        }
        idx += indices[i] * strides()[i];
    }
    return static_cast<float*>(impl->storage->ptr)[idx];
}

void Tensor::set_element(const std::vector<int64_t>& indices, float value) {
    if (device() == Device::CUDA) {
        Tensor cpu = this->to(Device::CPU);
        cpu.set_element(indices, value);
        cudaMemcpy(impl->data(), cpu.impl->data(), numel() * sizeof(float), cudaMemcpyHostToDevice);
        return;
    }
    if (indices.size() != shape().size()) {
        throw std::runtime_error("Indices dimension mismatch!");
    }
    int64_t idx = impl->offset;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape()[i]) {
            throw std::runtime_error("Index out of bounds!");
        }
        idx += indices[i] * strides()[i];
    }
    static_cast<float*>(impl->storage->ptr)[idx] = value;
}

// Autograd backward call
void Tensor::backward() {
    AutogradEngine::backward(*this);
}

// Formatting
void Tensor::print() const {
    if (device() == Device::CUDA) {
        this->to(Device::CPU).print();
        return;
    }
    
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape().size(); ++i) {
        std::cout << shape()[i] << (i + 1 < shape().size() ? ", " : "");
    }
    std::cout << "], strides=[";
    for (size_t i = 0; i < strides().size(); ++i) {
        std::cout << strides()[i] << (i + 1 < strides().size() ? ", " : "");
    }
    std::cout << "], device=" << (device() == Device::CPU ? "CPU" : "CUDA") 
              << ", requires_grad=" << (requires_grad() ? "true" : "false") << ")\n";

    if (numel() > 100) {
        // Summarized view
        std::cout << "[";
        for (int i = 0; i < std::min(numel(), 5LL); ++i) {
            std::cout << impl->data()[i] << ", ";
        }
        std::cout << "... ";
        for (int i = std::max(0LL, numel() - 5); i < numel(); ++i) {
            std::cout << impl->data()[i] << (i + 1 < numel() ? ", " : "");
        }
        std::cout << "]\n";
        return;
    }

    // Full multi-dimensional print (simplified for 1D and 2D)
    if (shape().size() == 1) {
        std::cout << "[";
        for (int64_t i = 0; i < shape()[0]; ++i) {
            std::cout << get_element({i}) << (i + 1 < shape()[0] ? ", " : "");
        }
        std::cout << "]\n";
    } else if (shape().size() == 2) {
        std::cout << "[\n";
        for (int64_t r = 0; r < shape()[0]; ++r) {
            std::cout << "  [";
            for (int64_t c = 0; c < shape()[1]; ++c) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) 
                          << get_element({r, c}) << (c + 1 < shape()[1] ? ", " : "");
            }
            std::cout << "]" << (r + 1 < shape()[0] ? ",\n" : "\n");
        }
        std::cout << "]\n";
    } else {
        // Flattened representation for higher dimensions
        std::cout << "[flat: ";
        for (int64_t i = 0; i < numel(); ++i) {
            std::cout << impl->data()[i] << (i + 1 < numel() ? ", " : "");
        }
        std::cout << "]\n";
    }
}

} // namespace sovereigngrad

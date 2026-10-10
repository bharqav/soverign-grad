#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/autograd.hpp"
#include "sovereigngrad/cuda_mock.hpp"
#include "sovereigngrad/ops.hpp"
#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <cstring>
#include <stdexcept>

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
    if (ptr && size > 0) {
        if (device == Device::CPU) {
            std::memset(ptr, 0, size);
        } else {
            CUDA_CHECK(cudaMemset(ptr, 0, size));
        }
    }
}

Storage::~Storage() {
    MemoryAllocator::instance().free(ptr, device);
}

TensorImpl::~TensorImpl() {}

// TensorImpl Implementation
TensorImpl::TensorImpl(const std::vector<int64_t>& shape, Device device, bool requires_grad)
    : shape(shape), strides(default_strides(shape)), offset(0), device(device), 
      dtype(DType::Float32), requires_grad(requires_grad), version(0) {
    size_t bytes = compute_numel(shape) * sizeof(float);
    storage = std::make_shared<Storage>(bytes, device);
}

TensorImpl::TensorImpl(const std::vector<int64_t>& shape, Device device, DType dtype, bool requires_grad)
    : shape(shape), strides(default_strides(shape)), offset(0), device(device), 
      dtype(dtype), requires_grad(requires_grad), version(0) {
    size_t elem_size = (dtype == DType::Float16) ? 2 : (dtype == DType::Int64 ? 8 : 4);
    size_t bytes = compute_numel(shape) * elem_size;
    storage = std::make_shared<Storage>(bytes, device);
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage, const std::vector<int64_t>& shape, 
                       const std::vector<int64_t>& strides, int64_t offset, Device device, bool requires_grad)
    : storage(storage), shape(shape), strides(strides), offset(offset), device(device), 
      dtype(DType::Float32), requires_grad(requires_grad), version(0) {}

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

Tensor::Tensor(const std::vector<int64_t>& shape, Device device, DType dtype, bool requires_grad) {
    impl = std::make_shared<TensorImpl>(shape, device, dtype, requires_grad);
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
        CUDA_CHECK(cudaMemset(t.impl->data(), 0, bytes));
    }
    return t;
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, Device device, bool requires_grad) {
    return full(shape, 1.0f, device, requires_grad);
}

Tensor Tensor::full(const std::vector<int64_t>& shape, float value, Device device, bool requires_grad) {
    Tensor t(shape, Device::CPU, requires_grad);
    int64_t total = t.numel();
    float* data = t.impl->data();
    for (int64_t i = 0; i < total; ++i) {
        data[i] = value;
    }
    if (device == Device::CUDA) {
        return t.to(Device::CUDA);
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
        return t.to(Device::CUDA);
    }
    return t;
}

Tensor Tensor::uniform(const std::vector<int64_t>& shape, float low, float high, Device device, bool requires_grad) {
    Tensor t(shape, Device::CPU, requires_grad);
    int64_t total = t.numel();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);
    
    float* data = t.impl->data();
    for (int64_t i = 0; i < total; ++i) {
        data[i] = dist(gen);
    }

    if (device == Device::CUDA) {
        return t.to(Device::CUDA);
    }
    return t;
}

Tensor Tensor::arange(float start, float end, float step, Device device, bool requires_grad) {
    if (step == 0.0f) throw std::runtime_error("arange step cannot be zero!");
    int64_t count = static_cast<int64_t>(std::ceil((end - start) / step));
    if (count <= 0) count = 0;
    std::vector<float> data(count);
    for (int64_t i = 0; i < count; ++i) {
        data[i] = start + static_cast<float>(i) * step;
    }
    return Tensor(data, {count}, device, requires_grad);
}

Tensor Tensor::tril(const std::vector<int64_t>& shape, int diagonal, Device device) {
    if (shape.size() < 2) throw std::runtime_error("tril requires at least 2D shape!");
    int64_t rows = shape[shape.size() - 2];
    int64_t cols = shape[shape.size() - 1];
    int64_t total = compute_numel(shape);
    Tensor res(shape, Device::CPU);
    float* d = res.impl->data();
    int64_t matrix_size = rows * cols;
    int64_t num_matrices = total / matrix_size;

    for (int64_t m = 0; m < num_matrices; ++m) {
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                d[m * matrix_size + r * cols + c] = (c <= r + diagonal) ? 1.0f : 0.0f;
            }
        }
    }
    return (device == Device::CUDA) ? res.to(Device::CUDA) : res;
}

Tensor Tensor::triu(const std::vector<int64_t>& shape, int diagonal, Device device) {
    if (shape.size() < 2) throw std::runtime_error("triu requires at least 2D shape!");
    int64_t rows = shape[shape.size() - 2];
    int64_t cols = shape[shape.size() - 1];
    int64_t total = compute_numel(shape);
    Tensor res(shape, Device::CPU);
    float* d = res.impl->data();
    int64_t matrix_size = rows * cols;
    int64_t num_matrices = total / matrix_size;

    for (int64_t m = 0; m < num_matrices; ++m) {
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                d[m * matrix_size + r * cols + c] = (c >= r + diagonal) ? 1.0f : 0.0f;
            }
        }
    }
    return (device == Device::CUDA) ? res.to(Device::CUDA) : res;
}

float Tensor::norm(float p) const {
    if (!impl || numel() == 0) return 0.0f;
    Tensor cpu_t = this->to(Device::CPU);
    const float* d = cpu_t.impl->data();
    int64_t n = cpu_t.numel();
    if (p == 2.0f) {
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < n; ++i) sum_sq += d[i] * d[i];
        return std::sqrt(sum_sq);
    }
    float sum_p = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum_p += std::pow(std::abs(d[i]), p);
    return std::pow(sum_p, 1.0f / p);
}

// Autograd Gradients
Tensor Tensor::grad() const {
    if (!impl || !impl->grad) return Tensor();
    return Tensor(impl->grad);
}

void Tensor::set_grad(const Tensor& g) {
    if (!impl) return;
    if (!g.impl) {
        impl->grad = nullptr;
    } else {
        impl->grad = g.impl;
    }
}

void Tensor::zero_grad() {
    if (impl && impl->grad) {
        size_t bytes = numel() * sizeof(float);
        if (device() == Device::CPU) {
            std::memset(impl->grad->data(), 0, bytes);
        } else {
            CUDA_CHECK(cudaMemset(impl->grad->data(), 0, bytes));
        }
    }
}

void Tensor::register_hook(std::function<Tensor(const Tensor&)> hook) {
    if (impl) {
        impl->hooks.push_back(hook);
    }
}

// Detach & Clone
Tensor Tensor::detach() const {
    if (!impl) return Tensor();
    auto new_impl = std::make_shared<TensorImpl>(
        impl->storage, impl->shape, impl->strides, impl->offset, impl->device, /*requires_grad=*/false
    );
    new_impl->version = impl->version;
    return Tensor(new_impl);
}

Tensor Tensor::clone() const {
    if (!impl) return Tensor();
    Tensor contig = this->contiguous();
    Tensor out(contig.shape(), contig.device(), contig.requires_grad());
    size_t bytes = contig.numel() * sizeof(float);
    if (contig.device() == Device::CPU) {
        std::memcpy(out.impl->data(), contig.impl->data(), bytes);
    } else {
        CUDA_CHECK(cudaMemcpy(out.impl->data(), contig.impl->data(), bytes, cudaMemcpyDeviceToDevice));
    }
    return out;
}

// -------------------------------------------------------------
// STRUCTURAL AUTOGRAD NODES (To, Transpose, View, Contiguous)
// -------------------------------------------------------------
struct ToNode : public Node {
    Device orig_dev;
    ToNode(const Tensor& in, Device orig_dev) : Node("To", {in}), orig_dev(orig_dev) {}
    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], grad_output.to(orig_dev));
    }
};

struct TransposeNode : public Node {
    int dim0, dim1;
    TransposeNode(const Tensor& in, int dim0, int dim1) : Node("Transpose", {in}), dim0(dim0), dim1(dim1) {}
    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], grad_output.transpose(dim0, dim1));
    }
};

struct ViewNode : public Node {
    std::vector<int64_t> in_shape;
    ViewNode(const Tensor& in, const std::vector<int64_t>& in_shape) : Node("View", {in}), in_shape(in_shape) {}
    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], grad_output.contiguous().view(in_shape));
    }
};

struct ContiguousNode : public Node {
    ContiguousNode(const Tensor& in) : Node("Contiguous", {in}) {}
    void backward(const Tensor& grad_output) override {
        accumulate_grad(inputs[0], grad_output);
    }
};

// PCIe transfers
Tensor Tensor::to(Device target_device) const {
    if (!impl || device() == target_device) {
        return *this;
    }

    Tensor contig = this->contiguous();
    Tensor out(shape(), target_device, requires_grad() && AutoGradMode::is_enabled());

    size_t bytes = numel() * sizeof(float);
    if (device() == Device::CPU && target_device == Device::CUDA) {
        CUDA_CHECK(cudaMemcpy(out.impl->data(), contig.impl->data(), bytes, cudaMemcpyHostToDevice));
    } else if (device() == Device::CUDA && target_device == Device::CPU) {
        CUDA_CHECK(cudaMemcpy(out.impl->data(), contig.impl->data(), bytes, cudaMemcpyDeviceToHost));
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<ToNode>(*this, device()));
    }

    return out;
}

Tensor Tensor::to(DType target_dtype) const {
    if (!impl) throw std::runtime_error("Tensor is undefined!");
    if (dtype() == target_dtype) return *this;
    Tensor out = this->clone();
    out.impl->dtype = target_dtype;
    return out;
}

// Strided views
Tensor Tensor::transpose(int dim0, int dim1) const {
    if (!impl) throw std::runtime_error("Tensor is undefined!");
    int dims = (int)shape().size();
    if (dim0 < 0 || dim0 >= dims || dim1 < 0 || dim1 >= dims) {
        throw std::runtime_error("Invalid dimensions for transpose!");
    }

    auto new_shape = shape();
    auto new_strides = strides();
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    auto res = Tensor(std::make_shared<TensorImpl>(
        impl->storage, new_shape, new_strides, impl->offset, device(), requires_grad() && AutoGradMode::is_enabled()
    ));
    res.impl->version = impl->version;

    if (AutoGradMode::is_enabled() && res.requires_grad()) {
        res.set_creator(std::make_shared<TransposeNode>(*this, dim0, dim1));
    }

    return res;
}

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    if (!impl) throw std::runtime_error("Tensor is undefined!");
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

    Tensor res;
    if (is_contiguous()) {
        auto new_strides = default_strides(inferred_shape);
        res = Tensor(std::make_shared<TensorImpl>(
            impl->storage, inferred_shape, new_strides, impl->offset, device(), requires_grad() && AutoGradMode::is_enabled()
        ));
        res.impl->version = impl->version;
    } else {
        res = this->contiguous().view(inferred_shape);
    }

    if (AutoGradMode::is_enabled() && res.requires_grad()) {
        res.set_creator(std::make_shared<ViewNode>(*this, shape()));
    }

    return res;
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    return view(new_shape);
}

Tensor Tensor::flatten(int start_dim, int end_dim) const {
    int dims = (int)shape().size();
    if (dims == 0) return view({1});
    if (start_dim < 0) start_dim += dims;
    if (end_dim < 0) end_dim += dims;
    if (start_dim < 0 || start_dim >= dims || end_dim < 0 || end_dim >= dims || start_dim > end_dim) {
        throw std::runtime_error("Invalid dimensions for flatten!");
    }
    std::vector<int64_t> new_shape;
    for (int i = 0; i < start_dim; ++i) new_shape.push_back(shape()[i]);
    int64_t flat_size = 1;
    for (int i = start_dim; i <= end_dim; ++i) flat_size *= shape()[i];
    new_shape.push_back(flat_size);
    for (int i = end_dim + 1; i < dims; ++i) new_shape.push_back(shape()[i]);
    return view(new_shape);
}

Tensor Tensor::slice(int dim, int64_t start, int64_t end, int64_t step) const {
    return slice_op(*this, dim, start, end, step);
}

Tensor Tensor::select(int dim, int64_t index) const {
    return select_op(*this, dim, index);
}

Tensor Tensor::narrow(int dim, int64_t start, int64_t length) const {
    return slice_op(*this, dim, start, start + length, 1);
}

Tensor Tensor::index_select(int dim, const Tensor& index) const {
    return index_select_op(*this, dim, index);
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
    if (!impl || is_contiguous()) {
        return *this;
    }

    Tensor out(shape(), device(), requires_grad() && AutoGradMode::is_enabled());
    if (device() == Device::CPU) {
        int64_t dst_idx = 0;
        recursive_strided_copy(impl->data(), out.impl->data(), shape(), strides(), 0, 0, dst_idx);
    } else {
        auto cpu_storage = std::make_shared<Storage>(impl->storage->size, Device::CPU);
        CUDA_CHECK(cudaMemcpy(cpu_storage->ptr, impl->storage->ptr, impl->storage->size, cudaMemcpyDeviceToHost));
        Tensor cpu_view(std::make_shared<TensorImpl>(
            cpu_storage, shape(), strides(), impl->offset, Device::CPU, requires_grad() && AutoGradMode::is_enabled()
        ));
        Tensor cpu_contig = cpu_view.contiguous();
        out = cpu_contig.to(Device::CUDA);
    }

    if (AutoGradMode::is_enabled() && out.requires_grad()) {
        out.set_creator(std::make_shared<ContiguousNode>(*this));
    }

    return out;
}

// In-place operations (bumps version!)
Tensor& Tensor::add_(const Tensor& other) {
    if (!impl) throw std::runtime_error("Tensor is undefined!");
    if (numel() != other.numel()) {
        throw std::runtime_error("In-place add_ requires matching element count!");
    }
    if (device() != other.device()) {
        throw std::runtime_error("Device mismatch in add_!");
    }
    
    int64_t total = numel();
    if (device() == Device::CPU) {
        float* d1 = impl->data();
        const float* d2 = other.impl->data();
        for (int64_t i = 0; i < total; ++i) d1[i] += d2[i];
    } else {
        Tensor sum_res = *this + other;
        CUDA_CHECK(cudaMemcpy(impl->data(), sum_res.impl->data(), total * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    impl->bump_version();
    return *this;
}

Tensor& Tensor::mul_(float scalar) {
    if (!impl) throw std::runtime_error("Tensor is undefined!");
    int64_t total = numel();
    if (device() == Device::CPU) {
        float* d = impl->data();
        for (int64_t i = 0; i < total; ++i) d[i] *= scalar;
    } else {
        Tensor cpu = this->to(Device::CPU);
        cpu.mul_(scalar);
        CUDA_CHECK(cudaMemcpy(impl->data(), cpu.impl->data(), total * sizeof(float), cudaMemcpyHostToDevice));
    }
    impl->bump_version();
    return *this;
}

// Element access
float Tensor::item() const {
    if (!impl || numel() != 1) {
        throw std::runtime_error("item() can only be called on 1-element tensors!");
    }
    if (device() == Device::CUDA) {
        return this->to(Device::CPU).item();
    }
    return impl->data()[0];
}

float Tensor::get_element(const std::vector<int64_t>& indices) const {
    if (!impl) throw std::runtime_error("Tensor is undefined!");
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
    if (!impl) throw std::runtime_error("Tensor is undefined!");
    if (device() == Device::CUDA) {
        Tensor cpu = this->to(Device::CPU);
        cpu.set_element(indices, value);
        CUDA_CHECK(cudaMemcpy(impl->data(), cpu.impl->data(), numel() * sizeof(float), cudaMemcpyHostToDevice));
        impl->bump_version();
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
    impl->bump_version();
}

// Autograd backward call
void Tensor::backward(bool retain_graph) {
    AutogradEngine::backward(*this, retain_graph);
}

std::string Tensor::summary() const {
    if (!impl) return "Tensor(undefined)";
    std::ostringstream ss;
    ss << "Tensor(shape=[";
    for (size_t i = 0; i < shape().size(); ++i) {
        ss << shape()[i] << (i + 1 < shape().size() ? ", " : "");
    }
    ss << "], dtype=" << dtype_to_string(dtype())
       << ", device=" << device_to_string(device())
       << ", requires_grad=" << (requires_grad() ? "true" : "false")
       << ", version=" << version() << ")";
    return ss.str();
}

// Formatting
void Tensor::print() const {
    if (!impl) {
        std::cout << "Tensor(undefined)\n";
        return;
    }
    if (device() == Device::CUDA) {
        this->to(Device::CPU).print();
        return;
    }
    
    std::cout << summary() << "\n";

    if (numel() > 100) {
        std::cout << "[";
        for (int64_t i = 0; i < std::min<int64_t>(numel(), 5); ++i) {
            std::cout << impl->data()[i] << ", ";
        }
        std::cout << "... ";
        for (int64_t i = std::max<int64_t>(0, numel() - 5); i < numel(); ++i) {
            std::cout << impl->data()[i] << (i + 1 < numel() ? ", " : "");
        }
        std::cout << "]\n";
        return;
    }

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
        std::cout << "[flat: ";
        for (int64_t i = 0; i < numel(); ++i) {
            std::cout << impl->data()[i] << (i + 1 < numel() ? ", " : "");
        }
        std::cout << "]\n";
    }
}

} // namespace sovereigngrad

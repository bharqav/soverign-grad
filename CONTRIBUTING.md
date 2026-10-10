# Contributing to SovereignGrad

Thank you for your interest in contributing to **SovereignGrad**! We welcome contributions from developers, machine learning researchers, and systems engineers.

This document outlines our architectural patterns, coding standards, step-by-step instructions for adding new operators and layers, and testing protocols.

---

## Table of Contents
1. [Development Prerequisites & Setup](#1-development-prerequisites--setup)
2. [How to Add a New Operator](#2-how-to-add-a-new-operator)
3. [How to Add a New Neural Network Layer](#3-how-to-add-a-new-neural-network-layer)
4. [Testing & Numerical Gradchecks](#4-testing--numerical-gradchecks)
5. [Code Style & Engineering Standards](#5-code-style--engineering-standards)
6. [Pull Request & Submission Workflow](#6-pull-request--submission-workflow)

---

## 1. Development Prerequisites & Setup

### Prerequisites
- **C++ Compiler**: Modern C++17 compliant compiler (`MSVC 2019+`, `GCC 9+`, or `Clang 10+`)
- **Build System**: `CMake 3.15+` and optionally `Ninja`
- **Optional GPU Acceleration**: `NVIDIA CUDA Toolkit 11.0+` (if compiling with physical GPU hardware acceleration)
- **Python (Optional for Benchmarks)**: `Python 3.8+` with `torch`

### Clone & Build
```bash
# Clone the repository
git clone https://github.com/your-username/SovereignGrad.git
cd SovereignGrad

# Configure build directory (CPU + CUDA Emulation mode)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF

# Build all targets
cmake --build build --config Release

# Run test suite
./build/sovereign_grad_test
```

---

## 2. How to Add a New Operator

Adding a new mathematical or neural operator requires 6 structured steps:

```mermaid
flowchart LR
    A["1. Declare op in ops.hpp"] --> B["2. CPU forward in ops.cpp"]
    B --> C["3. CUDA kernel in cuda_kernels.cu"]
    C --> D["4. Backward Node in ops.cpp"]
    D --> E["5. Wrap in Tensor class"]
    E --> F["6. Register Gradcheck & Test"]
```

### Step 1: Declare Forward Function in `include/sovereigngrad/ops.hpp`
Add your function signature:
```cpp
// Example: Hardswish activation
Tensor hardswish(const Tensor& input);
```

### Step 2: Implement CPU Forward in `src/ops.cpp`
Implement the mathematical logic:
```cpp
Tensor hardswish(const Tensor& input) {
    Tensor out(input.shape(), input.device(), input.requires_grad());
    float* out_data = out.impl->data();
    const float* in_data = input.impl->data();
    int64_t n = input.numel();

    if (input.device() == Device::CPU) {
        for (int64_t i = 0; i < n; ++i) {
            float x = in_data[i];
            if (x <= -3.0f) out_data[i] = 0.0f;
            else if (x >= 3.0f) out_data[i] = x;
            else out_data[i] = x * (x + 3.0f) / 6.0f;
        }
    } else {
        // Dispatched to CUDA kernel
        cuda::launch_hardswish_forward(in_data, out_data, (int)n);
    }

    if (AutoGradMode::is_enabled() && input.requires_grad()) {
        out.set_creator(std::make_shared<HardswishBackwardNode>(input));
    }
    return out;
}
```

### Step 3: Implement CUDA Kernel in `src/cuda_kernels.cu` & `include/sovereigngrad/cuda_kernels.cuh`
Declare the kernel in `cuda_kernels.cuh`:
```cpp
namespace sovereigngrad {
namespace cuda {
    void launch_hardswish_forward(const float* in, float* out, int n);
    void launch_hardswish_backward(const float* in, const float* grad_out, float* grad_in, int n);
}
}
```

### Step 4: Implement Autograd Backward Node in `src/ops.cpp`
```cpp
struct HardswishBackwardNode : public Node {
    Tensor in_saved;
    HardswishBackwardNode(const Tensor& in) : Node("Hardswish", {in}), in_saved(in.detach()) {}

    void backward(const Tensor& grad_output) override {
        Tensor d_in(in_saved.shape(), in_saved.device());
        int64_t n = in_saved.numel();
        if (in_saved.device() == Device::CPU) {
            const float* in_data = in_saved.impl->data();
            const float* dout = grad_output.impl->data();
            float* din = d_in.impl->data();
            for (int64_t i = 0; i < n; ++i) {
                float x = in_data[i];
                if (x <= -3.0f) din[i] = 0.0f;
                else if (x >= 3.0f) din[i] = dout[i];
                else din[i] = dout[i] * (2.0f * x + 3.0f) / 6.0f;
            }
        } else {
            cuda::launch_hardswish_backward(in_saved.impl->data(), grad_output.impl->data(), d_in.impl->data(), (int)n);
        }
        accumulate_grad(inputs[0], d_in);
    }
};
```

### Step 5: Wrap in `Tensor` Class in `include/sovereigngrad/tensor.hpp` & `src/tensor.cpp`
```cpp
// in tensor.hpp:
Tensor hardswish() const;

// in tensor.cpp:
Tensor Tensor::hardswish() const {
    return sovereigngrad::hardswish(*this);
}
```

### Step 6: Add Finite-Difference Gradcheck & Test
In `src/test_suite.cpp`:
```cpp
void test_hardswish_gradcheck(Device device) {
    auto fn = [](const std::vector<Tensor>& inputs) -> Tensor {
        return inputs[0].hardswish().sum();
    };
    Tensor a = Tensor::randn({16, 16}, 0.0f, 1.0f, device, true);
    assert(gradcheck(fn, {a}, 1e-3f, 1e-2f));
    std::cout << "  * Hardswish Gradcheck: PASSED\n";
}
```

---

## 3. How to Add a New Neural Network Layer

All layers inherit from `nn::Module` in `include/sovereigngrad/nn.hpp`:

1. Inherit from `nn::Module`.
2. Register trainable parameters using `register_parameter("name", tensor)`.
3. Register submodules using `register_module("name", module_ptr)`.
4. Implement `Tensor forward(const Tensor& x)` and `operator()`.

```cpp
class Hardswish : public Module {
public:
    Tensor forward(const Tensor& x) {
        return hardswish(x);
    }
    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};
```

---

## 4. Testing & Numerical Gradchecks

Always verify every new op and layer with finite-difference gradient checks on both `Device::CPU` and `Device::CUDA`:

```bash
# Run full automated test suite
./build/sovereign_grad_test
```

### Finite-Difference Formulation
The finite difference approximation used by `gradcheck` is:
$$f'(x_i) \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$
The analytical gradient computed via `loss.backward()` must match the numerical gradient within tolerance:
$$\max_i |g_{\text{analytical}, i} - g_{\text{numerical}, i}| \le \text{atol}$$

---

## 5. Code Style & Engineering Standards

- **C++ Standard**: C++17.
- **Header Guards**: `#pragma once` on all header files.
- **Naming Conventions**:
  - `snake_case` for functions, methods, and variables (`forward`, `backward`, `numel`, `in_features`).
  - `PascalCase` for classes, structs, and enums (`Tensor`, `LayerNorm`, `MemoryAllocator`, `Device`).
  - `UPPER_CASE` for macros (`LAUNCH_KERNEL`, `CUDA_CHECK`).
- **Memory Management**:
  - Always use `MemoryAllocator::instance().allocate()` / `free()` rather than raw `new`/`malloc`.
  - Avoid memory leaks: verify with AddressSanitizer (`-DENABLE_SANITIZERS=ON`).
- **Autograd Safety**:
  - Every `Node` implementation must call `check_versions()` at the beginning of `backward()`.
  - In-place tensor modifications must call `impl->bump_version()`.

---

## 6. Pull Request & Submission Workflow

1. **Fork & Branch**: Create a feature branch with a descriptive name:
   ```bash
   git checkout -b feature/add-layernorm-simd
   ```
2. **Implement & Test**: Write your feature, unit tests, and gradchecks. Verify that `./build/sovereign_grad_test` passes with 0 failures.
3. **Commit**: Use clear, conventional commit messages:
   ```bash
   git commit -m "feat(ops): implement Hardswish forward and backward autograd node"
   ```
4. **Submit PR**: Open a Pull Request on GitHub against the `main` branch. Ensure the description explains:
   - What the PR accomplishes.
   - Test results (gradcheck status and benchmark outputs).

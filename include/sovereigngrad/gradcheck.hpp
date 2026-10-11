#pragma once

#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/autograd.hpp"
#include <functional>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace sovereigngrad {

inline bool gradcheck(std::function<Tensor(Tensor&)> func, Tensor& input, float eps = 1e-3f, float atol = 1e-2f, float rtol = 1e-2f, bool verbose = false) {
    input.set_requires_grad(true);
    input.zero_grad();

    // 1. Analytical Gradient
    Tensor output = func(input);
    if (output.numel() != 1) {
        throw std::runtime_error("gradcheck requires a function that outputs a 1-element scalar tensor!");
    }
    output.backward(/*retain_graph=*/true);
    Tensor analytical_grad = input.grad().cpu().clone();

    // 2. Numerical Gradient via Finite Differences
    Tensor cpu_input = input.cpu().clone();
    int64_t total_elements = cpu_input.numel();
    std::vector<float> numerical_grads(total_elements, 0.0f);

    for (int64_t i = 0; i < total_elements; ++i) {
        float orig_val = cpu_input.impl->data()[i];

        // Evaluate f(x + eps)
        cpu_input.impl->data()[i] = orig_val + eps;
        Tensor input_plus = cpu_input.to(input.device());
        float loss_plus = 0.0f;
        {
            NoGradGuard no_grad;
            loss_plus = func(input_plus).item();
        }

        // Evaluate f(x - eps)
        cpu_input.impl->data()[i] = orig_val - eps;
        Tensor input_minus = cpu_input.to(input.device());
        float loss_minus = 0.0f;
        {
            NoGradGuard no_grad;
            loss_minus = func(input_minus).item();
        }

        // Restore original value
        cpu_input.impl->data()[i] = orig_val;

        // Central difference
        numerical_grads[i] = (loss_plus - loss_minus) / (2.0f * eps);
    }

    // 3. Compare Analytical vs Numerical
    const float* a_ptr = analytical_grad.impl->data();
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (int64_t i = 0; i < total_elements; ++i) {
        float a = a_ptr[i];
        float n = numerical_grads[i];
        float abs_err = std::abs(a - n);
        float denom = std::max(std::abs(a), std::abs(n)) + 1e-7f;
        float rel_err = abs_err / denom;

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;

        if (verbose && (abs_err > atol && rel_err > rtol)) {
            std::cout << "  [gradcheck mismatch at idx " << i << "] analytical: " 
                      << a << ", numerical: " << n << ", rel_err: " << rel_err << "\n";
        }
    }

    bool passed = (max_abs_err <= atol) || (max_rel_err <= rtol);
    if (verbose) {
        std::cout << "  Max Abs Error: " << max_abs_err << ", Max Rel Error: " << max_rel_err 
                  << " -> " << (passed ? "PASSED" : "FAILED") << "\n";
    }
    return passed;
}

} // namespace sovereigngrad

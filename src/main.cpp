#include "soverigngrad/tensor.hpp"
#include "soverigngrad/ops.hpp"
#include "soverigngrad/nn.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>

using namespace sovereigngrad;

// Function to generate synthetic 8x8 digit representations
std::pair<Tensor, Tensor> generate_synthetic_digits(int num_samples, Device device) {
    std::vector<float> X_data;
    std::vector<float> Y_data;
    
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);
    
    int num_classes = 3;
    X_data.reserve(num_samples * 64);
    Y_data.reserve(num_samples * num_classes);
    
    for (int i = 0; i < num_samples; ++i) {
        int label = i % num_classes;
        std::vector<float> pixels(64, 0.0f);
        
        if (label == 0) {
            // Draw a zero: circle border
            for (int r = 1; r < 7; ++r) {
                pixels[r * 8 + 1] = 1.0f;
                pixels[r * 8 + 6] = 1.0f;
            }
            for (int c = 1; c < 7; ++c) {
                pixels[1 * 8 + c] = 1.0f;
                pixels[6 * 8 + c] = 1.0f;
            }
        } else if (label == 1) {
            // Draw a one: vertical center line
            for (int r = 0; r < 8; ++r) {
                pixels[r * 8 + 4] = 1.0f;
            }
        } else {
            // Draw a two: diagonal + top/bottom lines
            for (int c = 1; c < 7; ++c) pixels[1 * 8 + c] = 1.0f;
            for (int r = 2; r < 6; ++r) pixels[r * 8 + (7 - r)] = 1.0f;
            for (int c = 1; c < 7; ++c) pixels[6 * 8 + c] = 1.0f;
        }
        
        // Add pixel noise
        for (int p = 0; p < 64; ++p) {
            float val = pixels[p] + noise_dist(gen);
            X_data.push_back(std::max(0.0f, std::min(val, 1.0f)));
        }
        
        // One-hot label encoding
        std::vector<float> target(num_classes, 0.0f);
        target[label] = 1.0f;
        for (float val : target) {
            Y_data.push_back(val);
        }
    }
    
    Tensor X(X_data, {num_samples, 64}, device);
    Tensor Y(Y_data, {num_samples, num_classes}, device);
    return {X, Y};
}

// Unit testing suite
void run_unit_tests(Device device) {
    std::cout << "\n============================================\n";
    std::cout << "          RUNNING SOVEREIGNGRAD UNIT TESTS\n";
    std::cout << "============================================\n";

    // Test 1: Shape, strides, and contiguous layout
    std::cout << "[Test 1] Layout and Strides: ";
    Tensor a({2, 3}, device);
    assert(a.shape() == std::vector<int64_t>({2, 3}));
    assert(a.strides() == std::vector<int64_t>({3, 1}));
    assert(a.is_contiguous() == true);
    std::cout << "PASSED\n";

    // Test 2: O(1) Transpose
    std::cout << "[Test 2] O(1) Transpose Strides: ";
    Tensor a_trans = a.transpose(0, 1);
    assert(a_trans.shape() == std::vector<int64_t>({3, 2}));
    assert(a_trans.strides() == std::vector<int64_t>({1, 3}));
    assert(a_trans.is_contiguous() == false);
    std::cout << "PASSED\n";

    // Test 3: Basic Arithmetic Forward Pass
    std::cout << "[Test 3] Arithmetic Forward Pass: ";
    Tensor t1({1.0f, 2.0f, 3.0f}, {1, 3}, device);
    Tensor t2({4.0f, 5.0f, 6.0f}, {1, 3}, device);
    Tensor t3 = t1 + t2;
    assert(t3.to(Device::CPU).get_element({0, 0}) == 5.0f);
    assert(t3.to(Device::CPU).get_element({0, 1}) == 7.0f);
    assert(t3.to(Device::CPU).get_element({0, 2}) == 9.0f);
    std::cout << "PASSED\n";

    // Test 4: Autograd backward chain rule derivatives
    std::cout << "[Test 4] Reverse-Mode Autograd: ";
    Tensor w({2.0f, -1.0f}, {2, 1}, device, /*requires_grad=*/true);
    Tensor x({3.0f, 4.0f}, {1, 2}, device); // input
    Tensor b(-0.5f, device, /*requires_grad=*/true); // bias
    
    // forward: y = x * w + b = (3*2 + 4*-1) - 0.5 = 6 - 4 - 0.5 = 1.5
    Tensor y = x.matmul(w) + b;
    assert(std::abs(y.to(Device::CPU).item() - 1.5f) < 1e-5f);

    // backward
    y.backward();
    
    // Gradients: dy/dw = x^T => [3, 4]^T
    // dy/db = 1.0
    Tensor dw = w.grad().to(Device::CPU);
    Tensor db = b.grad().to(Device::CPU);
    
    assert(std::abs(dw.get_element({0, 0}) - 3.0f) < 1e-5f);
    assert(std::abs(dw.get_element({1, 0}) - 4.0f) < 1e-5f);
    assert(std::abs(db.item() - 1.0f) < 1e-5f);
    std::cout << "PASSED\n";

    std::cout << "All unit tests PASSED successfully!\n";
    std::cout << "============================================\n\n";
}

int main() {
    std::cout << std::unitbuf;
    std::cout << "========================================================\n";
    std::cout << "         SOVEREIGNGRAD: DEEPMIND CAPSTONE ENGINE        \n";
    std::cout << "========================================================\n";

#ifdef SOVEREIGNGRAD_USE_CUDA
    Device target_device = Device::CUDA;
    std::cout << "Hardware Status: Physical NVIDIA GPU & NVCC found!\n";
    std::cout << "Backend selected: CUDA GPU Acceleration (VRAM, Hardware kernels)\n";
#else
    Device target_device = Device::CUDA; // Emulated CUDA uses device code on CPU
    std::cout << "Hardware Status: Physical NVIDIA GPU not found. Falling back to emulation.\n";
    std::cout << "Backend selected: CUDA Emulation (Executing .cu kernel grid on CPU threads!)\n";
#endif

    // Run unit tests
    run_unit_tests(target_device);

    // Create the training data: 300 samples of 8x8 pixels (64 features) belonging to classes 0, 1, or 2
    int num_samples = 300;
    std::cout << "Generating " << num_samples << " synthetic handwritten digits (8x8 pixels)...\n";
    auto dataset = generate_synthetic_digits(num_samples, target_device);
    Tensor X = dataset.first;
    Tensor Y = dataset.second;

    std::cout << "Inputs Shape: [" << X.shape()[0] << ", " << X.shape()[1] << "] on device\n";
    std::cout << "Labels Shape: [" << Y.shape()[0] << ", " << Y.shape()[1] << "] on device\n\n";

    // Instantiate Multi-Layer Perceptron (MLP)
    // Layer 1: 64 inputs -> 32 hidden units
    // Layer 2: 32 hidden units -> 3 classes
    std::cout << "Initializing SovereignGrad MLP (Linear -> ReLU -> Linear)...\n";
    Linear fc1(64, 32, target_device);
    Linear fc2(32, 3, target_device);

    // Collect parameters
    std::vector<Tensor> params;
    for (auto& p : fc1.parameters()) params.push_back(p);
    for (auto& p : fc2.parameters()) params.push_back(p);

    // Instantiate custom Adam Optimizer
    float lr = 0.015f;
    std::cout << "Instantiating custom Adam Optimizer (lr=" << lr << ", beta1=0.9, beta2=0.999)...\n\n";
    Adam optimizer(params, lr);

    // Training loop
    int epochs = 60;
    std::cout << "========================================================\n";
    std::cout << "               STARTING TRAINING LOOP                   \n";
    std::cout << "========================================================\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        optimizer.zero_grad();

        // 1. Forward Pass
        Tensor h1 = fc1.forward(X).relu();
        Tensor logits = fc2.forward(h1);
        
        // 2. Compute Loss
        Tensor loss = softmax_cross_entropy(logits, Y);

        // 3. Backward Pass
        loss.backward();

        // 4. Update Weights
        optimizer.step();

        // Compute metrics
        float loss_val = loss.to(Device::CPU).item();
        
        // Compute accuracy (on CPU for easy max-finding)
        Tensor cpu_logits = logits.to(Device::CPU);
        Tensor cpu_targets = Y.to(Device::CPU);
        int correct = 0;
        
        for (int i = 0; i < num_samples; ++i) {
            float max_val = -std::numeric_limits<float>::infinity();
            int pred_class = -1;
            for (int c = 0; c < 3; ++c) {
                float val = cpu_logits.get_element({i, c});
                if (val > max_val) {
                    max_val = val;
                    pred_class = c;
                }
            }
            
            int target_class = -1;
            for (int c = 0; c < 3; ++c) {
                if (cpu_targets.get_element({i, c}) == 1.0f) {
                    target_class = c;
                    break;
                }
            }
            
            if (pred_class == target_class) {
                correct++;
            }
        }
        
        float acc = static_cast<float>(correct) / num_samples * 100.0f;

        if (epoch == 1 || epoch % 5 == 0 || epoch == epochs) {
            std::cout << "Epoch " << std::setw(3) << epoch << "/" << epochs
                      << " | Loss: " << std::fixed << std::setprecision(5) << loss_val
                      << " | Accuracy: " << std::setw(6) << std::setprecision(2) << acc << "%"
                      << " | Cache Blocks: " 
                      << MemoryAllocator::instance().active_allocations(target_device) + 
                         MemoryAllocator::instance().cached_allocations(target_device)
                      << " (" << MemoryAllocator::instance().total_memory_allocated(target_device) / 1024 << " KB)\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "========================================================\n";
    std::cout << "Training completed in " << std::fixed << std::setprecision(3) 
              << duration.count() << " seconds!\n";
    std::cout << "========================================================\n\n";

    // Release memory allocator blocks
    MemoryAllocator::instance().release_all();
    std::cout << "SovereignGrad resources released. Exiting...\n";

    return 0;
}

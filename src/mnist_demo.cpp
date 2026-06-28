#include "soverigngrad/tensor.hpp"
#include "soverigngrad/ops.hpp"
#include "soverigngrad/nn.hpp"
#include "soverigngrad/mnist.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace sovereigngrad;

int main() {
    std::cout << "========================================================\n";
    std::cout << "        SOVEREIGNGRAD: MNIST CONVOLUTIONAL DEMO        \n";
    std::cout << "========================================================\n";

#ifdef SOVEREIGNGRAD_USE_CUDA
    Device target_device = Device::CUDA;
    std::cout << "Backend selected: Physical NVIDIA GPU (NVCC enabled)\n";
#else
    Device target_device = Device::CUDA; // Emulated CUDA uses CPU threads
    std::cout << "Backend selected: CUDA Emulation (Executing .cu kernel grids)\n";
#endif

    // Load or generate MNIST data
    int num_samples = 150;
    auto dataset = load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", num_samples, target_device);
    Tensor X = dataset.images;
    Tensor Y = dataset.labels;

    std::cout << "Inputs Shape: [" << X.shape()[0] << ", " << X.shape()[1] << ", " 
              << X.shape()[2] << ", " << X.shape()[3] << "] on device\n";
    std::cout << "Labels Shape: [" << Y.shape()[0] << ", " << Y.shape()[1] << "] on device\n\n";

    // Instantiate CNN: Conv2d -> BatchNorm2d -> ReLU -> Reshape -> Linear -> SoftmaxCrossEntropy
    std::cout << "Initializing SovereignGrad CNN...\n";
    std::cout << "  Layer 1: Conv2d(1 -> 4 channels, 3x3 kernel, padding=1)\n";
    Conv2d conv(1, 4, 3, /*stride=*/1, /*padding=*/1, target_device);
    
    std::cout << "  Layer 2: BatchNorm2d(4 channels)\n";
    BatchNorm2d bn(4, /*momentum=*/0.1f, /*eps=*/1e-5f, target_device);
    
    std::cout << "  Layer 3: Linear(3136 -> 10 output classes)\n\n";
    // 4 channels * 28 * 28 pixels = 3136 elements
    Linear fc(3136, 10, target_device);

    // Collect parameter weights
    std::vector<Tensor> params;
    for (auto& p : conv.parameters()) params.push_back(p);
    for (auto& p : bn.parameters()) params.push_back(p);
    for (auto& p : fc.parameters()) params.push_back(p);

    // Instantiate custom Adam Optimizer
    float lr = 0.02f;
    std::cout << "Instantiating custom Adam Optimizer (lr=" << lr << ")...\n\n";
    Adam optimizer(params, lr);

    // Training loop
    int epochs = 15;
    std::cout << "========================================================\n";
    std::cout << "               STARTING CNN TRAINING                    \n";
    std::cout << "========================================================\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        optimizer.zero_grad();
        
        // Ensure BatchNorm is in training mode
        bn.training = true;

        // 1. Forward Pass
        // input: [B, 1, 28, 28] -> conv: [B, 4, 28, 28] -> bn: [B, 4, 28, 28] -> relu: [B, 4, 28, 28]
        Tensor c1 = conv.forward(X);
        Tensor b1 = bn.forward(c1);
        Tensor r1 = b1.relu();
        
        // 2. Reshape to flat: [B, 3136]
        Tensor flat = r1.view({num_samples, 3136});
        
        // 3. Dense layer: [B, 10]
        Tensor logits = fc.forward(flat);

        // 4. Compute Softmax Cross Entropy Loss
        Tensor loss = softmax_cross_entropy(logits, Y);

        // 5. Backward Pass
        loss.backward();

        // 6. Adam weights step update
        optimizer.step();

        // Calculate loss value
        float loss_val = loss.to(Device::CPU).item();
        
        // Calculate train classification accuracy
        Tensor cpu_logits = logits.to(Device::CPU);
        Tensor cpu_targets = Y.to(Device::CPU);
        int correct = 0;
        
        for (int i = 0; i < num_samples; ++i) {
            float max_val = -std::numeric_limits<float>::infinity();
            int pred_class = -1;
            for (int c = 0; c < 10; ++c) {
                float val = cpu_logits.get_element({i, c});
                if (val > max_val) {
                    max_val = val;
                    pred_class = c;
                }
            }
            
            int target_class = -1;
            for (int c = 0; c < 10; ++c) {
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

        std::cout << "Epoch " << std::setw(2) << epoch << "/" << epochs
                  << " | CNN CrossEntropy Loss: " << std::fixed << std::setprecision(5) << loss_val
                  << " | Training Accuracy: " << std::setw(6) << std::setprecision(2) << acc << "%"
                  << " | VRAM Memory Allocations: " 
                  << MemoryAllocator::instance().active_allocations(target_device)
                  << " blocks\n";
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "========================================================\n";
    std::cout << "MNIST CNN Training completed in " << std::fixed << std::setprecision(3) 
              << duration.count() << " seconds!\n";
    std::cout << "========================================================\n\n";

    // Clean up allocator resources
    MemoryAllocator::instance().release_all();
    std::cout << "Resources released. Demo successfully executed!\n";

    return 0;
}

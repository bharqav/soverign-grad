#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/ops.hpp"
#include "sovereigngrad/nn.hpp"
#include "sovereigngrad/module.hpp"
#include "sovereigngrad/serialization.hpp"
#include "sovereigngrad/gradcheck.hpp"
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

// Production Digit Classifier Neural Network
class DigitClassifier : public nn::Module {
public:
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::LayerNorm> ln1;
    std::shared_ptr<nn::Linear> fc2;
    std::shared_ptr<nn::LayerNorm> ln2;
    std::shared_ptr<nn::Linear> fc3;
    std::shared_ptr<nn::Dropout> drop;

    DigitClassifier(int in_features, int hidden_dim, int num_classes, Device device = Device::CPU) {
        fc1 = std::make_shared<nn::Linear>(in_features, hidden_dim, device);
        ln1 = std::make_shared<nn::LayerNorm>(hidden_dim, 1e-5f, device);
        fc2 = std::make_shared<nn::Linear>(hidden_dim, hidden_dim, device);
        ln2 = std::make_shared<nn::LayerNorm>(hidden_dim, 1e-5f, device);
        fc3 = std::make_shared<nn::Linear>(hidden_dim, num_classes, device);
        drop = std::make_shared<nn::Dropout>(0.1f);

        register_module("fc1", fc1);
        register_module("ln1", ln1);
        register_module("fc2", fc2);
        register_module("ln2", ln2);
        register_module("fc3", fc3);
        register_module("drop", drop);
    }

    Tensor forward(const Tensor& x) {
        Tensor h1 = fc1->forward(x).gelu();
        h1 = ln1->forward(h1);
        h1 = drop->forward(h1);

        Tensor h2 = fc2->forward(h1).gelu();
        h2 = ln2->forward(h2);

        return fc3->forward(h2);
    }
};

void train_model(Device device) {
    std::cout << "\n=======================================================\n";
    std::cout << "  TRAINING MULTI-LAYER DIGIT CLASSIFIER ON " << device_to_string(device) << "\n";
    std::cout << "=======================================================\n";

    int num_samples = 300;
    int in_features = 64;
    int hidden_dim = 32;
    int num_classes = 3;
    int epochs = 60;
    float lr = 0.01f;

    auto [X_all, Y_all] = generate_synthetic_digits(num_samples, device);

    // Split train/test (80/20)
    int train_size = 240;
    int test_size = 60;

    std::vector<float> x_train_vec(train_size * in_features);
    std::vector<float> y_train_vec(train_size * num_classes);
    std::vector<float> x_test_vec(test_size * in_features);
    std::vector<float> y_test_vec(test_size * num_classes);

    Tensor cpu_X = X_all.cpu();
    Tensor cpu_Y = Y_all.cpu();

    std::memcpy(x_train_vec.data(), cpu_X.impl->data(), train_size * in_features * sizeof(float));
    std::memcpy(y_train_vec.data(), cpu_Y.impl->data(), train_size * num_classes * sizeof(float));

    std::memcpy(x_test_vec.data(), cpu_X.impl->data() + train_size * in_features, test_size * in_features * sizeof(float));
    std::memcpy(y_test_vec.data(), cpu_Y.impl->data() + train_size * num_classes, test_size * num_classes * sizeof(float));

    Tensor X_train(x_train_vec, {train_size, in_features}, device);
    Tensor Y_train(y_train_vec, {train_size, num_classes}, device);
    Tensor X_test(x_test_vec, {test_size, in_features}, device);
    Tensor Y_test(y_test_vec, {test_size, num_classes}, device);

    DigitClassifier model(in_features, hidden_dim, num_classes, device);
    nn::AdamW optimizer(model.parameters(), lr, 0.9f, 0.999f, 1e-8f, /*weight_decay=*/0.001f);

    MemoryAllocator::instance().reset_peak_memory_stats();
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        model.train();
        optimizer.zero_grad();

        Tensor logits = model.forward(X_train);
        Tensor loss = softmax_cross_entropy(logits, Y_train);

        loss.backward();
        optimizer.step();

        if (epoch % 15 == 0 || epoch == 1 || epoch == epochs) {
            model.eval();
            NoGradGuard no_grad;

            Tensor test_logits = model.forward(X_test);
            Tensor test_loss = softmax_cross_entropy(test_logits, Y_test);

            // Compute Accuracy
            Tensor cpu_test_logits = test_logits.cpu();
            int correct = 0;
            for (int i = 0; i < test_size; ++i) {
                int pred_cls = 0;
                float max_val = cpu_test_logits.get_element({i, 0});
                for (int c = 1; c < num_classes; ++c) {
                    float v = cpu_test_logits.get_element({i, c});
                    if (v > max_val) {
                        max_val = v;
                        pred_cls = c;
                    }
                }
                int actual_cls = (train_size + i) % num_classes;
                if (pred_cls == actual_cls) correct++;
            }
            float accuracy = (float)correct / (float)test_size * 100.0f;

            std::cout << "Epoch [" << std::setw(2) << epoch << "/" << epochs << "] "
                      << "Train Loss: " << std::fixed << std::setprecision(4) << loss.item()
                      << " | Test Loss: " << test_loss.item()
                      << " | Test Accuracy: " << std::setprecision(1) << accuracy << "%\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "\nTraining Completed in " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";
    std::cout << MemoryAllocator::instance().get_memory_summary(device) << "\n";

    // Save model checkpoint
    std::string checkpoint_file = "sovereign_digit_classifier.sgrd";
    serialization::save_module(model, checkpoint_file);
    std::cout << "Saved trained checkpoint to: " << checkpoint_file << "\n";
}

int main() {
    std::cout << "=======================================================\n";
    std::cout << "       SOVEREIGNGRAD PRODUCTION DEMONSTRATION          \n";
    std::cout << "=======================================================\n";

    // Train on CPU and CUDA Emulation
    train_model(Device::CPU);
    train_model(Device::CUDA);

    std::cout << "\nAll demonstrations finished successfully.\n";
    return 0;
}

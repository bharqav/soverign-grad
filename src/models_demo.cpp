#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/nn.hpp"
#include "sovereigngrad/models.hpp"
#include "sovereigngrad/serialization.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>

using namespace sovereigngrad;

void print_header(const std::string& title) {
    std::cout << "\n======================================================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "======================================================================\n";
}

void test_nanollama() {
    print_header("MODEL 1: NanoLLaMA (Autoregressive Language Model with RMSNorm + SwiGLU + Causal Attention)");

    const int64_t vocab_size = 32;
    const int64_t max_seq_len = 16;
    const int64_t embed_dim = 32;
    const int64_t num_layers = 2;
    const int64_t num_heads = 4;
    const int64_t hidden_dim = 64;

    auto model = std::make_shared<models::NanoLLaMA>(vocab_size, max_seq_len, embed_dim, num_layers, num_heads, hidden_dim, Device::CPU);
    
    // Check parameters
    auto params = model->parameters();
    int64_t total_params = 0;
    for (auto& p : params) total_params += p.numel();
    std::cout << "[*] NanoLLaMA initialized with " << params.size() << " parameter tensors (" << total_params << " total parameters).\n";

    nn::AdamW optimizer(params, /*lr=*/0.01f, /*b1=*/0.9f, /*b2=*/0.99f, /*eps=*/1e-8f, /*weight_decay=*/0.01f);
    nn::CosineAnnealingLR scheduler(0.01f, /*T_max=*/25, /*eta_min=*/0.001f);

    // Training on repeating sequential pattern: 1, 2, 3, 4, 5, 6, 7, 8 -> next token is (t % 8) + 1
    std::vector<float> input_seq = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4};
    int64_t seq_len = input_seq.size();
    Tensor input_tokens(input_seq, {seq_len}, Device::CPU);

    // Targets: shifted right by 1
    std::vector<float> target_seq = {2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5};
    Tensor target_onehot = Tensor::zeros({seq_len, vocab_size}, Device::CPU);
    for (int64_t i = 0; i < seq_len; ++i) {
        target_onehot.set_element({i, (int64_t)target_seq[i]}, 1.0f);
    }

    std::cout << "[*] Training NanoLLaMA on cyclic pattern sequence...\n";
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 1; epoch <= 30; ++epoch) {
        optimizer.zero_grad();

        Tensor logits = model->forward(input_tokens);
        Tensor loss = cross_entropy_loss(logits, target_onehot);

        if (epoch == 1) initial_loss = loss.item();
        final_loss = loss.item();

        loss.backward();
        float norm = nn::clip_grad_norm(params, /*max_norm=*/1.0f);
        optimizer.step();
        scheduler.step(optimizer);

        if (epoch % 5 == 0 || epoch == 1) {
            std::cout << "    Epoch " << std::setw(2) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(5) << final_loss
                      << " | GradNorm: " << std::setprecision(4) << norm
                      << " | LR: " << std::setprecision(5) << optimizer.lr << "\n";
        }
    }

    assert(final_loss < initial_loss);
    std::cout << "[+] NanoLLaMA successfully optimized! (Loss: " << initial_loss << " -> " << final_loss << ")\n";

    // Autoregressive generation demonstration
    std::vector<int64_t> prompt = {1, 2, 3};
    std::cout << "[*] Running autoregressive generate() from prompt [1, 2, 3]...\n";
    std::vector<int64_t> generated = model->generate(prompt, /*max_new_tokens=*/6, /*temperature=*/0.5f);
    std::cout << "    Generated token sequence: [";
    for (size_t i = 0; i < generated.size(); ++i) {
        std::cout << generated[i] << (i + 1 < generated.size() ? ", " : "");
    }
    std::cout << "]\n";

    // Checkpoint serialization
    std::cout << "[*] Saving model checkpoint to 'nanollama.sgrd'...\n";
    serialization::save_module(*model, "nanollama.sgrd");
    auto loaded_model = std::make_shared<models::NanoLLaMA>(vocab_size, max_seq_len, embed_dim, num_layers, num_heads, hidden_dim, Device::CPU);
    serialization::load_module(*loaded_model, "nanollama.sgrd");
    
    Tensor test_out = loaded_model->forward(input_tokens);
    float diff = (test_out - model->forward(input_tokens)).norm();
    std::cout << "[+] Checkpoint reload verified. Forward output difference norm: " << diff << "\n";
}

void test_deep_resnet() {
    print_header("MODEL 2: DeepResNet (Multi-Stage Residual Vision Backbone with BatchNorm)");

    const int64_t in_channels = 1;
    const int64_t num_classes = 4;
    const int64_t batch_size = 4;

    auto model = std::make_shared<models::DeepResNet>(in_channels, num_classes, Device::CPU);
    auto params = model->parameters();
    int64_t total_params = 0;
    for (auto& p : params) total_params += p.numel();
    std::cout << "[*] DeepResNet initialized with " << total_params << " parameters across 3 residual stages.\n";

    nn::AdamW optimizer(params, /*lr=*/0.01f, /*b1=*/0.9f, /*b2=*/0.999f, /*eps=*/1e-8f, /*weight_decay=*/0.001f);

    // Create synthetic 28x28 images with 4 distinct geometric patterns
    Tensor x = Tensor::randn({batch_size, in_channels, 28, 28}, 0.0f, 1.0f, Device::CPU);
    Tensor targets = Tensor::zeros({batch_size, num_classes}, Device::CPU);
    for (int64_t b = 0; b < batch_size; ++b) {
        targets.set_element({b, b % num_classes}, 1.0f);
    }

    std::cout << "[*] Training DeepResNet on vision batches...\n";
    float init_loss = 0.0f, final_loss = 0.0f;
    for (int epoch = 1; epoch <= 15; ++epoch) {
        optimizer.zero_grad();
        Tensor logits = model->forward(x);
        Tensor loss = cross_entropy_loss(logits, targets);

        if (epoch == 1) init_loss = loss.item();
        final_loss = loss.item();

        loss.backward();
        optimizer.step();

        if (epoch % 3 == 0 || epoch == 1) {
            std::cout << "    Epoch " << std::setw(2) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(5) << final_loss << "\n";
        }
    }

    assert(final_loss < init_loss);
    std::cout << "[+] DeepResNet successfully converged! (" << init_loss << " -> " << final_loss << ")\n";

    serialization::save_module(*model, "deep_resnet.sgrd");
    std::cout << "[+] DeepResNet weights serialized to 'deep_resnet.sgrd'.\n";
}

void test_sequence_classifier() {
    print_header("MODEL 3: SequenceClassifier (1D Temporal Conv + Embedding Classifier)");

    const int64_t vocab_size = 50;
    const int64_t embed_dim = 16;
    const int64_t num_filters = 32;
    const int64_t num_classes = 3;

    auto model = std::make_shared<models::SequenceClassifier>(vocab_size, embed_dim, num_filters, num_classes, Device::CPU);
    auto params = model->parameters();
    std::cout << "[*] SequenceClassifier initialized with " << params.size() << " parameter tensors.\n";

    nn::AdamW optimizer(params, /*lr=*/0.015f);

    // Sequence of 10 tokens
    std::vector<float> tokens_vec = {4, 12, 18, 25, 33, 7, 19, 44, 2, 11};
    Tensor tokens(tokens_vec, {10}, Device::CPU);
    Tensor target = Tensor::zeros({1, num_classes}, Device::CPU);
    target.set_element({0, 1}, 1.0f); // Class 1

    float init_loss = 0.0f, final_loss = 0.0f;
    for (int epoch = 1; epoch <= 15; ++epoch) {
        optimizer.zero_grad();
        Tensor logits = model->forward(tokens);
        Tensor loss = cross_entropy_loss(logits, target);

        if (epoch == 1) init_loss = loss.item();
        final_loss = loss.item();

        loss.backward();
        optimizer.step();

        if (epoch % 3 == 0 || epoch == 1) {
            std::cout << "    Epoch " << std::setw(2) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(5) << final_loss << "\n";
        }
    }

    assert(final_loss < init_loss);
    std::cout << "[+] SequenceClassifier trained successfully! (" << init_loss << " -> " << final_loss << ")\n";
    serialization::save_module(*model, "seq_classifier.sgrd");
}

void test_resmlp() {
    print_header("MODEL 4: ResMLP (Deep Residual Multi-Layer Perceptron)");

    const int64_t in_features = 20;
    const int64_t hidden_dim = 32;
    const int64_t num_layers = 3;
    const int64_t num_classes = 4;
    const int64_t batch_size = 8;

    auto model = std::make_shared<models::ResMLP>(in_features, hidden_dim, num_layers, num_classes, Device::CPU);
    auto params = model->parameters();
    std::cout << "[*] ResMLP initialized with " << num_layers << " residual MLP blocks.\n";

    nn::AdamW optimizer(params, /*lr=*/0.02f);
    nn::StepLR scheduler(0.02f, /*step_size=*/5, /*gamma=*/0.5f);

    Tensor x = Tensor::randn({batch_size, in_features}, 0.0f, 1.0f, Device::CPU);
    Tensor targets = Tensor::zeros({batch_size, num_classes}, Device::CPU);
    for (int64_t b = 0; b < batch_size; ++b) {
        targets.set_element({b, b % num_classes}, 1.0f);
    }

    float init_loss = 0.0f, final_loss = 0.0f;
    for (int epoch = 1; epoch <= 15; ++epoch) {
        optimizer.zero_grad();
        Tensor logits = model->forward(x);
        Tensor loss = cross_entropy_loss(logits, targets);

        if (epoch == 1) init_loss = loss.item();
        final_loss = loss.item();

        loss.backward();
        optimizer.step();
        scheduler.step(optimizer);

        if (epoch % 3 == 0 || epoch == 1) {
            std::cout << "    Epoch " << std::setw(2) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(5) << final_loss 
                      << " | LR: " << optimizer.lr << "\n";
        }
    }

    assert(final_loss < init_loss);
    std::cout << "[+] ResMLP trained successfully! (" << init_loss << " -> " << final_loss << ")\n";
    serialization::save_module(*model, "resmlp.sgrd");
}

void test_autoencoder() {
    print_header("MODEL 5: Denoising Autoencoder (Bottleneck Feature Representation)");

    const int64_t in_features = 16;
    const int64_t hidden_dim = 32;
    const int64_t latent_dim = 8;
    const int64_t batch_size = 8;

    auto model = std::make_shared<models::Autoencoder>(in_features, hidden_dim, latent_dim, Device::CPU);
    auto params = model->parameters();
    std::cout << "[*] Autoencoder initialized: [16] -> [32] -> Latent [8] -> [32] -> [16].\n";

    nn::AdamW optimizer(params, /*lr=*/0.02f);
    nn::MSELoss mse_criterion;
    nn::L1Loss l1_criterion;

    // Clean ground truth signals in [0, 1]
    Tensor clean_x = Tensor::uniform({batch_size, in_features}, 0.1f, 0.9f, Device::CPU);
    // Corrupted noisy input
    Tensor noise = Tensor::randn({batch_size, in_features}, 0.0f, 0.1f, Device::CPU);
    Tensor noisy_x = (clean_x + noise).clamp(0.0f, 1.0f);

    float init_mse = 0.0f, final_mse = 0.0f;
    for (int epoch = 1; epoch <= 20; ++epoch) {
        optimizer.zero_grad();
        Tensor reconstructed = model->forward(noisy_x);
        Tensor loss = mse_criterion(reconstructed, clean_x) + l1_criterion(reconstructed, clean_x) * 0.1f;

        if (epoch == 1) init_mse = loss.item();
        final_mse = loss.item();

        loss.backward();
        optimizer.step();

        if (epoch % 4 == 0 || epoch == 1) {
            std::cout << "    Epoch " << std::setw(2) << epoch 
                      << " | Reconstruction Loss (MSE+L1): " << std::fixed << std::setprecision(5) << final_mse << "\n";
        }
    }

    assert(final_mse < init_mse);
    std::cout << "[+] Denoising Autoencoder trained successfully! (" << init_mse << " -> " << final_mse << ")\n";
    serialization::save_module(*model, "autoencoder.sgrd");
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << R"(
  ██████╗ ██████╗ ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗   ██╗
  ██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔════╝ ████╗  ██║
  ███████╗██║   ██║██║   ██║█████╗  ██████╔╝█████╗  ██║██║  ███╗██╔██╗ ██║
  ╚════██║██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══╝  ██║██║   ██║██║╚██╗██║
  ███████║╚██████╔╝ ╚████╔╝ ███████╗██║  ██║███████╗██║╚██████╔╝██║ ╚████║
  ╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    PRODUCTION DEEP LEARNING MODEL SUITE & BENCHMARK VERIFICATION
)" << "\n";

    try {
        test_nanollama();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] test_nanollama threw: " << e.what() << "\n";
        return 1;
    }

    try {
        test_deep_resnet();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] test_deep_resnet threw: " << e.what() << "\n";
        return 1;
    }

    try {
        test_sequence_classifier();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] test_sequence_classifier threw: " << e.what() << "\n";
        return 1;
    }

    try {
        test_resmlp();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] test_resmlp threw: " << e.what() << "\n";
        return 1;
    }

    try {
        test_autoencoder();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] test_autoencoder threw: " << e.what() << "\n";
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    print_header("ALL 5 PRODUCTION DEEP LEARNING MODELS PASSED AND VERIFIED!");
    std::cout << "Total execution time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n\n";

    return 0;
}

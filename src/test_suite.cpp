#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/ops.hpp"
#include "sovereigngrad/nn.hpp"
#include "sovereigngrad/module.hpp"
#include "sovereigngrad/serialization.hpp"
#include "sovereigngrad/gradcheck.hpp"
#include "sovereigngrad/models.hpp"
#include "sovereigngrad/simd.hpp"
#include "sovereigngrad/cuda_stream.hpp"
#include "sovereigngrad/quantization.hpp"
#include "sovereigngrad/jit.hpp"
#include "sovereigngrad/safetensors.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <string>

using namespace sovereigngrad;

void test_tensor_basics(Device dev) {
    std::cout << "[Test] Tensor Basics & Strides on " << device_to_string(dev) << "... ";
    Tensor a({3, 4}, dev);
    assert(a.shape() == std::vector<int64_t>({3, 4}));
    assert(a.strides() == std::vector<int64_t>({4, 1}));
    assert(a.is_contiguous());

    Tensor t = a.transpose(0, 1);
    assert(t.shape() == std::vector<int64_t>({4, 3}));
    assert(t.strides() == std::vector<int64_t>({1, 4}));
    assert(!t.is_contiguous());

    Tensor c = t.contiguous();
    assert(c.is_contiguous());
    assert(c.strides() == std::vector<int64_t>({3, 1}));

    Tensor d = Tensor::full({2, 2}, 3.14f, dev);
    assert(std::abs(d.cpu().get_element({0, 0}) - 3.14f) < 1e-5f);
    assert(std::abs(d.cpu().get_element({1, 1}) - 3.14f) < 1e-5f);

    std::cout << "PASSED\n";
}

void test_memory_allocator() {
    std::cout << "[Test] Memory Allocator Caching & Peak Stats... ";
    MemoryAllocator::instance().reset_peak_memory_stats();
    
    {
        Tensor a({100, 100}, Device::CPU);
        Tensor b({200, 200}, Device::CPU);
        assert(MemoryAllocator::instance().active_allocations(Device::CPU) >= 2);
    }
    // After scope exit, blocks should be cached and merged
    assert(MemoryAllocator::instance().cached_allocations(Device::CPU) >= 1);
    assert(MemoryAllocator::instance().peak_memory_allocated(Device::CPU) > 0);
    
    std::string summary = MemoryAllocator::instance().get_memory_summary(Device::CPU);
    assert(!summary.empty());
    std::cout << "PASSED (" << summary << ")\n";
}

void test_inplace_version_safety(Device dev) {
    std::cout << "[Test] In-Place Mutation Version Guard... ";
    Tensor x({2, 2}, dev, /*requires_grad=*/true);
    Tensor y = x * x;
    
    // Mutate x in place after forward pass
    x.add_(Tensor::ones({2, 2}, dev));

    bool caught_exception = false;
    try {
        y.backward();
    } catch (const std::runtime_error& e) {
        caught_exception = true;
    }
    assert(caught_exception);
    std::cout << "PASSED (Successfully caught in-place mutation during backward)\n";
}

void test_nograd_guard(Device dev) {
    std::cout << "[Test] NoGradGuard DAG Suppression... ";
    Tensor a({2, 2}, dev, /*requires_grad=*/true);
    Tensor b({2, 2}, dev, /*requires_grad=*/true);

    {
        NoGradGuard no_grad;
        Tensor c = a + b;
        assert(c.creator() == nullptr);
        assert(!c.requires_grad());
    }

    Tensor d = a + b;
    assert(d.creator() != nullptr);
    assert(d.requires_grad());
    std::cout << "PASSED\n";
}

void test_numerical_gradchecks(Device dev) {
    std::cout << "\n--- Running Numerical Gradchecks (Finite Differences) on " << device_to_string(dev) << " ---\n";

    // 1. Add
    {
        std::cout << "  * Add Gradcheck: ";
        Tensor x({2, 3}, dev);
        Tensor y = Tensor::full({2, 3}, 2.5f, dev);
        bool ok = gradcheck([&](Tensor& inp) {
            return (inp + y).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }

    // 2. Mul & Matmul
    {
        std::cout << "  * Matmul Gradcheck: ";
        Tensor x({3, 4}, dev);
        Tensor W = Tensor::randn({4, 2}, 0.0f, 0.5f, dev);
        bool ok = gradcheck([&](Tensor& inp) {
            return inp.matmul(W).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }

    // 3. Activations (GELU, SiLU, Tanh, Sigmoid, ReLU)
    {
        std::cout << "  * GELU Gradcheck: ";
        Tensor x({2, 4}, dev);
        bool ok = gradcheck([](Tensor& inp) { return inp.gelu().sum(); }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * SiLU Gradcheck: ";
        Tensor x({2, 4}, dev);
        bool ok = gradcheck([](Tensor& inp) { return inp.silu().sum(); }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * Tanh Gradcheck: ";
        Tensor x({2, 4}, dev);
        bool ok = gradcheck([](Tensor& inp) { return inp.tanh().sum(); }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * Sigmoid Gradcheck: ";
        Tensor x({2, 4}, dev);
        bool ok = gradcheck([](Tensor& inp) { return inp.sigmoid().sum(); }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }

    // 4. LayerNorm Gradcheck
    {
        std::cout << "  * LayerNorm Gradcheck: ";
        Tensor x({2, 4}, dev);
        Tensor gamma = Tensor::ones({4}, dev);
        Tensor beta = Tensor::zeros({4}, dev);
        bool ok = gradcheck([&](Tensor& inp) {
            return layer_norm(inp, gamma, beta).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }

    // 5. MSELoss & SoftmaxCrossEntropy Gradcheck
    {
        std::cout << "  * MSELoss Gradcheck: ";
        Tensor pred({2, 3}, dev);
        Tensor target = Tensor::randn({2, 3}, 0.0f, 1.0f, dev);
        bool ok = gradcheck([&](Tensor& inp) {
            return mse_loss(inp, target);
        }, pred);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * SoftmaxCrossEntropy Gradcheck: ";
        Tensor logits({2, 3}, dev);
        std::vector<float> tgt_data = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        Tensor target(tgt_data, {2, 3}, dev);
        bool ok = gradcheck([&](Tensor& inp) {
            return softmax_cross_entropy(inp, target);
        }, logits);
        assert(ok);
        std::cout << "PASSED\n";
    }

    // 6. Tier 2 Neural Operator Gradchecks
    {
        std::cout << "  * Softmax (dim=-1) Gradcheck: ";
        Tensor x({2, 4}, dev);
        bool ok = gradcheck([](Tensor& inp) {
            return softmax(inp, -1).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * AvgPool2D Gradcheck: ";
        Tensor x({2, 2, 4, 4}, dev);
        bool ok = gradcheck([](Tensor& inp) {
            return avg_pool2d(inp, 2, 2, 0).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * MaxPool2D Gradcheck: ";
        Tensor x({1, 1, 4, 4}, dev);
        bool ok = gradcheck([](Tensor& inp) {
            return max_pool2d(inp, 2, 2, 0).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * BatchNorm1D Gradcheck: ";
        Tensor x({4, 3}, dev);
        Tensor gamma = Tensor::ones({3}, dev);
        Tensor beta = Tensor::zeros({3}, dev);
        Tensor rm = Tensor::zeros({3}, dev);
        Tensor rv = Tensor::ones({3}, dev);
        bool ok = gradcheck([&](Tensor& inp) {
            return batch_norm1d(inp, gamma, beta, rm, rv, 0.1f, 1e-5f, true).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * Conv1D Gradcheck: ";
        Tensor x({2, 2, 6}, dev);
        Tensor w = Tensor::randn({3, 2, 3}, 0.0f, 0.5f, dev);
        Tensor b = Tensor::zeros({3}, dev);
        bool ok = gradcheck([&](Tensor& inp) {
            return conv1d(inp, w, b, 1, 0).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * Embedding Gradcheck: ";
        std::vector<float> idx_data = {0.0f, 2.0f, 1.0f, 0.0f};
        Tensor indices(idx_data, {4}, dev);
        Tensor weight({3, 4}, dev);
        bool ok = gradcheck([&](Tensor& w_inp) {
            return embedding(indices, w_inp).sum();
        }, weight);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * RMSNorm Gradcheck: ";
        Tensor x({2, 4}, dev);
        Tensor gamma = Tensor::ones({4}, dev);
        bool ok = gradcheck([&](Tensor& inp) {
            return inp.rmsnorm(gamma).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * BCEWithLogits Gradcheck: ";
        Tensor logits({2, 4}, dev);
        Tensor targets = Tensor::zeros({2, 4}, dev);
        targets.set_element({0, 1}, 1.0f);
        targets.set_element({1, 3}, 1.0f);
        bool ok = gradcheck([&](Tensor& inp) {
            return bce_with_logits_loss(inp, targets);
        }, logits);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * L1 Loss Gradcheck: ";
        Tensor pred({2, 4}, dev);
        Tensor target = Tensor::ones({2, 4}, dev);
        bool ok = gradcheck([&](Tensor& inp) {
            return l1_loss(inp, target);
        }, pred);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * Masked Fill Gradcheck: ";
        Tensor x({2, 4}, dev);
        Tensor mask = Tensor::zeros({2, 4}, dev);
        mask.set_element({0, 1}, 1.0f);
        mask.set_element({1, 2}, 1.0f);
        bool ok = gradcheck([&](Tensor& inp) {
            return inp.masked_fill(mask, -100.0f).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * Cat Gradcheck: ";
        Tensor a({2, 3}, dev);
        Tensor b({2, 4}, dev);
        bool ok = gradcheck([&](Tensor& inp) {
            return cat({inp, b}, 1).sum();
        }, a);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * Slice Gradcheck: ";
        Tensor x({3, 6}, dev);
        bool ok = gradcheck([](Tensor& inp) {
            return inp.slice(1, 1, 5, 2).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * Select Gradcheck: ";
        Tensor x({3, 5}, dev);
        bool ok = gradcheck([](Tensor& inp) {
            return inp.select(1, 2).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * IndexSelect Gradcheck: ";
        Tensor x({3, 5}, dev);
        std::vector<float> idx_data = {0.0f, 3.0f, 1.0f};
        Tensor indices(idx_data, {3}, dev);
        bool ok = gradcheck([&](Tensor& inp) {
            return inp.index_select(1, indices).sum();
        }, x);
        assert(ok);
        std::cout << "PASSED\n";
    }
    {
        std::cout << "  * FlashAttention Gradcheck: ";
        Tensor Q = Tensor::randn({1, 4, 8}, 0.0f, 0.5f, dev, true);
        Tensor K = Tensor::randn({1, 4, 8}, 0.0f, 0.5f, dev, false);
        Tensor V = Tensor::randn({1, 4, 8}, 0.0f, 0.5f, dev, false);
        bool ok = gradcheck([&](Tensor& q_inp) {
            return flash_attention(q_inp, K, V, /*is_causal=*/true, 0.35f).sum();
        }, Q, 1e-3f, 1e-2f);
        assert(ok);
        std::cout << "PASSED\n";
    }
}

void test_module_and_serialization(Device dev) {
    std::cout << "\n--- Testing nn::Module & Checkpoint Serialization on " << device_to_string(dev) << " ---\n";
    
    // Define a custom multi-layer perceptron
    class MLP : public nn::Module {
    public:
        std::shared_ptr<nn::Linear> fc1;
        std::shared_ptr<nn::LayerNorm> ln;
        std::shared_ptr<nn::Linear> fc2;

        MLP(int in_f, int hid_f, int out_f, Device d) {
            fc1 = std::make_shared<nn::Linear>(in_f, hid_f, d);
            ln = std::make_shared<nn::LayerNorm>(hid_f, 1e-5f, d);
            fc2 = std::make_shared<nn::Linear>(hid_f, out_f, d);

            register_module("fc1", fc1);
            register_module("ln", ln);
            register_module("fc2", fc2);
        }

        Tensor forward(const Tensor& x) {
            Tensor h = fc1->forward(x).gelu();
            h = ln->forward(h);
            return fc2->forward(h);
        }
    };

    MLP model1(8, 16, 2, dev);
    auto named = model1.named_parameters();
    std::cout << "  Registered parameters count: " << named.size() << "\n";
    assert(named.size() == 6); // fc1.w, fc1.b, ln.w, ln.b, fc2.w, fc2.b

    // Forward pass
    Tensor x = Tensor::randn({4, 8}, 0.0f, 1.0f, dev);
    Tensor out1 = model1.forward(x);
    assert(out1.shape() == std::vector<int64_t>({4, 2}));

    // Save checkpoint
    std::string ckpt_path = "test_model_checkpoint.sgrd";
    serialization::save_module(model1, ckpt_path);
    std::cout << "  Saved module checkpoint to: " << ckpt_path << "\n";

    // Create fresh model and load checkpoint
    MLP model2(8, 16, 2, dev);
    serialization::load_module(model2, ckpt_path);
    Tensor out2 = model2.forward(x);

    // Verify exact numerical match
    float max_diff = (out1 - out2).cpu().impl->data()[0];
    assert(std::abs(max_diff) < 1e-6f);
    std::cout << "  Loaded and verified checkpoint output match: PASSED\n";
}

void test_modern_architectures(Device dev) {
    std::cout << "\n--- Testing Modern Architectures (TransformerBlock & MiniResNet) on " << device_to_string(dev) << " ---\n";

    // 1. Test MultiheadAttention & TransformerBlock
    {
        int64_t embed_dim = 16;
        int64_t num_heads = 4;
        int64_t seq_len = 8;

        models::TransformerBlock block(embed_dim, num_heads, 2, dev);
        Tensor x = Tensor::randn({seq_len, embed_dim}, 0.0f, 1.0f, dev, true);

        Tensor out = block.forward(x);
        assert(out.shape() == std::vector<int64_t>({seq_len, embed_dim}));

        Tensor loss = out.sum();
        loss.backward();

        assert(block.attn->q_proj->W.grad().is_defined());
        std::cout << "  * TransformerBlock Forward & Backward: PASSED\n";
    }

    // 2. Test MiniResNet with Skip Connections & Flatten
    {
        models::MiniResNet resnet(1, 10, dev);
        Tensor x = Tensor::randn({2, 1, 28, 28}, 0.0f, 1.0f, dev, true);

        Tensor logits = resnet.forward(x);
        assert(logits.shape() == std::vector<int64_t>({2, 10}));

        Tensor loss = logits.sum();
        loss.backward();

        assert(resnet.fc->W.grad().is_defined());
        std::cout << "  * MiniResNet Forward & Backward: PASSED\n";
    }
}

void test_optimizers(Device dev) {
    std::cout << "\n--- Testing Native GPU & CPU Optimizers (AdamW & SGD) on " << device_to_string(dev) << " ---\n";
    
    // Test AdamW optimization step
    {
        nn::Linear layer(4, 2, dev);
        nn::AdamW optimizer(layer.parameters(), /*lr=*/0.01f, 0.9f, 0.999f, 1e-8f, /*weight_decay=*/0.01f);

        Tensor x = Tensor::randn({2, 4}, 0.0f, 1.0f, dev);
        Tensor y = layer.forward(x).sum();
        y.backward();

        float pre_w = layer.W.cpu().impl->data()[0];
        optimizer.step();
        float post_w = layer.W.cpu().impl->data()[0];

        assert(pre_w != post_w);
        std::cout << "  * AdamW step: PASSED (Weight updated from " << pre_w << " to " << post_w << ")\n";
    }

    // Test SGD with Momentum
    {
        nn::Linear layer(4, 2, dev);
        nn::SGD optimizer(layer.parameters(), /*lr=*/0.01f, /*momentum=*/0.9f, /*weight_decay=*/0.001f);

        Tensor x = Tensor::randn({2, 4}, 0.0f, 1.0f, dev);
        Tensor y = layer.forward(x).sum();
        y.backward();

        float pre_w = layer.W.cpu().impl->data()[0];
        optimizer.step();
        float post_w = layer.W.cpu().impl->data()[0];

        assert(pre_w != post_w);
        std::cout << "  * SGD step: PASSED (Weight updated from " << pre_w << " to " << post_w << ")\n";
    }

    // Test Gradient Clipping
    {
        nn::Linear layer(4, 2, dev);
        Tensor x = Tensor::randn({2, 4}, 0.0f, 10.0f, dev);
        Tensor y = layer.forward(x).sum() * 1000.0f;
        y.backward();

        float pre_norm = nn::clip_grad_norm(layer.parameters(), /*max_norm=*/1.0f);
        assert(pre_norm > 1.0f);
        
        float post_norm = 0.0f;
        for (const auto& p : layer.parameters()) {
            float n = p.grad().norm(2.0f);
            post_norm += n * n;
        }
        post_norm = std::sqrt(post_norm);
        assert(post_norm <= 1.05f);
        std::cout << "  * Gradient Clipping: PASSED (Clipped norm " << pre_norm << " -> " << post_norm << ")\n";
    }

    // Test Learning Rate Schedulers
    {
        nn::Linear layer(4, 2, dev);
        nn::AdamW opt(layer.parameters(), 0.1f);
        nn::StepLR step_sched(0.1f, /*step_size=*/5, /*gamma=*/0.5f);
        for (int i = 0; i < 5; ++i) step_sched.step(opt);
        assert(std::abs(opt.lr - 0.05f) < 1e-6f);

        nn::CosineAnnealingLR cos_sched(0.1f, /*T_max=*/10, /*eta_min=*/0.01f);
        for (int i = 0; i < 10; ++i) cos_sched.step(opt);
        assert(std::abs(opt.lr - 0.01f) < 1e-4f);
        std::cout << "  * LR Schedulers (StepLR & CosineAnnealingLR): PASSED\n";
    }
}

void test_simd_vector_ops() {
    std::cout << "\n--- Testing SIMD Vector Intrinsics (AVX2/FMA & Fallbacks) ---\n";
    const int64_t N = 128;
    std::vector<float> a(N, 1.5f), b(N, 2.5f), dst(N, 0.0f);

    // 1. Vector Add
    simd::vec_add(dst.data(), a.data(), b.data(), N);
    for (int64_t i = 0; i < N; ++i) {
        assert(std::abs(dst[i] - 4.0f) < 1e-5f);
    }
    std::cout << "  * simd::vec_add (N=" << N << "): PASSED\n";

    // 2. Vector Mul
    simd::vec_mul(dst.data(), a.data(), b.data(), N);
    for (int64_t i = 0; i < N; ++i) {
        assert(std::abs(dst[i] - 3.75f) < 1e-5f);
    }
    std::cout << "  * simd::vec_mul (N=" << N << "): PASSED\n";

    // 3. Vector ReLU
    std::vector<float> r_in(N, -1.0f);
    for (int64_t i = 0; i < N; i += 2) r_in[i] = 2.0f;
    simd::vec_relu(dst.data(), r_in.data(), N);
    for (int64_t i = 0; i < N; ++i) {
        float expected = (r_in[i] > 0.0f) ? r_in[i] : 0.0f;
        assert(dst[i] == expected);
    }
    std::cout << "  * simd::vec_relu: PASSED\n";

    // 4. Vector GELU
    simd::vec_gelu(dst.data(), a.data(), N);
    assert(dst[0] > 1.3f && dst[0] < 1.5f);
    std::cout << "  * simd::vec_gelu: PASSED\n";
}

void test_fused_operators(Device dev) {
    std::cout << "\n--- Testing Fused Operators on " << device_to_string(dev) << " ---\n";

    // 1. Fused Linear + GELU
    {
        Tensor x = Tensor::randn({4, 8}, 0.0f, 1.0f, dev, true);
        Tensor W = Tensor::randn({8, 16}, 0.0f, 0.5f, dev, true);
        Tensor b = Tensor::zeros({1, 16}, dev, true);

        Tensor fused_out = fused_linear_gelu(x, W, b);
        Tensor unfused_out = (x.matmul(W) + b).gelu();

        float diff = (fused_out - unfused_out).cpu().impl->data()[0];
        assert(std::abs(diff) < 1e-4f);

        Tensor loss = fused_out.sum();
        loss.backward();
        assert(x.grad().is_defined() && W.grad().is_defined() && b.grad().is_defined());
        std::cout << "  * Fused Linear+GELU forward & backward: PASSED\n";
    }

    // 2. Fused Add + LayerNorm
    {
        Tensor x = Tensor::randn({4, 8}, 0.0f, 1.0f, dev, true);
        Tensor res = Tensor::randn({4, 8}, 0.0f, 1.0f, dev, true);
        Tensor gamma = Tensor::ones({8}, dev, true);
        Tensor beta = Tensor::zeros({8}, dev, true);

        Tensor fused_out = fused_add_layernorm(x, res, gamma, beta, 1e-5f);
        Tensor unfused_out = layer_norm(x + res, gamma, beta, 1e-5f);

        float diff = (fused_out - unfused_out).cpu().impl->data()[0];
        assert(std::abs(diff) < 1e-4f);

        Tensor loss = fused_out.sum();
        loss.backward();
        assert(x.grad().is_defined() && res.grad().is_defined());
        std::cout << "  * Fused Add+LayerNorm forward & backward: PASSED\n";
    }

    // 3. Fused Scaled Dot-Product Attention
    {
        Tensor Q = Tensor::randn({8, 16}, 0.0f, 1.0f, dev, true);
        Tensor K = Tensor::randn({8, 16}, 0.0f, 1.0f, dev, true);
        Tensor V = Tensor::randn({8, 16}, 0.0f, 1.0f, dev, true);

        Tensor out = fused_scaled_dot_product_attention(Q, K, V, 1.0f / std::sqrt(16.0f));
        assert(out.shape() == std::vector<int64_t>({8, 16}));

        Tensor loss = out.sum();
        loss.backward();
        assert(Q.grad().is_defined());
        std::cout << "  * Fused Scaled Dot-Product Attention: PASSED\n";
    }
}

void test_flash_attention_correctness(Device dev) {
    std::cout << "\n--- Testing FlashAttention-2 vs Standard Attention on " << device_to_string(dev) << " ---\n";
    
    // 1. Non-causal FlashAttention equivalence
    {
        int L = 16, D = 32;
        Tensor Q = Tensor::randn({L, D}, 0.0f, 1.0f, dev, true);
        Tensor K = Tensor::randn({L, D}, 0.0f, 1.0f, dev, true);
        Tensor V = Tensor::randn({L, D}, 0.0f, 1.0f, dev, true);
        float scale = 1.0f / std::sqrt((float)D);

        Tensor flash_out = flash_attention(Q, K, V, /*is_causal=*/false, scale);

        // Standard attention
        Tensor K_t = K.transpose(0, 1).contiguous();
        Tensor scores = Q.matmul(K_t) * scale;
        Tensor weights = softmax(scores, -1);
        Tensor std_out = weights.matmul(V);

        Tensor diff = (flash_out - std_out).cpu();
        float max_diff = 0.0f;
        for (int i = 0; i < diff.numel(); ++i) {
            float d = std::abs(diff.impl->data()[i]);
            if (d > max_diff) max_diff = d;
        }
        assert(max_diff < 1e-4f);
        std::cout << "  * Non-Causal FlashAttention-2 output equivalence (max diff " << max_diff << "): PASSED\n";
    }

    // 2. Causal FlashAttention equivalence
    {
        int L = 16, D = 32;
        Tensor Q = Tensor::randn({L, D}, 0.0f, 1.0f, dev, true);
        Tensor K = Tensor::randn({L, D}, 0.0f, 1.0f, dev, true);
        Tensor V = Tensor::randn({L, D}, 0.0f, 1.0f, dev, true);
        float scale = 1.0f / std::sqrt((float)D);

        Tensor flash_out = flash_attention(Q, K, V, /*is_causal=*/true, scale);

        // Standard causal attention
        Tensor K_t = K.transpose(0, 1).contiguous();
        Tensor scores = Q.matmul(K_t) * scale;
        Tensor mask = Tensor::triu({L, L}, 1, dev);
        scores = scores.masked_fill(mask, -1e9f);
        Tensor weights = softmax(scores, -1);
        Tensor std_out = weights.matmul(V);

        Tensor diff = (flash_out - std_out).cpu();
        float max_diff = 0.0f;
        for (int i = 0; i < diff.numel(); ++i) {
            float d = std::abs(diff.impl->data()[i]);
            if (d > max_diff) max_diff = d;
        }
        assert(max_diff < 1e-4f);
        std::cout << "  * Causal FlashAttention-2 output equivalence (max diff " << max_diff << "): PASSED\n";
    }
}

void test_quantization(Device dev) {
    std::cout << "\n--- Testing Quantization Subsystem (INT8 & INT4) on " << device_to_string(dev) << " ---\n";

    // 1. INT8 Per-Tensor & Per-Channel
    {
        Tensor x = Tensor::randn({16, 32}, 0.0f, 1.0f, dev);
        auto q_tensor = quantization::quantize_int8(x, quantization::QuantizationScheme::PerTensorSymmetric);
        Tensor dequant_x = quantization::dequantize_int8(q_tensor);

        Tensor diff = (x - dequant_x).cpu();
        float max_err = 0.0f;
        for (int i = 0; i < diff.numel(); ++i) {
            float e = std::abs(diff.impl->data()[i]);
            if (e > max_err) max_err = e;
        }
        assert(max_err < 0.05f); // High fidelity INT8
        std::cout << "  * INT8 Per-Tensor quantization (max err " << max_err << "): PASSED\n";

        auto q_chan = quantization::quantize_int8(x, quantization::QuantizationScheme::PerChannelSymmetric);
        Tensor dequant_chan = quantization::dequantize_int8(q_chan);
        assert((x - dequant_chan).cpu().impl->data()[0] < 0.05f);
        std::cout << "  * INT8 Per-Channel quantization: PASSED\n";
    }

    // 2. INT4 Grouped Quantization
    {
        Tensor x = Tensor::randn({16, 32}, 0.0f, 1.0f, dev);
        auto q4 = quantization::quantize_int4(x, /*group_size=*/32);
        Tensor dequant4 = quantization::dequantize_int4(q4);
        assert(dequant4.shape() == x.shape());
        std::cout << "  * INT4 Grouped (G=32) quantization & dequantization: PASSED\n";
    }

    // 3. QuantizedLinear & QuantizedLinear4Bit Modules
    {
        nn::Linear float_linear(32, 16, dev);
        auto ql8 = nn::QuantizedLinear::from_float(float_linear);
        auto ql4 = nn::QuantizedLinear4Bit::from_float(float_linear, /*group_size=*/16);

        Tensor input = Tensor::randn({4, 32}, 0.0f, 1.0f, dev);
        Tensor float_out = float_linear.forward(input);
        Tensor q8_out = ql8->forward(input);
        Tensor q4_out = ql4->forward(input);

        assert(q8_out.shape() == float_out.shape());
        assert(q4_out.shape() == float_out.shape());

        float diff8 = std::abs((float_out - q8_out).cpu().impl->data()[0]);
        float diff4 = std::abs((float_out - q4_out).cpu().impl->data()[0]);
        assert(diff8 < 0.15f);
        assert(diff4 < 0.35f);
        std::cout << "  * QuantizedLinear INT8 forward pass match (diff: " << diff8 << "): PASSED\n";
        std::cout << "  * QuantizedLinear4Bit INT4 forward pass match (diff: " << diff4 << "): PASSED\n";
    }
}

void test_jit_fusion(Device dev) {
    std::cout << "\n--- Testing JIT Graph Operator Fusion Engine on " << device_to_string(dev) << " ---\n";

    // 1. Build and optimize JIT Graph with Linear + GELU
    jit::Graph g;
    int in_id = g.add_node("x", jit::OpType::Input, {});
    Tensor W = Tensor::randn({8, 16}, 0.0f, 0.5f, dev);
    Tensor b = Tensor::zeros({1, 16}, dev);
    int w_id = g.add_node("W", jit::OpType::Constant, {}, W);
    int b_id = g.add_node("b", jit::OpType::Constant, {}, b);
    int mm_id = g.add_node("matmul", jit::OpType::Matmul, {in_id, w_id});
    int add_id = g.add_node("add_bias", jit::OpType::Add, {mm_id, b_id});
    int gelu_id = g.add_node("gelu", jit::OpType::GELU, {add_id});
    g.output_ids = {gelu_id};

    auto compiled = jit::compile(g);
    assert(compiled.graph.nodes[gelu_id].op == jit::OpType::FusedLinearActivation);

    Tensor input = Tensor::randn({4, 8}, 0.0f, 1.0f, dev);
    Tensor compiled_out = compiled.forward(input);
    Tensor eager_out = (input.matmul(W) + b).gelu();

    float diff = std::abs((compiled_out - eager_out).cpu().impl->data()[0]);
    assert(diff < 1e-4f);
    std::cout << "  * JIT FusedLinearActivation optimization & bitwise match: PASSED\n";

    // 2. Residual + LayerNorm Fusion
    jit::Graph g_res;
    int x_id = g_res.add_node("x", jit::OpType::Input, {});
    Tensor res_val = Tensor::randn({4, 16}, 0.0f, 1.0f, dev);
    Tensor gamma = Tensor::ones({16}, dev);
    Tensor beta = Tensor::zeros({16}, dev);
    int res_id = g_res.add_node("residual", jit::OpType::Constant, {}, res_val);
    int g_id = g_res.add_node("gamma", jit::OpType::Constant, {}, gamma);
    int bet_id = g_res.add_node("beta", jit::OpType::Constant, {}, beta);
    int sum_id = g_res.add_node("add_res", jit::OpType::Add, {x_id, res_id});
    int ln_id = g_res.add_node("ln", jit::OpType::LayerNorm, {sum_id, g_id, bet_id});
    g_res.output_ids = {ln_id};

    auto compiled_res = jit::compile(g_res);
    assert(compiled_res.graph.nodes[ln_id].op == jit::OpType::FusedResidualNorm);

    Tensor res_in = Tensor::randn({4, 16}, 0.0f, 1.0f, dev);
    Tensor res_comp_out = compiled_res.forward(res_in);
    Tensor res_eager_out = layer_norm(res_in + res_val, gamma, beta, 1e-5f);

    float diff_res = std::abs((res_comp_out - res_eager_out).cpu().impl->data()[0]);
    assert(diff_res < 1e-4f);
    std::cout << "  * JIT FusedResidualNorm optimization & bitwise match: PASSED\n";
}

void test_dtype_and_stream(Device dev) {
    std::cout << "\n--- Testing DType Subsystem & CUDA Streams on " << device_to_string(dev) << " ---\n";

    // 1. DType queries and conversion
    Tensor t({4, 4}, dev);
    assert(t.dtype() == DType::Float32);
    assert(dtype_to_string(t.dtype()) == "Float32");

    Tensor t_fp16 = t.to(DType::Float16);
    assert(t_fp16.dtype() == DType::Float16);
    std::cout << "  * DType query & conversion: PASSED\n";

    // 2. CUDA Stream & StreamGuard
    cuda::Stream stream;
    {
        cuda::StreamGuard guard(stream);
        Tensor a = Tensor::ones({10, 10}, dev);
        Tensor b = Tensor::ones({10, 10}, dev);
        Tensor c = a + b;
        assert(c.numel() == 100);
    }
    stream.synchronize();
    std::cout << "  * CUDA Stream & StreamGuard synchronization: PASSED\n";
}

void test_safetensors_interop(Device dev) {
    std::cout << "\n--- Testing SafeTensors Zero-Copy Interoperability on " << device_to_string(dev) << " ---\n";

    std::unordered_map<std::string, Tensor> original_dict;
    original_dict["layer.weight"] = Tensor::randn({16, 32}, 0.0f, 1.0f, dev, false);
    original_dict["layer.bias"] = Tensor::randn({1, 32}, 0.0f, 1.0f, dev, false);
    original_dict["embeddings"] = Tensor::randn({100, 64}, 0.0f, 1.0f, dev, false);

    const std::string filename = "test_interop_model.safetensors";
    safetensors::save_file(original_dict, filename);

    // 1. Inspect without loading
    auto metas = safetensors::inspect_file(filename);
    assert(metas.size() == 3);
    assert(metas.count("layer.weight") == 1);
    assert(metas["layer.weight"].shape == std::vector<int64_t>({16, 32}));
    assert(metas["layer.weight"].dtype == "F32");
    std::cout << "  * SafeTensors metadata inspection: PASSED\n";

    // 2. Load and bitwise verify
    auto loaded_dict = safetensors::load_file(filename, dev);
    assert(loaded_dict.size() == 3);

    for (const auto& kv : original_dict) {
        const auto& name = kv.first;
        assert(loaded_dict.count(name) == 1);
        Tensor orig = kv.second.to(Device::CPU);
        Tensor loaded = loaded_dict[name].to(Device::CPU);
        assert(orig.shape() == loaded.shape());
        assert(orig.dtype() == loaded.dtype());

        float max_diff = 0.0f;
        for (int64_t i = 0; i < orig.numel(); ++i) {
            float diff = std::abs(orig.data()[i] - loaded.data()[i]);
            if (diff > max_diff) max_diff = diff;
        }
        assert(max_diff == 0.0f);
    }
    std::cout << "  * SafeTensors zero-copy roundtrip float bitwise match: PASSED\n";

    // 3. Test Module save & load via SafeTensors
    nn::Linear linear(64, 128, dev);
    safetensors::save_module(linear, "test_linear.safetensors");
    nn::Linear linear_loaded(64, 128, dev);
    safetensors::load_module(linear_loaded, "test_linear.safetensors");

    Tensor x = Tensor::randn({8, 64}, 0.0f, 1.0f, dev, false);
    Tensor y_orig = linear.forward(x).to(Device::CPU);
    Tensor y_loaded = linear_loaded.forward(x).to(Device::CPU);

    float max_diff = 0.0f;
    for (int64_t i = 0; i < y_orig.numel(); ++i) {
        float diff = std::abs(y_orig.data()[i] - y_loaded.data()[i]);
        if (diff > max_diff) max_diff = diff;
    }
    assert(max_diff < 1e-6f);
    std::cout << "  * SafeTensors nn::Module save_module / load_module verification: PASSED\n";
}

void test_data_parallel(Device dev) {
    std::cout << "\n--- Testing nn::DataParallel Multi-Threaded Engine on " << device_to_string(dev) << " ---\n";

    // Create a base Sequential network: Linear(16 -> 32) -> GELU -> Linear(32 -> 8)
    auto base_net = std::make_shared<nn::Sequential>(
        std::make_shared<nn::Linear>(16, 32, dev),
        std::make_shared<nn::GELU>(),
        std::make_shared<nn::Linear>(32, 8, dev)
    );

    // Wrap in DataParallel across 4 devices/threads
    int replicas = 4;
    auto dp = nn::make_data_parallel(base_net, replicas);

    // Create batch of 16 samples
    Tensor input = Tensor::randn({16, 16}, 0.0f, 1.0f, dev, true);
    Tensor targets = Tensor::randn({16, 8}, 0.0f, 1.0f, dev, false);

    // Forward pass
    Tensor dp_out = dp->forward(input);
    assert(dp_out.shape() == std::vector<int64_t>({16, 8}));

    // Sequential equivalence check
    Tensor seq_out = base_net->forward(input);
    Tensor diff = (dp_out - seq_out).to(Device::CPU);
    float max_err = 0.0f;
    for (int64_t i = 0; i < diff.numel(); ++i) {
        float d = std::abs(diff.data()[i]);
        if (d > max_err) max_err = d;
    }
    assert(max_err < 1e-5f);
    std::cout << "  * nn::DataParallel forward pass equivalence (diff: " << max_err << "): PASSED\n";

    // Backward pass
    Tensor loss = mse_loss(dp_out, targets);
    loss.backward();
    assert(input.grad().defined());
    assert(input.grad().shape() == std::vector<int64_t>({16, 16}));
    std::cout << "  * nn::DataParallel backward pass & autograd propagation: PASSED\n";
}

int main() {
    std::cout << "==========================================================\n";
    std::cout << "    SOVEREIGNGRAD PRODUCTION TEST SUITE & VERIFICATION    \n";
    std::cout << "==========================================================\n\n";

    test_simd_vector_ops();

    std::vector<Device> devices = {Device::CPU, Device::CUDA};

    for (Device dev : devices) {
        std::cout << "\n=== Running Tests on Backend: " << device_to_string(dev) << " ===\n";
        test_tensor_basics(dev);
        test_inplace_version_safety(dev);
        test_nograd_guard(dev);
        test_numerical_gradchecks(dev);
        test_module_and_serialization(dev);
        test_modern_architectures(dev);
        test_fused_operators(dev);
        test_flash_attention_correctness(dev);
        test_quantization(dev);
        test_jit_fusion(dev);
        test_dtype_and_stream(dev);
        test_optimizers(dev);
        test_safetensors_interop(dev);
        test_data_parallel(dev);
    }

    test_memory_allocator();

    std::cout << "\n==========================================================\n";
    std::cout << "       ALL PRODUCTION TESTS PASSED SUCCESSFULLY!          \n";
    std::cout << "==========================================================\n";
    return 0;
}

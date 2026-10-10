#pragma once

#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/nn.hpp"
#include <memory>
#include <vector>
#include <cmath>
#include <random>

namespace sovereigngrad {
namespace models {

// -------------------------------------------------------------
// 1. RESIDUAL CONVOLUTIONAL VISION NETWORK (DeepResNet)
// -------------------------------------------------------------
class ResNetBlock : public nn::Module {
public:
    std::shared_ptr<nn::Conv2d> conv1;
    std::shared_ptr<nn::BatchNorm2d> bn1;
    std::shared_ptr<nn::Conv2d> conv2;
    std::shared_ptr<nn::BatchNorm2d> bn2;
    std::shared_ptr<nn::Conv2d> shortcut_conv;
    bool downsample = false;

    ResNetBlock(int64_t in_channels, int64_t out_channels, int stride = 1, Device device = Device::CPU) {
        conv1 = std::make_shared<nn::Conv2d>(in_channels, out_channels, 3, stride, 1, device);
        bn1 = std::make_shared<nn::BatchNorm2d>(out_channels, 0.1f, 1e-5f, device);
        conv2 = std::make_shared<nn::Conv2d>(out_channels, out_channels, 3, 1, 1, device);
        bn2 = std::make_shared<nn::BatchNorm2d>(out_channels, 0.1f, 1e-5f, device);

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);

        if (stride != 1 || in_channels != out_channels) {
            downsample = true;
            shortcut_conv = std::make_shared<nn::Conv2d>(in_channels, out_channels, 1, stride, 0, device);
            register_module("shortcut_conv", shortcut_conv);
        }
    }

    Tensor forward(const Tensor& x) {
        Tensor residual = x;
        if (downsample) {
            residual = shortcut_conv->forward(x);
        }

        Tensor out = bn1->forward(conv1->forward(x)).relu();
        out = bn2->forward(conv2->forward(out));
        out = (out + residual).relu();
        return out;
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

class DeepResNet : public nn::Module {
public:
    std::shared_ptr<nn::Conv2d> stem_conv;
    std::shared_ptr<nn::BatchNorm2d> stem_bn;
    std::shared_ptr<ResNetBlock> layer1;
    std::shared_ptr<ResNetBlock> layer2;
    std::shared_ptr<ResNetBlock> layer3;
    std::shared_ptr<nn::AvgPool2d> pool;
    std::shared_ptr<nn::Linear> fc;

    DeepResNet(int64_t in_channels, int64_t num_classes, Device device = Device::CPU) {
        // Stem: [B, C, 28, 28] -> [B, 16, 28, 28]
        stem_conv = std::make_shared<nn::Conv2d>(in_channels, 16, 3, 1, 1, device);
        stem_bn = std::make_shared<nn::BatchNorm2d>(16, 0.1f, 1e-5f, device);
        
        // Stage 1: [B, 16, 28, 28] -> [B, 16, 28, 28]
        layer1 = std::make_shared<ResNetBlock>(16, 16, 1, device);
        
        // Stage 2: [B, 16, 28, 28] -> [B, 32, 14, 14]
        layer2 = std::make_shared<ResNetBlock>(16, 32, 2, device);
        
        // Stage 3: [B, 32, 14, 14] -> [B, 64, 7, 7]
        layer3 = std::make_shared<ResNetBlock>(32, 64, 2, device);
        
        // Global/Spatial Pool: [B, 64, 7, 7] -> [B, 64, 3, 3] -> 64 * 3 * 3 = 576
        pool = std::make_shared<nn::AvgPool2d>(2, 2, 0);
        fc = std::make_shared<nn::Linear>(576, num_classes, device);

        register_module("stem_conv", stem_conv);
        register_module("stem_bn", stem_bn);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("pool", pool);
        register_module("fc", fc);
    }

    Tensor forward(const Tensor& x) {
        Tensor h = stem_bn->forward(stem_conv->forward(x)).relu();
        h = layer1->forward(h);
        h = layer2->forward(h);
        h = layer3->forward(h);
        h = pool->forward(h);
        Tensor flat = h.flatten(1, -1);
        return fc->forward(flat);
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

using MiniResNet = DeepResNet;

// -------------------------------------------------------------
// STANDARD TRANSFORMER BLOCK & MINIGPT
// -------------------------------------------------------------
class TransformerBlock : public nn::Module {
public:
    std::shared_ptr<nn::LayerNorm> ln1;
    std::shared_ptr<nn::MultiheadAttention> attn;
    std::shared_ptr<nn::LayerNorm> ln2;
    std::shared_ptr<nn::Linear> mlp_fc1;
    std::shared_ptr<nn::Linear> mlp_fc2;

    TransformerBlock(int64_t embed_dim, int64_t num_heads, int64_t mlp_ratio = 4, Device device = Device::CPU) {
        ln1 = std::make_shared<nn::LayerNorm>(embed_dim, 1e-5f, device);
        attn = std::make_shared<nn::MultiheadAttention>(embed_dim, num_heads, device);
        ln2 = std::make_shared<nn::LayerNorm>(embed_dim, 1e-5f, device);
        mlp_fc1 = std::make_shared<nn::Linear>(embed_dim, embed_dim * mlp_ratio, device);
        mlp_fc2 = std::make_shared<nn::Linear>(embed_dim * mlp_ratio, embed_dim, device);

        register_module("ln1", ln1);
        register_module("attn", attn);
        register_module("ln2", ln2);
        register_module("mlp_fc1", mlp_fc1);
        register_module("mlp_fc2", mlp_fc2);
    }

    Tensor forward(const Tensor& x) {
        Tensor norm_x1 = ln1->forward(x);
        Tensor h = x + attn->forward(norm_x1);
        Tensor norm_x2 = ln2->forward(h);
        Tensor mlp_out = mlp_fc2->forward(mlp_fc1->forward(norm_x2).gelu());
        return h + mlp_out;
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

class MiniGPT : public nn::Module {
public:
    int64_t vocab_size;
    int64_t max_seq_len;
    int64_t embed_dim;

    std::shared_ptr<nn::Embedding> token_embeddings;
    std::shared_ptr<nn::Embedding> position_embeddings;
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    std::shared_ptr<nn::LayerNorm> ln_f;
    std::shared_ptr<nn::Linear> lm_head;

    MiniGPT(int64_t vocab_size, int64_t max_seq_len, int64_t embed_dim, 
            int64_t num_layers, int64_t num_heads, Device device = Device::CPU)
        : vocab_size(vocab_size), max_seq_len(max_seq_len), embed_dim(embed_dim) {
        
        token_embeddings = std::make_shared<nn::Embedding>(vocab_size, embed_dim, device);
        position_embeddings = std::make_shared<nn::Embedding>(max_seq_len, embed_dim, device);
        ln_f = std::make_shared<nn::LayerNorm>(embed_dim, 1e-5f, device);
        lm_head = std::make_shared<nn::Linear>(embed_dim, vocab_size, device);

        register_module("token_embeddings", token_embeddings);
        register_module("position_embeddings", position_embeddings);

        for (int64_t i = 0; i < num_layers; ++i) {
            auto block = std::make_shared<TransformerBlock>(embed_dim, num_heads, 4, device);
            blocks.push_back(block);
            register_module("block_" + std::to_string(i), block);
        }

        register_module("ln_f", ln_f);
        register_module("lm_head", lm_head);
    }

    Tensor forward(const Tensor& token_ids, const Tensor& pos_ids) {
        Tensor tok_emb = token_embeddings->forward(token_ids);
        Tensor pos_emb = position_embeddings->forward(pos_ids);
        Tensor h = tok_emb + pos_emb;

        for (auto& block : blocks) {
            h = block->forward(h);
        }

        h = ln_f->forward(h);
        return lm_head->forward(h);
    }
};

// -------------------------------------------------------------
// 2. MODERN AUTOREGRESSIVE LLM (NanoLLaMA)
// -------------------------------------------------------------
class LLaMABlock : public nn::Module {
public:
    std::shared_ptr<nn::RMSNorm> norm1;
    std::shared_ptr<nn::CausalSelfAttention> attn;
    std::shared_ptr<nn::RMSNorm> norm2;
    std::shared_ptr<nn::SwiGLU> mlp;

    LLaMABlock(int64_t embed_dim, int64_t num_heads, int64_t hidden_dim, Device device = Device::CPU) {
        norm1 = std::make_shared<nn::RMSNorm>(embed_dim, 1e-5f, device);
        attn = std::make_shared<nn::CausalSelfAttention>(embed_dim, num_heads, device);
        norm2 = std::make_shared<nn::RMSNorm>(embed_dim, 1e-5f, device);
        mlp = std::make_shared<nn::SwiGLU>(embed_dim, hidden_dim, device);

        register_module("norm1", norm1);
        register_module("attn", attn);
        register_module("norm2", norm2);
        register_module("mlp", mlp);
    }

    Tensor forward(const Tensor& x) {
        // Pre-RMSNorm Causal Attention with Residual Connection
        Tensor h = x + attn->forward(norm1->forward(x));
        // Pre-RMSNorm SwiGLU Feed-Forward with Residual Connection
        return h + mlp->forward(norm2->forward(h));
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

class NanoLLaMA : public nn::Module {
public:
    int64_t vocab_size;
    int64_t max_seq_len;
    int64_t embed_dim;

    std::shared_ptr<nn::Embedding> token_embeddings;
    std::vector<std::shared_ptr<LLaMABlock>> layers;
    std::shared_ptr<nn::RMSNorm> norm_f;
    std::shared_ptr<nn::Linear> lm_head;

    NanoLLaMA(int64_t vocab_size, int64_t max_seq_len, int64_t embed_dim,
              int64_t num_layers, int64_t num_heads, int64_t hidden_dim, Device device = Device::CPU)
        : vocab_size(vocab_size), max_seq_len(max_seq_len), embed_dim(embed_dim) {
        
        token_embeddings = std::make_shared<nn::Embedding>(vocab_size, embed_dim, device);
        norm_f = std::make_shared<nn::RMSNorm>(embed_dim, 1e-5f, device);
        lm_head = std::make_shared<nn::Linear>(embed_dim, vocab_size, device);

        register_module("token_embeddings", token_embeddings);

        for (int64_t i = 0; i < num_layers; ++i) {
            auto block = std::make_shared<LLaMABlock>(embed_dim, num_heads, hidden_dim, device);
            layers.push_back(block);
            register_module("layer_" + std::to_string(i), block);
        }

        register_module("norm_f", norm_f);
        register_module("lm_head", lm_head);
    }

    Tensor forward(const Tensor& token_ids) {
        // token_ids: [L]
        Tensor h = token_embeddings->forward(token_ids); // [L, D]

        for (auto& layer : layers) {
            h = layer->forward(h);
        }

        h = norm_f->forward(h);
        return lm_head->forward(h); // [L, vocab_size]
    }

    // Autoregressive token generation loop
    std::vector<int64_t> generate(const std::vector<int64_t>& prompt, int max_new_tokens, float temperature = 1.0f) {
        std::vector<int64_t> result = prompt;
        
        for (int step = 0; step < max_new_tokens; ++step) {
            int64_t cur_len = (int64_t)result.size();
            std::vector<float> token_floats(result.begin(), result.end());
            Tensor tokens_t(token_floats, {cur_len}, Device::CPU);

            Tensor logits = forward(tokens_t); // [L, vocab_size]
            // Extract logits of last position
            Tensor last_logits({1, vocab_size}, Device::CPU);
            const float* log_ptr = logits.to(Device::CPU).impl->data();
            float* dst_ptr = last_logits.impl->data();
            for (int64_t v = 0; v < vocab_size; ++v) {
                dst_ptr[v] = log_ptr[(cur_len - 1) * vocab_size + v] / (temperature > 0 ? temperature : 1.0f);
            }

            Tensor probs = softmax(last_logits, -1);
            const float* prob_ptr = probs.impl->data();

            // Greedy or highest probability token selection
            int64_t best_token = 0;
            float best_p = -1.0f;
            for (int64_t v = 0; v < vocab_size; ++v) {
                if (prob_ptr[v] > best_p) {
                    best_p = prob_ptr[v];
                    best_token = v;
                }
            }
            result.push_back(best_token);
        }
        return result;
    }
};

// -------------------------------------------------------------
// 3. SEQUENCE CLASSIFIER (1D Conv + Embedding NLP / Signal)
// -------------------------------------------------------------
class SequenceClassifier : public nn::Module {
public:
    int64_t vocab_size;
    int64_t embed_dim;
    int64_t num_classes;

    std::shared_ptr<nn::Embedding> embedding;
    std::shared_ptr<nn::Conv1d> conv1;
    std::shared_ptr<nn::Conv1d> conv2;
    std::shared_ptr<nn::Dropout> dropout;
    std::shared_ptr<nn::Linear> fc;

    SequenceClassifier(int64_t vocab_size, int64_t embed_dim, int64_t num_filters,
                       int64_t num_classes, Device device = Device::CPU)
        : vocab_size(vocab_size), embed_dim(embed_dim), num_classes(num_classes) {
        
        embedding = std::make_shared<nn::Embedding>(vocab_size, embed_dim, device);
        // x: [1, embed_dim, seq_len] -> Conv1D kernel size 3
        conv1 = std::make_shared<nn::Conv1d>(embed_dim, num_filters, 3, 1, 1, device);
        conv2 = std::make_shared<nn::Conv1d>(num_filters, num_filters, 3, 1, 1, device);
        dropout = std::make_shared<nn::Dropout>(0.1f);
        fc = std::make_shared<nn::Linear>(num_filters, num_classes, device);

        register_module("embedding", embedding);
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("dropout", dropout);
        register_module("fc", fc);
    }

    Tensor forward(const Tensor& token_ids) {
        // token_ids: [L]
        Tensor emb = embedding->forward(token_ids); // [L, D]
        int64_t L = token_ids.shape()[0];
        
        // Reshape for Conv1D: [1, D, L]
        Tensor emb_t = emb.transpose(0, 1).contiguous().view({1, embed_dim, L});
        Tensor c1 = conv1->forward(emb_t).relu(); // [1, num_filters, L]
        Tensor c2 = conv2->forward(c1).relu(); // [1, num_filters, L]

        // Global temporal average pooling along L (dim 2) -> [1, num_filters]
        Tensor pooled = c2.mean(2, /*keepdim=*/false); // [1, num_filters]
        Tensor dropped = dropout->forward(pooled);
        return fc->forward(dropped); // [1, num_classes]
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

// -------------------------------------------------------------
// 4. DEEP RESIDUAL MULTI-LAYER PERCEPTRON (ResMLP)
// -------------------------------------------------------------
class ResMLPBlock : public nn::Module {
public:
    std::shared_ptr<nn::LayerNorm> ln;
    std::shared_ptr<nn::Linear> fc1;
    std::shared_ptr<nn::Linear> fc2;
    std::shared_ptr<nn::Dropout> drop;

    ResMLPBlock(int64_t dim, int64_t hidden_dim, float dropout_p = 0.1f, Device device = Device::CPU) {
        ln = std::make_shared<nn::LayerNorm>(dim, 1e-5f, device);
        fc1 = std::make_shared<nn::Linear>(dim, hidden_dim, device);
        fc2 = std::make_shared<nn::Linear>(hidden_dim, dim, device);
        drop = std::make_shared<nn::Dropout>(dropout_p);

        register_module("ln", ln);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("drop", drop);
    }

    Tensor forward(const Tensor& x) {
        Tensor h = ln->forward(x);
        h = fc1->forward(h).gelu();
        h = drop->forward(fc2->forward(h));
        return x + h;
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

class ResMLP : public nn::Module {
public:
    std::shared_ptr<nn::Linear> in_proj;
    std::vector<std::shared_ptr<ResMLPBlock>> blocks;
    std::shared_ptr<nn::LayerNorm> ln_f;
    std::shared_ptr<nn::Linear> out_proj;

    ResMLP(int64_t in_features, int64_t hidden_dim, int64_t num_layers, int64_t num_classes, Device device = Device::CPU) {
        in_proj = std::make_shared<nn::Linear>(in_features, hidden_dim, device);
        register_module("in_proj", in_proj);

        for (int64_t i = 0; i < num_layers; ++i) {
            auto b = std::make_shared<ResMLPBlock>(hidden_dim, hidden_dim * 2, 0.1f, device);
            blocks.push_back(b);
            register_module("block_" + std::to_string(i), b);
        }

        ln_f = std::make_shared<nn::LayerNorm>(hidden_dim, 1e-5f, device);
        out_proj = std::make_shared<nn::Linear>(hidden_dim, num_classes, device);

        register_module("ln_f", ln_f);
        register_module("out_proj", out_proj);
    }

    Tensor forward(const Tensor& x) {
        Tensor h = in_proj->forward(x);
        for (auto& b : blocks) {
            h = b->forward(h);
        }
        h = ln_f->forward(h);
        return out_proj->forward(h);
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

// -------------------------------------------------------------
// 5. DENOISING / BOTTLE-NECK AUTOENCODER (Autoencoder)
// -------------------------------------------------------------
class Autoencoder : public nn::Module {
public:
    std::shared_ptr<nn::Linear> enc1;
    std::shared_ptr<nn::Linear> enc2;
    std::shared_ptr<nn::Linear> dec1;
    std::shared_ptr<nn::Linear> dec2;

    Autoencoder(int64_t in_features, int64_t hidden_dim, int64_t latent_dim, Device device = Device::CPU) {
        enc1 = std::make_shared<nn::Linear>(in_features, hidden_dim, device);
        enc2 = std::make_shared<nn::Linear>(hidden_dim, latent_dim, device);
        dec1 = std::make_shared<nn::Linear>(latent_dim, hidden_dim, device);
        dec2 = std::make_shared<nn::Linear>(hidden_dim, in_features, device);

        register_module("enc1", enc1);
        register_module("enc2", enc2);
        register_module("dec1", dec1);
        register_module("dec2", dec2);
    }

    Tensor encode(const Tensor& x) {
        return enc2->forward(enc1->forward(x).gelu());
    }

    Tensor decode(const Tensor& z) {
        return dec2->forward(dec1->forward(z).gelu()).sigmoid();
    }

    Tensor forward(const Tensor& x) {
        Tensor latent = encode(x);
        return decode(latent);
    }

    Tensor operator()(const Tensor& x) { return forward(x); }
};

} // namespace models
} // namespace sovereigngrad

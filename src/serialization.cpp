#include "sovereigngrad/serialization.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdint>

namespace sovereigngrad {
namespace serialization {

static const char MAGIC[4] = {'S', 'G', 'R', 'D'};
static const uint32_t FORMAT_VERSION = 1;

void save_checkpoint(const std::unordered_map<std::string, Tensor>& state_dict, const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing checkpoint: " + filepath);
    }

    // 1. Write Header
    out.write(MAGIC, 4);
    out.write(reinterpret_cast<const char*>(&FORMAT_VERSION), sizeof(uint32_t));
    uint32_t num_tensors = static_cast<uint32_t>(state_dict.size());
    out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(uint32_t));

    // 2. Write Each Tensor
    for (const auto& [name, tensor] : state_dict) {
        // Name
        uint32_t name_len = static_cast<uint32_t>(name.size());
        out.write(reinterpret_cast<const char*>(&name_len), sizeof(uint32_t));
        out.write(name.data(), name_len);

        // Shape metadata
        Tensor contig = tensor.cpu().contiguous();
        auto shape = contig.shape();
        uint32_t rank = static_cast<uint32_t>(shape.size());
        out.write(reinterpret_cast<const char*>(&rank), sizeof(uint32_t));
        for (int64_t d : shape) {
            out.write(reinterpret_cast<const char*>(&d), sizeof(int64_t));
        }

        // Data payload
        int64_t numel = contig.numel();
        out.write(reinterpret_cast<const char*>(&numel), sizeof(int64_t));
        out.write(reinterpret_cast<const char*>(contig.impl->data()), numel * sizeof(float));
    }
}

std::unordered_map<std::string, Tensor> load_checkpoint(const std::string& filepath, Device device) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading checkpoint: " + filepath);
    }

    // 1. Validate Header
    char magic[4];
    in.read(magic, 4);
    if (std::memcmp(magic, MAGIC, 4) != 0) {
        throw std::runtime_error("Invalid checkpoint file format or corrupted magic header in: " + filepath);
    }

    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != FORMAT_VERSION) {
        throw std::runtime_error("Unsupported checkpoint format version: " + std::to_string(version));
    }

    uint32_t num_tensors = 0;
    in.read(reinterpret_cast<char*>(&num_tensors), sizeof(uint32_t));

    std::unordered_map<std::string, Tensor> state_dict;

    // 2. Read Tensors
    for (uint32_t i = 0; i < num_tensors; ++i) {
        uint32_t name_len = 0;
        in.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        std::string name(name_len, '\0');
        in.read(&name[0], name_len);

        uint32_t rank = 0;
        in.read(reinterpret_cast<char*>(&rank), sizeof(uint32_t));
        std::vector<int64_t> shape(rank);
        for (uint32_t r = 0; r < rank; ++r) {
            in.read(reinterpret_cast<char*>(&shape[r]), sizeof(int64_t));
        }

        int64_t numel = 0;
        in.read(reinterpret_cast<char*>(&numel), sizeof(int64_t));

        Tensor tensor(shape, Device::CPU);
        in.read(reinterpret_cast<char*>(tensor.impl->data()), numel * sizeof(float));

        if (device == Device::CUDA) {
            state_dict[name] = tensor.to(Device::CUDA);
        } else {
            state_dict[name] = tensor;
        }
    }

    return state_dict;
}

void save_module(const nn::Module& module, const std::string& filepath) {
    save_checkpoint(module.state_dict(), filepath);
}

void load_module(nn::Module& module, const std::string& filepath, bool strict) {
    auto state_dict = load_checkpoint(filepath);
    module.load_state_dict(state_dict, strict);
}

} // namespace serialization
} // namespace sovereigngrad

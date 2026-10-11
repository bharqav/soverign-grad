#pragma once

#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/module.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace sovereigngrad {
namespace safetensors {

struct TensorMeta {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_offset_start = 0;
    size_t data_offset_end = 0;
};

// Save a state dictionary of tensors to an industry-standard SafeTensors (.safetensors) file
void save_file(const std::unordered_map<std::string, Tensor>& state_dict, const std::string& filepath);

// Load a state dictionary of tensors from a SafeTensors file
std::unordered_map<std::string, Tensor> load_file(const std::string& filepath, Device device = Device::CPU);

// Convenience methods for saving/loading an nn::Module directly via SafeTensors
void save_module(const nn::Module& module, const std::string& filepath);
void load_module(nn::Module& module, const std::string& filepath, bool strict = true);

// Inspect metadata of a SafeTensors file without loading tensor payload into memory
std::unordered_map<std::string, TensorMeta> inspect_file(const std::string& filepath);

} // namespace safetensors
} // namespace sovereigngrad

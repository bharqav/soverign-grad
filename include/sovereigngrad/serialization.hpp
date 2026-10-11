#pragma once

#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/module.hpp"
#include <string>
#include <unordered_map>

namespace sovereigngrad {
namespace serialization {

// Save a state dictionary of tensors to a binary file
void save_checkpoint(const std::unordered_map<std::string, Tensor>& state_dict, const std::string& filepath);

// Load a state dictionary of tensors from a binary file
std::unordered_map<std::string, Tensor> load_checkpoint(const std::string& filepath, Device device = Device::CPU);

// Convenience methods for saving/loading a module directly
void save_module(const nn::Module& module, const std::string& filepath);
void load_module(nn::Module& module, const std::string& filepath, bool strict = true);

} // namespace serialization
} // namespace sovereigngrad

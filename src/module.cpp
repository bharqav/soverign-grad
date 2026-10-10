#include "sovereigngrad/module.hpp"
#include "sovereigngrad/cuda_mock.hpp"
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace sovereigngrad {
namespace nn {

void Module::train(bool mode) {
    training_ = mode;
    for (auto& [name, sub] : submodules_) {
        (void)name;
        if (sub) {
            sub->train(mode);
        }
    }
}

void Module::to(Device device) {
    for (auto& [name, param_ptr] : parameters_) {
        (void)name;
        if (param_ptr && param_ptr->is_defined()) {
            *param_ptr = param_ptr->to(device);
        }
    }
    for (auto& [name, sub] : submodules_) {
        (void)name;
        if (sub) {
            sub->to(device);
        }
    }
}

void Module::register_parameter(const std::string& name, Tensor& param) {
    parameters_[name] = &param;
}

void Module::register_module(const std::string& name, std::shared_ptr<Module> module) {
    submodules_[name] = module;
}

std::vector<Tensor> Module::parameters() const {
    std::vector<Tensor> result;
    for (const auto& [name, param_ptr] : parameters_) {
        (void)name;
        if (param_ptr && param_ptr->is_defined()) {
            result.push_back(*param_ptr);
        }
    }
    for (const auto& [name, sub] : submodules_) {
        (void)name;
        if (sub) {
            auto sub_params = sub->parameters();
            result.insert(result.end(), sub_params.begin(), sub_params.end());
        }
    }
    return result;
}

std::vector<std::pair<std::string, Tensor>> Module::named_parameters(const std::string& prefix) const {
    std::vector<std::pair<std::string, Tensor>> result;
    for (const auto& [name, param_ptr] : parameters_) {
        if (param_ptr && param_ptr->is_defined()) {
            std::string full_name = prefix.empty() ? name : prefix + "." + name;
            result.push_back({full_name, *param_ptr});
        }
    }
    for (const auto& [name, sub] : submodules_) {
        if (sub) {
            std::string sub_prefix = prefix.empty() ? name : prefix + "." + name;
            auto sub_named = sub->named_parameters(sub_prefix);
            result.insert(result.end(), sub_named.begin(), sub_named.end());
        }
    }
    return result;
}

void Module::zero_grad() {
    for (auto& [name, param_ptr] : parameters_) {
        (void)name;
        if (param_ptr && param_ptr->is_defined()) {
            param_ptr->zero_grad();
        }
    }
    for (auto& [name, sub] : submodules_) {
        (void)name;
        if (sub) {
            sub->zero_grad();
        }
    }
}

std::unordered_map<std::string, Tensor> Module::state_dict(const std::string& prefix) const {
    std::unordered_map<std::string, Tensor> result;
    for (const auto& [name, param_ptr] : parameters_) {
        if (param_ptr && param_ptr->is_defined()) {
            std::string full_name = prefix.empty() ? name : prefix + "." + name;
            result[full_name] = param_ptr->clone();
        }
    }
    for (const auto& [name, sub] : submodules_) {
        if (sub) {
            std::string sub_prefix = prefix.empty() ? name : prefix + "." + name;
            auto sub_dict = sub->state_dict(sub_prefix);
            for (const auto& [k, v] : sub_dict) {
                result[k] = v;
            }
        }
    }
    return result;
}

void Module::load_state_dict(const std::unordered_map<std::string, Tensor>& state_dict, bool strict) {
    auto named = named_parameters();
    for (const auto& [name, tensor] : named) {
        auto it = state_dict.find(name);
        if (it != state_dict.end()) {
            if (it->second.shape() != tensor.shape()) {
                throw std::runtime_error("Shape mismatch for parameter '" + name + "': expected [" +
                                         tensor.summary() + "] but got [" + it->second.summary() + "]");
            }
            Tensor target_tensor = it->second.to(tensor.device());
            size_t bytes = tensor.numel() * sizeof(float);
            if (tensor.device() == Device::CPU) {
                std::memcpy(tensor.impl->data(), target_tensor.impl->data(), bytes);
            } else {
                CUDA_CHECK(cudaMemcpy(tensor.impl->data(), target_tensor.impl->data(), bytes, cudaMemcpyDeviceToDevice));
            }
        } else if (strict) {
            throw std::runtime_error("Missing parameter in state_dict: '" + name + "'");
        }
    }
}

} // namespace nn
} // namespace sovereigngrad

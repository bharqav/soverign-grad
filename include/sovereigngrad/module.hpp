#pragma once

#include "sovereigngrad/tensor.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace sovereigngrad {
namespace nn {

class Module {
public:
    Module() = default;
    virtual ~Module() = default;

    // Mode switching
    virtual void train(bool mode = true);
    void eval() { train(false); }
    bool is_training() const { return training_; }

    // Device placement
    virtual void to(Device device);
    void cpu() { to(Device::CPU); }
    void cuda() { to(Device::CUDA); }

    // Forward pass callable interface
    virtual Tensor forward(const Tensor& input) { return input; }
    virtual Tensor operator()(const Tensor& input) { return forward(input); }

    // Parameter & Submodule Registration
    void register_parameter(const std::string& name, Tensor& param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);

    // Parameter & Gradient Access
    std::vector<Tensor> parameters() const;
    std::vector<std::pair<std::string, Tensor>> named_parameters(const std::string& prefix = "") const;
    void zero_grad();

    // State Dict & Serialization
    std::unordered_map<std::string, Tensor> state_dict(const std::string& prefix = "") const;
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state_dict, bool strict = true);

protected:
    bool training_ = true;
    std::unordered_map<std::string, Tensor*> parameters_;
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;
};

} // namespace nn
} // namespace sovereigngrad

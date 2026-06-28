#pragma once

#include "soverigngrad/tensor.hpp"
#include <string>

namespace sovereigngrad {

struct MNISTDataset {
    Tensor images; // shape: [N, 1, 28, 28]
    Tensor labels; // shape: [N, 10] (one-hot)
    int64_t num_samples = 0;
};

// Loads official MNIST files, or generates a high-quality synthetic digit dataset if files are not found
MNISTDataset load_mnist(const std::string& images_path, const std::string& labels_path, int num_samples = 250, Device device = Device::CPU);

} // namespace sovereigngrad

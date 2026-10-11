#pragma once

#include "sovereigngrad/tensor.hpp"
#include <vector>
#include <cstdint>
#include <memory>

namespace sovereigngrad {
namespace quantization {

enum class QuantizationScheme {
    PerTensorSymmetric,
    PerChannelSymmetric,
    GroupedAsymmetric
};

struct QuantizedTensor {
    Tensor qdata;       // INT8 values (or packed 4-bit nibbles stored as uint8/float)
    Tensor scales;      // FP32 scaling factors
    Tensor zero_points; // Optional zero points
    QuantizationScheme scheme;
    std::vector<int64_t> original_shape;
    int64_t group_size;
    int bits;           // 8 or 4
};

// INT8 Quantization
QuantizedTensor quantize_int8(const Tensor& x, QuantizationScheme scheme = QuantizationScheme::PerChannelSymmetric);
Tensor dequantize_int8(const QuantizedTensor& qx);

// INT4 Quantization
QuantizedTensor quantize_int4(const Tensor& x, int64_t group_size = 32);
Tensor dequantize_int4(const QuantizedTensor& qx);

// Direct Low-Bit Matrix Multiplication
Tensor quantized_matmul_int8(const Tensor& x, const QuantizedTensor& q_weight, const Tensor& bias = Tensor());
Tensor quantized_matmul_int4(const Tensor& x, const QuantizedTensor& q_weight, const Tensor& bias = Tensor());

} // namespace quantization
} // namespace sovereigngrad

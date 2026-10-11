#include "sovereigngrad/quantization.hpp"
#include "sovereigngrad/ops.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sovereigngrad {
namespace quantization {

QuantizedTensor quantize_int8(const Tensor& x, QuantizationScheme scheme) {
    Tensor cpu_x = x.to(Device::CPU).contiguous();
    int64_t total = cpu_x.numel();
    const float* in_ptr = cpu_x.impl->data();

    QuantizedTensor result;
    result.original_shape = x.shape();
    result.scheme = scheme;
    result.bits = 8;
    result.group_size = total;

    if (scheme == QuantizationScheme::PerTensorSymmetric || x.shape().size() < 2) {
        float max_abs = 1e-8f;
        for (int64_t i = 0; i < total; ++i) {
            float a = std::fabs(in_ptr[i]);
            if (a > max_abs) max_abs = a;
        }

        float scale = max_abs / 127.0f;
        float inv_scale = 127.0f / max_abs;

        Tensor qdata(x.shape(), Device::CPU, false);
        float* q_ptr = qdata.impl->data();

        for (int64_t i = 0; i < total; ++i) {
            float v = std::round(in_ptr[i] * inv_scale);
            q_ptr[i] = std::max(-127.0f, std::min(127.0f, v));
        }

        result.qdata = (x.device() == Device::CUDA) ? qdata.to(Device::CUDA) : qdata;
        result.scales = Tensor(std::vector<float>{scale}, std::vector<int64_t>{1}, x.device(), false);
    } else {
        // Per-Channel (Row-wise)
        int64_t rows = x.shape()[0];
        int64_t cols = total / rows;

        Tensor qdata(x.shape(), Device::CPU, false);
        Tensor scales(std::vector<int64_t>{rows}, Device::CPU, false);

        float* q_ptr = qdata.impl->data();
        float* s_ptr = scales.impl->data();

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t r = 0; r < rows; ++r) {
            float max_abs = 1e-8f;
            for (int64_t c = 0; c < cols; ++c) {
                float a = std::fabs(in_ptr[r * cols + c]);
                if (a > max_abs) max_abs = a;
            }

            float scale = max_abs / 127.0f;
            float inv_scale = 127.0f / max_abs;
            s_ptr[r] = scale;

            for (int64_t c = 0; c < cols; ++c) {
                float v = std::round(in_ptr[r * cols + c] * inv_scale);
                q_ptr[r * cols + c] = std::max(-127.0f, std::min(127.0f, v));
            }
        }

        result.qdata = (x.device() == Device::CUDA) ? qdata.to(Device::CUDA) : qdata;
        result.scales = (x.device() == Device::CUDA) ? scales.to(Device::CUDA) : scales;
    }

    return result;
}

Tensor dequantize_int8(const QuantizedTensor& qx) {
    Tensor cpu_q = qx.qdata.to(Device::CPU).contiguous();
    Tensor cpu_s = qx.scales.to(Device::CPU).contiguous();
    Tensor out(qx.original_shape, Device::CPU, false);

    const float* q_ptr = cpu_q.impl->data();
    const float* s_ptr = cpu_s.impl->data();
    float* out_ptr = out.impl->data();
    int64_t total = out.numel();

    if (qx.scheme == QuantizationScheme::PerTensorSymmetric || cpu_s.numel() == 1) {
        float scale = s_ptr[0];
        for (int64_t i = 0; i < total; ++i) {
            out_ptr[i] = q_ptr[i] * scale;
        }
    } else {
        int64_t rows = qx.original_shape[0];
        int64_t cols = total / rows;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t r = 0; r < rows; ++r) {
            float scale = s_ptr[r];
            for (int64_t c = 0; c < cols; ++c) {
                out_ptr[r * cols + c] = q_ptr[r * cols + c] * scale;
            }
        }
    }

    return (qx.qdata.device() == Device::CUDA) ? out.to(Device::CUDA) : out;
}

QuantizedTensor quantize_int4(const Tensor& x, int64_t group_size) {
    Tensor cpu_x = x.to(Device::CPU).contiguous();
    int64_t total = cpu_x.numel();
    const float* in_ptr = cpu_x.impl->data();

    if (group_size <= 0) group_size = 32;
    int64_t num_groups = (total + group_size - 1) / group_size;

    QuantizedTensor result;
    result.original_shape = x.shape();
    result.scheme = QuantizationScheme::GroupedAsymmetric;
    result.bits = 4;
    result.group_size = group_size;

    Tensor qdata(x.shape(), Device::CPU, false);
    Tensor scales(std::vector<int64_t>{num_groups}, Device::CPU, false);

    float* q_ptr = qdata.impl->data();
    float* s_ptr = scales.impl->data();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t g_start = g * group_size;
        int64_t g_end = std::min(total, g_start + group_size);

        float max_abs = 1e-8f;
        for (int64_t i = g_start; i < g_end; ++i) {
            float a = std::fabs(in_ptr[i]);
            if (a > max_abs) max_abs = a;
        }

        float scale = max_abs / 7.0f;
        float inv_scale = 7.0f / max_abs;
        s_ptr[g] = scale;

        for (int64_t i = g_start; i < g_end; ++i) {
            float v = std::round(in_ptr[i] * inv_scale);
            q_ptr[i] = std::max(-8.0f, std::min(7.0f, v));
        }
    }

    result.qdata = (x.device() == Device::CUDA) ? qdata.to(Device::CUDA) : qdata;
    result.scales = (x.device() == Device::CUDA) ? scales.to(Device::CUDA) : scales;
    return result;
}

Tensor dequantize_int4(const QuantizedTensor& qx) {
    Tensor cpu_q = qx.qdata.to(Device::CPU).contiguous();
    Tensor cpu_s = qx.scales.to(Device::CPU).contiguous();
    Tensor out(qx.original_shape, Device::CPU, false);

    const float* q_ptr = cpu_q.impl->data();
    const float* s_ptr = cpu_s.impl->data();
    float* out_ptr = out.impl->data();
    int64_t total = out.numel();
    int64_t group_size = qx.group_size;
    int64_t num_groups = cpu_s.numel();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t g_start = g * group_size;
        int64_t g_end = std::min(total, g_start + group_size);
        float scale = s_ptr[g];

        for (int64_t i = g_start; i < g_end; ++i) {
            out_ptr[i] = q_ptr[i] * scale;
        }
    }

    return (qx.qdata.device() == Device::CUDA) ? out.to(Device::CUDA) : out;
}

Tensor quantized_matmul_int8(const Tensor& x, const QuantizedTensor& q_weight, const Tensor& bias) {
    // x: [B, K], weight: [K, N] (stored as [K, N] or per-channel [N, K])
    Tensor dequant_w = dequantize_int8(q_weight);
    Tensor out = x.matmul(dequant_w);
    if (bias.is_defined()) {
        out = out + bias;
    }
    return out;
}

Tensor quantized_matmul_int4(const Tensor& x, const QuantizedTensor& q_weight, const Tensor& bias) {
    Tensor dequant_w = dequantize_int4(q_weight);
    Tensor out = x.matmul(dequant_w);
    if (bias.is_defined()) {
        out = out + bias;
    }
    return out;
}

} // namespace quantization
} // namespace sovereigngrad

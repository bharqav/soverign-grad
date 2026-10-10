#pragma once

#include "sovereigngrad/allocator.hpp"
#include "sovereigngrad/cuda_mock.hpp"
#include <memory>
#include <functional>

namespace sovereigngrad {
namespace cuda {

class Stream {
public:
    Stream() {
#ifdef SOVEREIGNGRAD_USE_CUDA
        cudaStreamCreate(&raw_stream_);
#else
        raw_stream_ = nullptr;
#endif
    }

    ~Stream() {
#ifdef SOVEREIGNGRAD_USE_CUDA
        if (raw_stream_) {
            cudaStreamDestroy(raw_stream_);
        }
#endif
    }

    void synchronize() const {
#ifdef SOVEREIGNGRAD_USE_CUDA
        if (raw_stream_) {
            cudaStreamSynchronize(raw_stream_);
        }
#else
        cudaDeviceSynchronize();
#endif
    }

#ifdef SOVEREIGNGRAD_USE_CUDA
    cudaStream_t get() const { return raw_stream_; }
#else
    void* get() const { return raw_stream_; }
#endif

private:
#ifdef SOVEREIGNGRAD_USE_CUDA
    cudaStream_t raw_stream_;
#else
    void* raw_stream_ = nullptr;
#endif
};

class StreamGuard {
public:
    explicit StreamGuard(const Stream& stream) : stream_(stream) {
        // Switch active stream context
    }

    ~StreamGuard() {
        // Restore active stream context
    }

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;

private:
    const Stream& stream_;
};

inline void synchronize() {
    cudaDeviceSynchronize();
}

} // namespace cuda
} // namespace sovereigngrad

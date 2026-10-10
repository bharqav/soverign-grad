#pragma once

#ifndef SOVEREIGNGRAD_USE_CUDA

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>

using std::sqrt;

// CUDA execution modifiers
#define __global__
#define __device__
#define __host__
#define __shared__ thread_local static

// Emulated __syncthreads() does nothing since threads are emulated sequentially
inline void __syncthreads() {}

// CUDA grid and block dimensions
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
};

namespace sovereigngrad {
namespace cpu_emu {
    struct CudaContext {
        dim3 threadIdx_;
        dim3 blockIdx_;
        dim3 blockDim_;
        dim3 gridDim_;
    };
    
    // Thread-local context for executing kernels
    extern thread_local CudaContext ctx;
}
}

// Global kernel execution variables
#define threadIdx (::sovereigngrad::cpu_emu::ctx.threadIdx_)
#define blockIdx (::sovereigngrad::cpu_emu::ctx.blockIdx_)
#define blockDim (::sovereigngrad::cpu_emu::ctx.blockDim_)
#define gridDim (::sovereigngrad::cpu_emu::ctx.gridDim_)

// CUDA error types
enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 1,
    cudaErrorInvalidValue = 2,
    cudaErrorLaunchFailure = 3
};

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 0,
    cudaMemcpyDeviceToHost = 1,
    cudaMemcpyDeviceToDevice = 2,
    cudaMemcpyHostToHost = 3
};

inline const char* cudaGetErrorString(cudaError_t err) {
    switch (err) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
        case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
        default: return "cudaErrorUnknown";
    }
}

inline cudaError_t cudaGetLastError() {
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

// Emulated CUDA memory management
inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (!devPtr) return cudaErrorInvalidValue;
    *devPtr = std::malloc(size);
    if (!*devPtr) return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* devPtr) {
    if (devPtr) {
        std::free(devPtr);
    }
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    (void)kind;
    if (count > 0 && dst && src) {
        std::memcpy(dst, src, count);
    }
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    if (count > 0 && devPtr) {
        std::memset(devPtr, value, count);
    }
    return cudaSuccess;
}

// Emulated CUDA kernel launch macro
#define LAUNCH_KERNEL(kernel, grid, block, ...) \
    do { \
        dim3 _emu_grid = (grid); \
        dim3 _emu_block = (block); \
        for (unsigned int bz = 0; bz < _emu_grid.z; ++bz) { \
            for (unsigned int by = 0; by < _emu_grid.y; ++by) { \
                for (unsigned int bx = 0; bx < _emu_grid.x; ++bx) { \
                    for (unsigned int tz = 0; tz < _emu_block.z; ++tz) { \
                        for (unsigned int ty = 0; ty < _emu_block.y; ++ty) { \
                            for (unsigned int tx = 0; tx < _emu_block.x; ++tx) { \
                                ::sovereigngrad::cpu_emu::ctx.gridDim_ = _emu_grid; \
                                ::sovereigngrad::cpu_emu::ctx.blockDim_ = _emu_block; \
                                ::sovereigngrad::cpu_emu::ctx.blockIdx_ = dim3(bx, by, bz); \
                                ::sovereigngrad::cpu_emu::ctx.threadIdx_ = dim3(tx, ty, tz); \
                                kernel(__VA_ARGS__); \
                            } \
                        } \
                    } \
                } \
            } \
        } \
    } while (0)

#else

// Real CUDA headers
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define LAUNCH_KERNEL(kernel, grid, block, ...) \
    kernel<<<grid, block>>>(__VA_ARGS__)

#endif

// Robust error check macro for both real CUDA and CPU emulation
#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error [") + __FILE__ + ":" + std::to_string(__LINE__) + "]: " + cudaGetErrorString(_err)); \
        } \
    } while (0)

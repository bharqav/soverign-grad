#pragma once

#include <vector>
#include <mutex>
#include <cstddef>
#include <string>

namespace sovereigngrad {

enum class Device {
    CPU,
    CUDA
};

inline std::string device_to_string(Device d) {
    return (d == Device::CPU) ? "CPU" : "CUDA";
}

struct Block {
    void* ptr = nullptr;
    size_t size = 0;
    bool in_use = false;
    Device device = Device::CPU;
    bool is_raw_allocation = false;
    void* raw_parent = nullptr;
};

class MemoryAllocator {
public:
    static MemoryAllocator& instance() {
        static MemoryAllocator inst;
        return inst;
    }

    // Allocate memory on the specified device
    void* allocate(size_t bytes, Device device);

    // Free memory back to the cache
    void free(void* ptr, Device device);

    // Release all cached blocks back to the OS
    void release_all();

    // Statistics & Diagnostics
    size_t active_allocations(Device device) const;
    size_t cached_allocations(Device device) const;
    size_t total_memory_allocated(Device device) const;
    size_t peak_memory_allocated(Device device) const;
    void reset_peak_memory_stats();
    std::string get_memory_summary(Device device) const;

    ~MemoryAllocator();

private:
    MemoryAllocator() = default;
    
    // Prevent copying
    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;

    std::vector<Block> blocks_;
    mutable std::mutex mutex_;
    size_t peak_cpu_memory_ = 0;
    size_t peak_cuda_memory_ = 0;

    // Helper to merge adjacent free blocks
    void merge_free_blocks();
    void update_peak_stats();
};

} // namespace sovereigngrad

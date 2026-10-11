#include "sovereigngrad/allocator.hpp"
#include "sovereigngrad/cuda_mock.hpp"
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace sovereigngrad {

#ifndef SOVEREIGNGRAD_USE_CUDA
namespace cpu_emu {
    thread_local CudaContext ctx;
}
#endif

MemoryAllocator::~MemoryAllocator() {
    release_all();
}

void MemoryAllocator::update_peak_stats() {
    size_t cur_cpu = 0;
    size_t cur_cuda = 0;
    for (const auto& b : blocks_) {
        if (b.in_use) {
            if (b.device == Device::CPU) cur_cpu += b.size;
            else cur_cuda += b.size;
        }
    }
    if (cur_cpu > peak_cpu_memory_) peak_cpu_memory_ = cur_cpu;
    if (cur_cuda > peak_cuda_memory_) peak_cuda_memory_ = cur_cuda;
}

void* MemoryAllocator::allocate(size_t bytes, Device device) {
    if (bytes == 0) return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Search for a matching free block in the cache (Best-fit approach)
    size_t best_idx = static_cast<size_t>(-1);
    size_t best_size = static_cast<size_t>(-1);

    for (size_t i = 0; i < blocks_.size(); ++i) {
        const auto& block = blocks_[i];
        if (!block.in_use && block.device == device && block.size >= bytes) {
            if (block.size < best_size) {
                best_size = block.size;
                best_idx = i;
            }
        }
    }

    // 2. If a suitable free block is found
    if (best_idx != static_cast<size_t>(-1)) {
        void* ret_ptr = blocks_[best_idx].ptr;
        void* raw_parent = blocks_[best_idx].raw_parent;
        
        // Split the block if the remaining part is large enough (>= 1024 bytes)
        size_t min_split_bytes = 1024;
        if (blocks_[best_idx].size - bytes >= min_split_bytes) {
            Block split_block;
            split_block.ptr = static_cast<void*>(static_cast<char*>(ret_ptr) + bytes);
            split_block.size = blocks_[best_idx].size - bytes;
            split_block.in_use = false;
            split_block.device = device;
            split_block.is_raw_allocation = false;
            split_block.raw_parent = raw_parent;

            blocks_[best_idx].size = bytes;
            blocks_[best_idx].in_use = true;

            blocks_.insert(blocks_.begin() + best_idx + 1, split_block);
        } else {
            blocks_[best_idx].in_use = true;
        }
        update_peak_stats();
        return ret_ptr;
    }

    // 3. If no cached block is found, allocate a new one from the OS / driver
    size_t min_alloc = (device == Device::CUDA) ? 1024 * 1024 : 4096; // 1MB for GPU, 4KB for CPU
    size_t alloc_size = std::max(bytes, min_alloc);

    void* raw_ptr = nullptr;
    if (device == Device::CPU) {
        raw_ptr = std::malloc(alloc_size);
        if (!raw_ptr) {
            throw std::bad_alloc();
        }
    } else {
        cudaError_t err = cudaMalloc(&raw_ptr, alloc_size);
        if (err != cudaSuccess) {
            std::ostringstream ss;
            ss << "CUDA Allocator Out of Memory: requested " << alloc_size 
               << " bytes, active allocations: " << active_allocations(Device::CUDA)
               << ", total managed: " << total_memory_allocated(Device::CUDA)
               << " bytes. Error: " << cudaGetErrorString(err);
            throw std::runtime_error(ss.str());
        }
    }

    // Create the active block
    Block active_block;
    active_block.ptr = raw_ptr;
    active_block.size = bytes;
    active_block.in_use = true;
    active_block.device = device;
    active_block.is_raw_allocation = true;
    active_block.raw_parent = raw_ptr;
    blocks_.push_back(active_block);

    // Create the leftover block if we allocated more than requested
    if (alloc_size > bytes) {
        Block leftover_block;
        leftover_block.ptr = static_cast<void*>(static_cast<char*>(raw_ptr) + bytes);
        leftover_block.size = alloc_size - bytes;
        leftover_block.in_use = false;
        leftover_block.device = device;
        leftover_block.is_raw_allocation = false;
        leftover_block.raw_parent = raw_ptr;
        blocks_.push_back(leftover_block);
    }

    update_peak_stats();
    return raw_ptr;
}

void MemoryAllocator::free(void* ptr, Device device) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = std::find_if(blocks_.begin(), blocks_.end(), [ptr, device](const Block& b) {
        return b.ptr == ptr && b.device == device;
    });

    if (it != blocks_.end()) {
        it->in_use = false;
        merge_free_blocks();
    }
}

void MemoryAllocator::merge_free_blocks() {
    if (blocks_.size() < 2) return;

    std::sort(blocks_.begin(), blocks_.end(), [](const Block& a, const Block& b) {
        if (a.device != b.device) {
            return a.device < b.device;
        }
        return a.ptr < b.ptr;
    });

    for (size_t i = 0; i < blocks_.size() - 1; ) {
        Block& curr = blocks_[i];
        Block& next = blocks_[i + 1];

        if (!curr.in_use && !next.in_use && curr.device == next.device &&
            curr.raw_parent == next.raw_parent &&
            (static_cast<char*>(curr.ptr) + curr.size == static_cast<char*>(next.ptr))) {
            curr.size += next.size;
            blocks_.erase(blocks_.begin() + i + 1);
        } else {
            ++i;
        }
    }
}

void MemoryAllocator::release_all() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& block : blocks_) {
        if (block.is_raw_allocation && block.ptr) {
            if (block.device == Device::CPU) {
                std::free(block.ptr);
            } else {
                cudaFree(block.ptr);
            }
        }
    }
    blocks_.clear();
}

size_t MemoryAllocator::active_allocations(Device device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& b : blocks_) {
        if (b.in_use && b.device == device) count++;
    }
    return count;
}

size_t MemoryAllocator::cached_allocations(Device device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& b : blocks_) {
        if (!b.in_use && b.device == device) count++;
    }
    return count;
}

size_t MemoryAllocator::total_memory_allocated(Device device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& b : blocks_) {
        if (b.device == device) total += b.size;
    }
    return total;
}

size_t MemoryAllocator::peak_memory_allocated(Device device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return (device == Device::CPU) ? peak_cpu_memory_ : peak_cuda_memory_;
}

void MemoryAllocator::reset_peak_memory_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    peak_cpu_memory_ = 0;
    peak_cuda_memory_ = 0;
    update_peak_stats();
}

std::string MemoryAllocator::get_memory_summary(Device device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t active_bytes = 0;
    size_t cached_bytes = 0;
    size_t active_cnt = 0;
    size_t cached_cnt = 0;

    for (const auto& b : blocks_) {
        if (b.device == device) {
            if (b.in_use) {
                active_bytes += b.size;
                active_cnt++;
            } else {
                cached_bytes += b.size;
                cached_cnt++;
            }
        }
    }

    std::ostringstream ss;
    ss << "[" << device_to_string(device) << " Memory Summary] "
       << "Active: " << active_bytes / 1024.0f << " KB (" << active_cnt << " blocks), "
       << "Cached: " << cached_bytes / 1024.0f << " KB (" << cached_cnt << " blocks), "
       << "Peak: " << (device == Device::CPU ? peak_cpu_memory_ : peak_cuda_memory_) / 1024.0f << " KB";
    return ss.str();
}

} // namespace sovereigngrad

#include "soverigngrad/allocator.hpp"
#include "soverigngrad/cuda_mock.hpp"
#include <cstdlib>
#include <algorithm>
#include <iostream>

namespace sovereigngrad {

#ifndef SOVEREIGNGRAD_USE_CUDA
namespace cpu_emu {
    thread_local CudaContext ctx;
}
#endif

MemoryAllocator::~MemoryAllocator() {
    release_all();
}

void* MemoryAllocator::allocate(size_t bytes, Device device) {
    if (bytes == 0) return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Search for a matching free block in the cache (Best-fit approach)
    size_t best_idx = -1;
    size_t best_size = -1;

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
        // Save values locally to prevent iterator/reference invalidation during insert!
        void* ret_ptr = blocks_[best_idx].ptr;
        void* raw_parent = blocks_[best_idx].raw_parent;
        
        // Split the block if the remaining part is large enough (e.g. >= 1024 bytes)
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

            // Insert split block right after the current one (invalidates references, but ret_ptr is safe)
            blocks_.insert(blocks_.begin() + best_idx + 1, split_block);
        } else {
            blocks_[best_idx].in_use = true;
        }
        return ret_ptr;
    }

    // 3. If no cached block is found, allocate a new one from the OS / driver
    // We allocate a minimum chunk size to prevent future fragmentation and repeated calls to malloc/cudaMalloc
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
            throw std::runtime_error(std::string("CUDA Allocator Out of Memory: ") + cudaGetErrorString(err));
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

    return raw_ptr;
}

void MemoryAllocator::free(void* ptr, Device device) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // Find the block corresponding to the pointer
    auto it = std::find_if(blocks_.begin(), blocks_.end(), [ptr, device](const Block& b) {
        return b.ptr == ptr && b.device == device;
    });

    if (it != blocks_.end()) {
        it->in_use = false;
        // Merge free blocks to prevent fragmentation
        merge_free_blocks();
    } else {
        // Safe Fallback: If not found in cache, it has already been cleaned by release_all() or is unmanaged. Do nothing to prevent double-free!
    }
}

void MemoryAllocator::merge_free_blocks() {
    if (blocks_.size() < 2) return;

    // To merge blocks, we first sort them by pointer address (only valid within the same device)
    // Wait, let's group by device, then sort by pointer address
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
            // Merge next into curr
            curr.size += next.size;
            blocks_.erase(blocks_.begin() + i + 1);
            // Don't increment i; check if the new merged block can be merged with the next one
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

} // namespace sovereigngrad

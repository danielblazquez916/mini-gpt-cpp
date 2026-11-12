#include "mini_gpt/cuda/memory.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mini_gpt::cuda {
namespace {

std::unordered_map<std::size_t, std::vector<void*>> free_blocks;
std::size_t allocated_bytes = 0;
std::size_t live_bytes = 0;
std::size_t peak_live_bytes = 0;
std::mutex pool_mutex;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

void* allocate_device(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(pool_mutex);
    auto& blocks = free_blocks[bytes];
    void* pointer = nullptr;
    if (!blocks.empty()) {
        pointer = blocks.back();
        blocks.pop_back();
    } else {
        check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc memory pool");
        allocated_bytes += bytes;
    }
    live_bytes += bytes;
    peak_live_bytes = std::max(peak_live_bytes, live_bytes);
    return pointer;
}

void deallocate_device(void* pointer, std::size_t bytes) {
    if (pointer == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(pool_mutex);
    live_bytes -= bytes;
    free_blocks[bytes].push_back(pointer);
}

std::size_t cached_device_bytes() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    return allocated_bytes;
}

std::size_t live_device_bytes() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    return live_bytes;
}

std::size_t peak_live_device_bytes() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    return peak_live_bytes;
}

void reset_peak_live_device_bytes() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    peak_live_bytes = live_bytes;
}

}  // namespace mini_gpt::cuda

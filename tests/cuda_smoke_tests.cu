#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void add_vectors(const float* lhs, const float* rhs, float* output, std::size_t size) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = lhs[index] + rhs[index];
    }
}

}  // namespace

int main() {
    try {
        constexpr std::size_t size = 1024;
        std::vector<float> lhs(size);
        std::vector<float> rhs(size);
        for (std::size_t i = 0; i < size; ++i) {
            lhs[i] = static_cast<float>(i % 17) * 0.25F;
            rhs[i] = static_cast<float>(i % 13) * 0.5F;
        }

        float* device_lhs = nullptr;
        float* device_rhs = nullptr;
        float* device_output = nullptr;
        check_cuda(cudaMalloc(&device_lhs, size * sizeof(float)), "cudaMalloc lhs");
        check_cuda(cudaMalloc(&device_rhs, size * sizeof(float)), "cudaMalloc rhs");
        check_cuda(cudaMalloc(&device_output, size * sizeof(float)), "cudaMalloc output");
        check_cuda(cudaMemcpy(device_lhs, lhs.data(), size * sizeof(float), cudaMemcpyHostToDevice),
                   "copy lhs to device");
        check_cuda(cudaMemcpy(device_rhs, rhs.data(), size * sizeof(float), cudaMemcpyHostToDevice),
                   "copy rhs to device");

        add_vectors<<<(size + 255) / 256, 256>>>(device_lhs, device_rhs, device_output, size);
        check_cuda(cudaGetLastError(), "vector addition kernel");
        check_cuda(cudaDeviceSynchronize(), "synchronize vector addition");

        std::vector<float> output(size);
        check_cuda(cudaMemcpy(output.data(), device_output, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "copy output to host");
        for (std::size_t i = 0; i < size; ++i) {
            const float expected = lhs[i] + rhs[i];
            if (output[i] != expected) {
                throw std::runtime_error("CUDA result differs from CPU reference");
            }
        }

        check_cuda(cudaFree(device_lhs), "free lhs");
        check_cuda(cudaFree(device_rhs), "free rhs");
        check_cuda(cudaFree(device_output), "free output");
        std::cout << "[PASS] CUDA vector addition matches CPU reference\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}

#include "mini_gpt/cuda/matmul.cuh"
#include "mini_gpt/cuda/memory.cuh"
#include "mini_gpt/cuda/registry.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace mini_gpt::cuda {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuBLAS status " + std::to_string(status));
    }
}

}  // namespace

void matmul(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t rows,
    std::size_t inner,
    std::size_t columns) {
    if (rows == 0 || inner == 0 || columns == 0) {
        return;
    }

    float* device_lhs = nullptr;
    float* device_rhs = nullptr;
    float* device_output = nullptr;
    cublasHandle_t handle = nullptr;
    try {
        device_lhs = static_cast<float*>(allocate_device(rows * inner * sizeof(float)));
        check_cuda(device_lhs == nullptr ? cudaErrorMemoryAllocation : cudaSuccess, "allocate lhs");
        device_rhs = static_cast<float*>(allocate_device(inner * columns * sizeof(float)));
        device_output = static_cast<float*>(allocate_device(rows * columns * sizeof(float)));
        check_cuda(device_rhs == nullptr ? cudaErrorMemoryAllocation : cudaSuccess, "allocate rhs");
        check_cuda(device_output == nullptr ? cudaErrorMemoryAllocation : cudaSuccess, "allocate output");

        const float* lhs_kernel = device_lhs;
        const float* rhs_kernel = device_rhs;
        float* output_kernel = device_output;
        if (is_registered(lhs)) {
            lhs_kernel = static_cast<const float*>(device_for_host(lhs));
        } else {
            check_cuda(cudaMemcpy(
                           device_lhs,
                           lhs,
                           rows * inner * sizeof(float),
                           cudaMemcpyHostToDevice),
                       "copy lhs to device");
        }
        if (is_registered(rhs)) {
            rhs_kernel = static_cast<const float*>(device_for_host(rhs));
        } else {
            check_cuda(cudaMemcpy(
                           device_rhs,
                           rhs,
                           inner * columns * sizeof(float),
                           cudaMemcpyHostToDevice),
                       "copy rhs to device");
        }
        if (is_registered(output)) {
            output_kernel = static_cast<float*>(device_for_host(output));
        }

        check_cublas(cublasCreate(&handle), "cublasCreate");
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        check_cublas(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                static_cast<int>(columns),
                static_cast<int>(rows),
                static_cast<int>(inner),
                &alpha,
                rhs_kernel,
                static_cast<int>(columns),
                lhs_kernel,
                static_cast<int>(inner),
                &beta,
                output_kernel,
                static_cast<int>(columns)),
            "cublasSgemm");
        check_cuda(cudaDeviceSynchronize(), "synchronize cuBLAS matmul");
        if (is_registered(output)) {
            mark_device_modified(output);
        } else {
            check_cuda(cudaMemcpy(
                           output,
                           output_kernel,
                           rows * columns * sizeof(float),
                           cudaMemcpyDeviceToHost),
                       "copy output to host");
        }

        check_cublas(cublasDestroy(handle), "cublasDestroy");
        deallocate_device(device_lhs, rows * inner * sizeof(float));
        deallocate_device(device_rhs, inner * columns * sizeof(float));
        deallocate_device(device_output, rows * columns * sizeof(float));
    } catch (...) {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
        deallocate_device(device_lhs, rows * inner * sizeof(float));
        deallocate_device(device_rhs, inner * columns * sizeof(float));
        deallocate_device(device_output, rows * columns * sizeof(float));
        throw;
    }
}

}  // namespace mini_gpt::cuda

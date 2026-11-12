#include "mini_gpt/tensor.hpp"
#include "mini_gpt/cuda/registry.cuh"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mini_gpt::Tensor;

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_shape(const Tensor& tensor, const Tensor::Shape& expected) {
    check(tensor.shape() == expected, "Incorrect output shape");
}

void test_known_matrices() {
    const Tensor lhs({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    const Tensor rhs({3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12});
    const Tensor result = lhs.matmul(rhs);
    check_shape(result, {2, 2});
    const std::vector<float> expected = {58, 64, 139, 154};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        check(result.data()[i] == expected[i], "Incorrect matrix product");
    }
}

void test_zero_inner_dimension() {
    const Tensor lhs = Tensor::zeros({2, 0});
    const Tensor rhs = Tensor::zeros({0, 3});
    const Tensor result = lhs.matmul(rhs);
    check_shape(result, {2, 3});
    for (const float value : result.data()) {
        check(value == 0.0F, "Zero inner dimension must produce zeros");
    }
}

void test_cpu_reference() {
    const Tensor lhs = Tensor::random({4, 5}, 123, -1.0F, 1.0F);
    const Tensor rhs = Tensor::random({5, 3}, 456, -1.0F, 1.0F);
    const Tensor result = lhs.matmul(rhs);
    check_shape(result, {4, 3});
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            float expected = 0.0F;
            for (std::size_t inner = 0; inner < 5; ++inner) {
                expected += lhs.at({row, inner}) * rhs.at({inner, column});
            }
            check(
                std::fabs(result.at({row, column}) - expected) < 1.0e-5F,
                "CUDA result differs from CPU reference");
        }
    }
}

void test_resident_inputs_and_mutable_access() {
    const Tensor lhs({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    const Tensor rhs({3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12});
    const Tensor first = lhs.matmul(rhs);
    mini_gpt::cuda::reset_transfer_stats();
    const Tensor second = lhs.matmul(rhs);
    auto stats = mini_gpt::cuda::transfer_stats();
    check(stats.host_to_device_bytes == 0, "Unchanged inputs were uploaded again");
    check(stats.device_to_host_bytes == 0, "A CUDA result was downloaded without a host read");
    check(second.data()[0] == 58.0F, "Persistent CUDA matmul returned an incorrect value");
    stats = mini_gpt::cuda::transfer_stats();
    check(stats.device_to_host_bytes == second.size() * sizeof(float), "Host read did not synchronize once");
    check(second.data()[1] == 64.0F, "Cached host read returned an incorrect value");
    check(mini_gpt::cuda::transfer_stats().device_to_host_bytes == second.size() * sizeof(float),
          "A repeated host read downloaded the tensor again");

    Tensor mutable_lhs = lhs;
    mutable_lhs.at({0, 0}) = 2.0F;
    check(mutable_lhs.matmul(rhs).data()[0] == 65.0F, "Host mutation was not uploaded");
    mini_gpt::cuda::reset_transfer_stats();
    const Tensor changed = mutable_lhs.matmul(rhs);
    check(mini_gpt::cuda::transfer_stats().host_to_device_bytes == mutable_lhs.size() * sizeof(float),
          "A retained mutable reference did not keep host changes synchronized");
    check(changed.data()[0] == 65.0F, "Mutable host access produced an incorrect result");
}

}  // namespace

int main() {
    try {
        test_known_matrices();
        test_zero_inner_dimension();
        test_cpu_reference();
        test_resident_inputs_and_mutable_access();
        std::cout << "[PASS] CUDA matmul matches CPU reference\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}

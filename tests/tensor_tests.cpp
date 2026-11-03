#include "mini_gpt/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using mini_gpt::Tensor;
using Shape = Tensor::Shape;

class TestFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::string location(const char* file, int line) {
    return std::string(file) + ':' + std::to_string(line);
}

void check(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        throw TestFailure(location(file, line) + ": check failed: " + expression);
    }
}

void check_near(
    double actual,
    double expected,
    double tolerance,
    const char* actual_expression,
    const char* expected_expression,
    const char* file,
    int line) {
    if (std::fabs(actual - expected) > tolerance) {
        std::ostringstream message;
        message << location(file, line) << ": " << actual_expression << " == " << actual
                << ", expected " << expected_expression << " == " << expected;
        throw TestFailure(message.str());
    }
}

template <typename Exception, typename Function>
void check_throws(Function&& function, const char* expression, const char* file, int line) {
    try {
        std::forward<Function>(function)();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        throw TestFailure(
            location(file, line) + ": " + expression +
            " threw a different exception: " + error.what());
    } catch (...) {
        throw TestFailure(
            location(file, line) + ": " + expression + " threw a non-standard exception");
    }

    throw TestFailure(location(file, line) + ": " + expression + " did not throw");
}

#define CHECK(expression) \
    check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

#define CHECK_NEAR(actual, expected, tolerance) \
    check_near( \
        static_cast<double>(actual), \
        static_cast<double>(expected), \
        static_cast<double>(tolerance), \
        #actual, \
        #expected, \
        __FILE__, \
        __LINE__)

#define CHECK_THROWS(exception_type, ...) \
    check_throws<exception_type>( \
        [&]() { static_cast<void>((__VA_ARGS__)); }, #__VA_ARGS__, __FILE__, __LINE__)

void check_values(const Tensor& tensor, std::initializer_list<float> expected) {
    CHECK(tensor.size() == expected.size());
    std::size_t index = 0;
    for (const float expected_value : expected) {
        check_near(
            tensor.data()[index],
            expected_value,
            1.0e-6,
            "tensor value",
            "expected value",
            __FILE__,
            __LINE__);
        ++index;
    }
}

void test_zero_initialization_and_strides() {
    const Tensor tensor = Tensor::zeros({2, 3, 4});
    CHECK(tensor.rank() == 3);
    CHECK(tensor.shape() == Shape({2, 3, 4}));
    CHECK(tensor.strides() == Shape({12, 4, 1}));
    CHECK(tensor.size() == 24);
    CHECK(tensor.data().size() == 24);
    for (const float value : tensor.data()) {
        CHECK(value == 0.0F);
    }
}

void test_one_initialization() {
    const Tensor tensor = Tensor::ones({2, 1, 3});
    CHECK(tensor.shape() == Shape({2, 1, 3}));
    CHECK(tensor.strides() == Shape({3, 3, 1}));
    CHECK(tensor.size() == 6);
    for (const float value : tensor.data()) {
        CHECK(value == 1.0F);
    }
}

void test_explicit_data_and_checked_access() {
    Tensor tensor({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    const Tensor& const_tensor = tensor;

    CHECK(const_tensor.at({0, 0}) == 1.0F);
    CHECK(const_tensor.at({0, 2}) == 3.0F);
    CHECK(const_tensor.at({1, 0}) == 4.0F);
    CHECK(&const_tensor.at({1, 2}) == &const_tensor.data()[5]);

    tensor.at({1, 2}) = 9.0F;
    CHECK(const_tensor.at({1, 2}) == 9.0F);
}

void test_scalar_tensor() {
    Tensor scalar = Tensor::ones({});
    CHECK(scalar.rank() == 0);
    CHECK(scalar.shape().empty());
    CHECK(scalar.strides().empty());
    CHECK(scalar.size() == 1);
    CHECK(scalar.at({}) == 1.0F);
    scalar.at({}) = 4.0F;
    CHECK(scalar.data()[0] == 4.0F);
}

void test_empty_tensor() {
    const Tensor tensor = Tensor::zeros({2, 0, 3});
    CHECK(tensor.rank() == 3);
    CHECK(tensor.strides() == Shape({0, 3, 1}));
    CHECK(tensor.size() == 0);
    CHECK(tensor.data().empty());
    CHECK_THROWS(std::out_of_range, tensor.at({0, 0, 0}));
}

void test_constructor_rejects_wrong_data_size() {
    CHECK_THROWS(
        std::invalid_argument,
        Tensor({2, 2}, std::vector<float>{1.0F, 2.0F, 3.0F}));
    CHECK_THROWS(std::invalid_argument, Tensor({0, 2}, std::vector<float>{1.0F}));
}

void test_constructor_detects_size_overflow() {
    constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
    CHECK_THROWS(std::length_error, Tensor::zeros({maximum, 2}));
}

void test_access_rejects_wrong_rank() {
    Tensor tensor = Tensor::zeros({2, 3});
    CHECK_THROWS(std::invalid_argument, tensor.at({0}));
    CHECK_THROWS(std::invalid_argument, tensor.at({0, 0, 0}));
}

void test_access_rejects_out_of_range_indices() {
    const Tensor tensor = Tensor::zeros({2, 3});
    CHECK_THROWS(std::out_of_range, tensor.at({2, 0}));
    CHECK_THROWS(std::out_of_range, tensor.at({0, 3}));
}

void test_random_is_reproducible() {
    const Tensor first = Tensor::random({4, 5}, 1234U, -0.75F, 0.25F);
    const Tensor second = Tensor::random({4, 5}, 1234U, -0.75F, 0.25F);
    const Tensor different = Tensor::random({4, 5}, 4321U, -0.75F, 0.25F);

    CHECK(std::equal(
        first.data().begin(), first.data().end(), second.data().begin()));
    CHECK(!std::equal(
        first.data().begin(), first.data().end(), different.data().begin()));
    for (const float value : first.data()) {
        CHECK(value >= -0.75F);
        CHECK(value < 0.25F);
    }
}

void test_random_rejects_invalid_bounds() {
    CHECK_THROWS(std::invalid_argument, Tensor::random({2}, 0U, 1.0F, 0.0F));
    CHECK_THROWS(
        std::invalid_argument,
        Tensor::random(
            {2}, 0U, 0.0F, std::numeric_limits<float>::infinity()));
}

void test_addition() {
    const Tensor lhs({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    const Tensor rhs({2, 3}, std::vector<float>{6, 5, 4, 3, 2, 1});
    const Tensor result = lhs + rhs;

    CHECK(result.shape() == Shape({2, 3}));
    CHECK(result.strides() == Shape({3, 1}));
    check_values(result, {7, 7, 7, 7, 7, 7});
    check_values(lhs, {1, 2, 3, 4, 5, 6});
    check_values(rhs, {6, 5, 4, 3, 2, 1});
}

void test_subtraction() {
    const Tensor lhs({2, 2}, std::vector<float>{8, 3, -1, 5});
    const Tensor rhs({2, 2}, std::vector<float>{2, 4, -3, 1});
    check_values(lhs - rhs, {6, -1, 2, 4});
}

void test_multiplication() {
    const Tensor lhs({2, 2}, std::vector<float>{2, -3, 0.5F, 4});
    const Tensor rhs({2, 2}, std::vector<float>{5, 2, 4, -1});
    check_values(lhs * rhs, {10, -6, 2, -4});
}

void test_elementwise_rejects_shape_mismatch() {
    const Tensor lhs = Tensor::zeros({2, 3});
    const Tensor different_shape = Tensor::zeros({3, 2});
    const Tensor different_rank = Tensor::zeros({2, 3, 1});

    CHECK_THROWS(std::invalid_argument, lhs + different_shape);
    CHECK_THROWS(std::invalid_argument, lhs - different_shape);
    CHECK_THROWS(std::invalid_argument, lhs * different_shape);
    CHECK_THROWS(std::invalid_argument, lhs + different_rank);
}

void test_elementwise_supports_empty_tensors() {
    const Tensor lhs = Tensor::ones({2, 0});
    const Tensor rhs = Tensor::ones({2, 0});
    CHECK((lhs + rhs).size() == 0);
    CHECK((lhs - rhs).shape() == Shape({2, 0}));
    CHECK((lhs * rhs).data().empty());
}

void test_matmul() {
    const Tensor lhs({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    const Tensor rhs({3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12});
    const Tensor result = lhs.matmul(rhs);

    CHECK(result.shape() == Shape({2, 2}));
    CHECK(result.strides() == Shape({2, 1}));
    check_values(result, {58, 64, 139, 154});
}

void test_matmul_vector_shaped_matrices() {
    const Tensor lhs({1, 4}, std::vector<float>{1, 2, 3, 4});
    const Tensor rhs({4, 1}, std::vector<float>{2, -1, 0.5F, 3});
    const Tensor result = lhs.matmul(rhs);

    CHECK(result.shape() == Shape({1, 1}));
    check_values(result, {13.5F});
}

void test_matmul_rejects_incompatible_inner_dimensions() {
    const Tensor lhs = Tensor::zeros({2, 3});
    const Tensor rhs = Tensor::zeros({4, 2});
    CHECK_THROWS(std::invalid_argument, lhs.matmul(rhs));
}

void test_matmul_requires_matrices() {
    const Tensor matrix = Tensor::zeros({2, 2});
    const Tensor vector = Tensor::zeros({2});
    const Tensor tensor3d = Tensor::zeros({1, 2, 2});

    CHECK_THROWS(std::invalid_argument, vector.matmul(matrix));
    CHECK_THROWS(std::invalid_argument, matrix.matmul(vector));
    CHECK_THROWS(std::invalid_argument, tensor3d.matmul(matrix));
}

void test_matmul_with_zero_inner_dimension() {
    const Tensor lhs = Tensor::zeros({2, 0});
    const Tensor rhs = Tensor::zeros({0, 3});
    const Tensor result = lhs.matmul(rhs);

    CHECK(result.shape() == Shape({2, 3}));
    CHECK(result.size() == 6);
    for (const float value : result.data()) {
        CHECK(value == 0.0F);
    }
}

void test_transpose() {
    const Tensor tensor({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    const Tensor result = tensor.transpose();

    CHECK(result.shape() == Shape({3, 2}));
    CHECK(result.strides() == Shape({2, 1}));
    check_values(result, {1, 4, 2, 5, 3, 6});
    check_values(tensor, {1, 2, 3, 4, 5, 6});
}

void test_transpose_twice_restores_original() {
    const Tensor tensor({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    const Tensor result = tensor.transpose().transpose();
    CHECK(result.shape() == tensor.shape());
    CHECK(result.strides() == tensor.strides());
    check_values(result, {1, 2, 3, 4, 5, 6});
}

void test_transpose_empty_matrix() {
    const Tensor tensor = Tensor::zeros({0, 3});
    const Tensor result = tensor.transpose();
    CHECK(result.shape() == Shape({3, 0}));
    CHECK(result.strides() == Shape({0, 1}));
    CHECK(result.data().empty());
}

void test_transpose_requires_matrix() {
    CHECK_THROWS(std::invalid_argument, Tensor::zeros({3}).transpose());
    CHECK_THROWS(std::invalid_argument, Tensor::zeros({1, 2, 3}).transpose());
}

void test_reshape_preserves_data_and_storage() {
    Tensor tensor({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    const float* const original_data = tensor.data().data();

    tensor.reshape({3, 2});
    CHECK(tensor.shape() == Shape({3, 2}));
    CHECK(tensor.strides() == Shape({2, 1}));
    CHECK(tensor.data().data() == original_data);
    check_values(tensor, {1, 2, 3, 4, 5, 6});

    tensor.reshape({6});
    CHECK(tensor.rank() == 1);
    CHECK(tensor.strides() == Shape({1}));
    CHECK(tensor.data().data() == original_data);
}

void test_reshape_rejects_size_mismatch_without_modifying_tensor() {
    Tensor tensor = Tensor::ones({2, 3});
    const float* const original_data = tensor.data().data();

    CHECK_THROWS(std::invalid_argument, tensor.reshape({4, 2}));
    CHECK(tensor.shape() == Shape({2, 3}));
    CHECK(tensor.strides() == Shape({3, 1}));
    CHECK(tensor.data().data() == original_data);
    CHECK(tensor.size() == 6);
}

void test_reshape_empty_tensor() {
    Tensor tensor = Tensor::zeros({2, 0, 3});
    tensor.reshape({0, 3});
    CHECK(tensor.shape() == Shape({0, 3}));
    CHECK(tensor.strides() == Shape({3, 1}));
    CHECK(tensor.size() == 0);
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"zero initialization and strides", test_zero_initialization_and_strides},
        {"one initialization", test_one_initialization},
        {"explicit data and checked access", test_explicit_data_and_checked_access},
        {"scalar tensor", test_scalar_tensor},
        {"empty tensor", test_empty_tensor},
        {"constructor rejects wrong data size", test_constructor_rejects_wrong_data_size},
        {"constructor detects size overflow", test_constructor_detects_size_overflow},
        {"access rejects wrong rank", test_access_rejects_wrong_rank},
        {"access rejects out-of-range indices", test_access_rejects_out_of_range_indices},
        {"random is reproducible", test_random_is_reproducible},
        {"random rejects invalid bounds", test_random_rejects_invalid_bounds},
        {"addition", test_addition},
        {"subtraction", test_subtraction},
        {"multiplication", test_multiplication},
        {"elementwise rejects shape mismatch", test_elementwise_rejects_shape_mismatch},
        {"elementwise supports empty tensors", test_elementwise_supports_empty_tensors},
        {"matmul", test_matmul},
        {"matmul vector-shaped matrices", test_matmul_vector_shaped_matrices},
        {"matmul rejects incompatible dimensions", test_matmul_rejects_incompatible_inner_dimensions},
        {"matmul requires matrices", test_matmul_requires_matrices},
        {"matmul with zero inner dimension", test_matmul_with_zero_inner_dimension},
        {"transpose", test_transpose},
        {"transpose twice restores original", test_transpose_twice_restores_original},
        {"transpose empty matrix", test_transpose_empty_matrix},
        {"transpose requires matrix", test_transpose_requires_matrix},
        {"reshape preserves data and storage", test_reshape_preserves_data_and_storage},
        {"reshape rejects size mismatch", test_reshape_rejects_size_mismatch_without_modifying_tensor},
        {"reshape empty tensor", test_reshape_empty_tensor},
    };

    int failures = 0;
    for (const TestCase& test : tests) {
        try {
            test.function();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const TestFailure& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << "\n  " << error.what() << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << "\n  Unexpected exception: "
                      << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << "\n  Unknown exception\n";
        }
    }

    const std::size_t test_count = sizeof(tests) / sizeof(tests[0]);
    std::cout << "\n" << (test_count - static_cast<std::size_t>(failures)) << '/'
              << test_count << " tests passed\n";
    return failures == 0 ? 0 : 1;
}

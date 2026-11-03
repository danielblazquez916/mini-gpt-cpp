#include "mini_gpt/cross_entropy.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using mini_gpt::Tensor;
using mini_gpt::cross_entropy;
using mini_gpt::perplexity;
using mini_gpt::shift_tokens;

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
        throw TestFailure(
            location(file, line) + ": " + actual_expression + " != " + expected_expression +
            " (actual " + std::to_string(actual) + ", expected " +
            std::to_string(expected) + ")");
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

void test_known_values_and_average() {
    const Tensor logits(
        {1, 2, 3},
        std::vector<float>{
            std::log(0.7F), std::log(0.2F), std::log(0.1F),
            std::log(0.1F), std::log(0.6F), std::log(0.3F)});
    const std::vector<std::uint32_t> targets = {0, 1};
    const double expected = (-std::log(0.7) - std::log(0.6)) / 2.0;
    CHECK_NEAR(cross_entropy(logits, targets), expected, 1.0e-6);
}

void test_uniform_distribution_and_batch_shapes() {
    const Tensor logits = Tensor::zeros({2, 2, 4});
    const std::vector<std::uint32_t> targets = {0, 1, 2, 3};
    CHECK_NEAR(cross_entropy(logits, targets), std::log(4.0), 1.0e-6);
}

void test_confident_correct_and_incorrect_predictions() {
    const Tensor correct({1, 1, 3}, std::vector<float>{10, 0, 0});
    const Tensor incorrect({1, 1, 3}, std::vector<float>{0, 10, 0});
    const std::vector<std::uint32_t> targets = {0};
    const float correct_loss = cross_entropy(correct, targets);
    const float incorrect_loss = cross_entropy(incorrect, targets);
    CHECK(correct_loss < 0.001F);
    CHECK(incorrect_loss > 9.9F);
    CHECK(incorrect_loss > correct_loss + 9.0F);
}

void test_extreme_logits_are_stable() {
    const Tensor logits(
        {1, 2, 3},
        std::vector<float>{10000, 9999, -10000, -10000, -10001, -20000});
    const std::vector<std::uint32_t> targets = {1, 0};
    const float loss = cross_entropy(logits, targets);
    CHECK(std::isfinite(loss));
    CHECK_NEAR(loss, 0.8132616875, 1.0e-5);
}

void test_mask_controls_average_and_skips_invalid_positions() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const Tensor logits(
        {1, 3, 2},
        std::vector<float>{
            std::log(0.8F), std::log(0.2F),
            nan, nan,
            std::log(0.9F), std::log(0.1F)});
    const std::vector<std::uint32_t> targets = {0, 99, 0};
    const std::vector<std::uint8_t> mask = {1, 0, 1};
    const double expected = (-std::log(0.8) - std::log(0.9)) / 2.0;
    CHECK_NEAR(cross_entropy(logits, targets, mask), expected, 1.0e-6);
}

void test_autoregressive_shift() {
    const std::vector<std::uint32_t> tokens = {10, 11, 12, 13, 14};
    const auto shifted = shift_tokens(tokens);
    CHECK(shifted.inputs == std::vector<std::uint32_t>({10, 11, 12, 13}));
    CHECK(shifted.targets == std::vector<std::uint32_t>({11, 12, 13, 14}));
    const std::vector<std::uint32_t> single = {10};
    CHECK_THROWS(std::invalid_argument, shift_tokens(single));
}

void test_perplexity() {
    CHECK_NEAR(perplexity(std::log(4.0F)), 4.0, 1.0e-4);
    CHECK_NEAR(perplexity(0.0F), 1.0, 1.0e-6);
    CHECK_THROWS(std::invalid_argument, perplexity(-1.0F));
}

void test_dimensions_and_target_validation() {
    const std::vector<std::uint32_t> empty;
    CHECK_THROWS(std::invalid_argument, cross_entropy(Tensor::zeros({2, 3}), empty));

    const Tensor logits = Tensor::zeros({1, 2, 3});
    const std::vector<std::uint32_t> wrong_count = {0};
    CHECK_THROWS(std::invalid_argument, cross_entropy(logits, wrong_count));
    const std::vector<std::uint32_t> invalid_id = {0, 3};
    CHECK_THROWS(std::out_of_range, cross_entropy(logits, invalid_id));

    const Tensor no_vocabulary = Tensor::zeros({1, 1, 0});
    const std::vector<std::uint32_t> target = {0};
    CHECK_THROWS(std::invalid_argument, cross_entropy(no_vocabulary, target));
}

void test_mask_empty_and_nonfinite_validation() {
    const Tensor logits = Tensor::zeros({1, 2, 3});
    const std::vector<std::uint32_t> targets = {0, 0};
    const std::vector<std::uint8_t> wrong_size = {1};
    CHECK_THROWS(std::invalid_argument, cross_entropy(logits, targets, wrong_size));
    const std::vector<std::uint8_t> invalid_value = {1, 2};
    CHECK_THROWS(std::invalid_argument, cross_entropy(logits, targets, invalid_value));
    const std::vector<std::uint8_t> all_ignored = {0, 0};
    CHECK_THROWS(std::invalid_argument, cross_entropy(logits, targets, all_ignored));

    const Tensor empty_positions = Tensor::zeros({1, 0, 3});
    const std::vector<std::uint32_t> empty;
    CHECK_THROWS(std::invalid_argument, cross_entropy(empty_positions, empty));

    const Tensor nonfinite(
        {1, 1, 2},
        std::vector<float>{0.0F, std::numeric_limits<float>::infinity()});
    const std::vector<std::uint32_t> target = {0};
    CHECK_THROWS(std::invalid_argument, cross_entropy(nonfinite, target));
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"known values and average", test_known_values_and_average},
        {"uniform distribution and batch shapes", test_uniform_distribution_and_batch_shapes},
        {"confident correct and incorrect predictions", test_confident_correct_and_incorrect_predictions},
        {"extreme logits are stable", test_extreme_logits_are_stable},
        {"mask controls average and skips invalid positions", test_mask_controls_average_and_skips_invalid_positions},
        {"autoregressive shift", test_autoregressive_shift},
        {"perplexity", test_perplexity},
        {"dimensions and target validation", test_dimensions_and_target_validation},
        {"mask empty and nonfinite validation", test_mask_empty_and_nonfinite_validation},
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

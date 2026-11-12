#include "mini_gpt/inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using mini_gpt::GPT;
using mini_gpt::SamplingStrategy;
using mini_gpt::Tensor;

class TestFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void check(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        throw TestFailure(std::string(file) + ':' + std::to_string(line) + ": " + expression);
    }
}

void check_near(double actual, double expected, double tolerance, const char* file, int line) {
    if (std::fabs(actual - expected) > tolerance) {
        throw TestFailure(
            std::string(file) + ':' + std::to_string(line) + ": actual " +
            std::to_string(actual) + ", expected " + std::to_string(expected));
    }
}

template <typename Exception, typename Function>
void check_throws(Function&& function) {
    try {
        std::forward<Function>(function)();
    } catch (const Exception&) {
        return;
    } catch (...) {
        throw TestFailure("Unexpected exception type");
    }
    throw TestFailure("Expected exception was not thrown");
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)
#define CHECK_NEAR(actual, expected, tolerance) \
    check_near(static_cast<double>(actual), static_cast<double>(expected), tolerance, __FILE__, __LINE__)
#define CHECK_THROWS(exception_type, ...) \
    check_throws<exception_type>([&]() { static_cast<void>((__VA_ARGS__)); })

void test_top_k_limits_candidates_and_validates() {
    const std::vector<float> logits = {0.1F, 3.0F, 2.0F, 0.0F};
    std::mt19937 generator(123);
    for (std::size_t i = 0; i < 50; ++i) {
        const auto selected = GPT::select_token(logits, 1.0F, SamplingStrategy::sampling, generator, 2);
        CHECK(selected == 1 || selected == 2);
    }
    CHECK(GPT::select_token(logits, 1.0F, SamplingStrategy::greedy, generator, 0) == 1);
    CHECK_THROWS(std::invalid_argument, GPT::select_token(
        logits, 1.0F, SamplingStrategy::sampling, generator, 5));
}

void test_top_k_disabled_and_temperature_probabilities() {
    const std::vector<float> logits = {1.0F, 2.0F};
    const auto standard = GPT::softmax(logits, 1.0F);
    CHECK_NEAR(standard[0], 0.2689414214, 1.0e-6);
    CHECK_NEAR(standard[1], 0.7310585786, 1.0e-6);
    const auto tempered = GPT::softmax(logits, 2.0F);
    CHECK_NEAR(tempered[0], 0.3775406688, 1.0e-6);
    CHECK_NEAR(tempered[1], 0.6224593312, 1.0e-6);
}

void test_repetition_penalty_sign_uniqueness_and_window() {
    std::vector<float> logits = {2.0F, -2.0F, 1.0F, 4.0F};
    const std::vector<std::uint32_t> context = {0, 0, 1, 2};
    GPT::apply_repetition_penalty(context, logits, 2.0F);
    CHECK_NEAR(logits[0], 1.0, 1.0e-6);
    CHECK_NEAR(logits[1], -4.0, 1.0e-6);
    CHECK_NEAR(logits[2], 0.5, 1.0e-6);
    CHECK_NEAR(logits[3], 4.0, 1.0e-6);

    GPT::apply_repetition_penalty(context, logits, 1.0F);
    CHECK_NEAR(logits[0], 1.0, 1.0e-6);
    CHECK_NEAR(logits[1], -4.0, 1.0e-6);

    std::vector<float> windowed = {2.0F, -2.0F, 1.0F, 4.0F};
    GPT::apply_repetition_penalty(context, windowed, 2.0F, 1);
    CHECK_NEAR(windowed[0], 2.0, 1.0e-6);
    CHECK_NEAR(windowed[1], -2.0, 1.0e-6);
    CHECK_NEAR(windowed[2], 0.5, 1.0e-6);
}

GPT constant_model() {
    GPT model(4, 2, 1, 1, 4, 4, 123);
    std::fill(model.final_norm().gamma().data().begin(), model.final_norm().gamma().data().end(), 0.0F);
    std::fill(model.final_norm().beta().data().begin(), model.final_norm().beta().data().end(), 1.0F);
    std::fill(model.output_weights().data().begin(), model.output_weights().data().end(), 0.0F);
    for (std::size_t dimension = 0; dimension < model.embedding_dim(); ++dimension) {
        model.output_weights().at({dimension, 1}) = std::log(3.0F) / model.embedding_dim();
        model.output_weights().at({dimension, 2}) = std::log(9.0F) / model.embedding_dim();
    }
    return model;
}

void test_top_probabilities_are_correct() {
    GPT model = constant_model();
    const std::vector<std::uint32_t> context = {1};
    const auto top = mini_gpt::top_token_probabilities(model, context, 3);
    CHECK(top.size() == 3);
    CHECK(top[0].id == 2 && top[0].rank == 1);
    CHECK(top[1].id == 1 && top[1].rank == 2);
    CHECK(top[2].id == 0 && top[2].rank == 3);
    CHECK_NEAR(top[0].probability, 9.0 / 14.0, 1.0e-5);
    CHECK_NEAR(top[1].probability, 3.0 / 14.0, 1.0e-5);
    CHECK_NEAR(top[2].probability, 1.0 / 14.0, 1.0e-5);
}

void test_continuation_loss_and_ranking() {
    GPT model = constant_model();
    const auto original_weights = model.output_weights();
    const std::vector<std::uint32_t> prompt = {1};
    const std::vector<std::uint32_t> expected = {2, 1};
    const auto evaluation = mini_gpt::evaluate_continuation(model, prompt, expected);

    CHECK(evaluation.tokens.size() == 2);
    CHECK(evaluation.tokens[0].id == 2);
    CHECK(evaluation.tokens[0].rank == 1);
    CHECK(evaluation.tokens[1].id == 1);
    CHECK(evaluation.tokens[1].rank == 2);
    CHECK_NEAR(evaluation.tokens[0].probability, 9.0 / 14.0, 1.0e-5);
    CHECK_NEAR(evaluation.tokens[1].probability, 3.0 / 14.0, 1.0e-5);
    const double expected_loss =
        (-std::log(9.0 / 14.0) - std::log(3.0 / 14.0)) / 2.0;
    CHECK_NEAR(evaluation.loss, expected_loss, 1.0e-5);
    CHECK(std::equal(
        original_weights.data().begin(), original_weights.data().end(),
        model.output_weights().data().begin()));
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"Top-k limits candidates and validates", test_top_k_limits_candidates_and_validates},
        {"Top-k disabled and temperature probabilities", test_top_k_disabled_and_temperature_probabilities},
        {"repetition penalty sign uniqueness and window", test_repetition_penalty_sign_uniqueness_and_window},
        {"top probabilities are correct", test_top_probabilities_are_correct},
        {"continuation loss and ranking", test_continuation_loss_and_ranking},
    };

    int failures = 0;
    for (const TestCase& test : tests) {
        try {
            test.function();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << "\n  " << error.what() << '\n';
        }
    }
    std::cout << "\n" << (sizeof(tests) / sizeof(tests[0]) - failures) << '/'
              << sizeof(tests) / sizeof(tests[0]) << " tests passed\n";
    return failures == 0 ? 0 : 1;
}

#include "mini_gpt/adamw.hpp"
#include "mini_gpt/cross_entropy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using mini_gpt::AdamW;
using mini_gpt::AdamWConfig;
using mini_gpt::AdamWParameter;
using mini_gpt::AdamWParameters;
using mini_gpt::GPT;
using mini_gpt::Tensor;

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

AdamWParameter binding(Tensor& parameter, Tensor& gradient, bool decay = false) {
    return AdamWParameter{&parameter, &gradient, decay, "test"};
}

AdamWConfig test_config() {
    AdamWConfig config;
    config.learning_rate = 0.1F;
    config.weight_decay = 0.0F;
    return config;
}

class ScopedFile {
public:
    explicit ScopedFile(std::filesystem::path path) : path_(std::move(path)) {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    ~ScopedFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void test_first_step_and_bias_correction() {
    Tensor parameter({1}, std::vector<float>{1.0F});
    Tensor gradient({1}, std::vector<float>{0.5F});
    AdamW optimizer(AdamWParameters{binding(parameter, gradient)}, test_config());
    optimizer.step();

    CHECK(optimizer.step_count() == 1);
    CHECK_NEAR(optimizer.state(0).first_moment.at({0}), 0.05, 1.0e-6);
    CHECK_NEAR(optimizer.state(0).second_moment.at({0}), 0.00025, 1.0e-8);
    CHECK_NEAR(parameter.at({0}), 0.9, 1.0e-5);
    CHECK_NEAR(gradient.at({0}), 0.5, 1.0e-6);
}

void test_multiple_steps_and_moments() {
    Tensor parameter({1}, std::vector<float>{1.0F});
    Tensor gradient({1}, std::vector<float>{0.5F});
    AdamW optimizer(AdamWParameters{binding(parameter, gradient)}, test_config());
    optimizer.step();
    optimizer.step();

    CHECK(optimizer.step_count() == 2);
    CHECK_NEAR(optimizer.state(0).first_moment.at({0}), 0.095, 1.0e-6);
    CHECK_NEAR(optimizer.state(0).second_moment.at({0}), 0.00049975, 1.0e-8);
    CHECK_NEAR(parameter.at({0}), 0.8, 1.0e-5);
}

void test_adamw_decoupled_weight_decay() {
    Tensor parameter({1}, std::vector<float>{1.0F});
    Tensor gradient = Tensor::zeros({1});
    AdamWConfig config = test_config();
    config.weight_decay = 0.1F;
    AdamW optimizer(AdamWParameters{binding(parameter, gradient, true)}, config);
    optimizer.step();

    CHECK_NEAR(parameter.at({0}), 0.99, 1.0e-6);
    CHECK(gradient.at({0}) == 0.0F);

    Tensor l2_parameter({1}, std::vector<float>{1.0F});
    Tensor l2_gradient({1}, std::vector<float>{0.1F});
    AdamW l2_equivalent(AdamWParameters{binding(l2_parameter, l2_gradient)}, test_config());
    l2_equivalent.step();
    CHECK_NEAR(l2_parameter.at({0}), 0.9, 1.0e-5);
    CHECK(std::fabs(parameter.at({0}) - l2_parameter.at({0})) > 0.08F);
}

void test_zero_gradient_without_decay_and_reproducibility() {
    Tensor first({1}, std::vector<float>{1.0F});
    Tensor second({1}, std::vector<float>{1.0F});
    Tensor first_gradient = Tensor::zeros({1});
    Tensor second_gradient = Tensor::zeros({1});
    AdamW first_optimizer(AdamWParameters{binding(first, first_gradient)}, test_config());
    AdamW second_optimizer(AdamWParameters{binding(second, second_gradient)}, test_config());
    first_optimizer.step();
    second_optimizer.step();
    CHECK(first.at({0}) == 1.0F);
    CHECK(second.at({0}) == 1.0F);
    CHECK(first.at({0}) == second.at({0}));
}

void test_gradient_clipping() {
    Tensor first_parameter = Tensor::zeros({1});
    Tensor second_parameter = Tensor::zeros({1});
    Tensor first_gradient({1}, std::vector<float>{3.0F});
    Tensor second_gradient({1}, std::vector<float>{4.0F});
    const AdamWParameters parameters = {
        binding(first_parameter, first_gradient),
        binding(second_parameter, second_gradient)};

    const double original_norm = mini_gpt::clip_gradients(parameters, 1.0);
    CHECK_NEAR(original_norm, 5.0, 1.0e-6);
    CHECK_NEAR(first_gradient.at({0}), 0.6, 1.0e-6);
    CHECK_NEAR(second_gradient.at({0}), 0.8, 1.0e-6);
    CHECK_NEAR(mini_gpt::gradient_global_norm(parameters), 1.0, 1.0e-6);
}

void test_optimizer_state_persistence() {
    const ScopedFile file(std::filesystem::current_path() / "mini_gpt_adamw_state_test.bin");
    Tensor parameter({1}, std::vector<float>{1.0F});
    Tensor gradient({1}, std::vector<float>{0.5F});
    AdamW original(AdamWParameters{binding(parameter, gradient)}, test_config());
    original.step();
    original.step();
    original.save(file.path());

    Tensor restored_parameter({1}, std::vector<float>{parameter.at({0})});
    Tensor restored_gradient({1}, std::vector<float>{0.5F});
    AdamW restored = AdamW::load(
        file.path(), AdamWParameters{binding(restored_parameter, restored_gradient)});
    CHECK(restored.step_count() == 2);
    CHECK_NEAR(restored.state(0).first_moment.at({0}), original.state(0).first_moment.at({0}), 0.0);
    CHECK_NEAR(restored.state(0).second_moment.at({0}), original.state(0).second_moment.at({0}), 0.0);

    original.step();
    restored.step();
    CHECK(restored.step_count() == 3);
    CHECK_NEAR(parameter.at({0}), 0.7, 1.0e-5);
    CHECK_NEAR(restored_parameter.at({0}), parameter.at({0}), 0.0);
    CHECK_NEAR(restored.state(0).first_moment.at({0}), original.state(0).first_moment.at({0}), 0.0);
    CHECK_NEAR(restored.state(0).second_moment.at({0}), original.state(0).second_moment.at({0}), 0.0);
}

void test_nonfinite_gradient_is_rejected_without_updates() {
    Tensor parameter({1}, std::vector<float>{1.0F});
    Tensor gradient({1}, std::vector<float>{std::numeric_limits<float>::quiet_NaN()});
    AdamW optimizer(AdamWParameters{binding(parameter, gradient)}, test_config());
    CHECK_THROWS(std::runtime_error, optimizer.step());
    CHECK(parameter.at({0}) == 1.0F);
    CHECK(optimizer.step_count() == 0);
}

std::vector<float> parameter_values(GPT& model) {
    std::vector<float> values;
    for (const auto& binding : mini_gpt::gpt_parameters(model)) {
        values.insert(
            values.end(), binding.parameter->data().begin(), binding.parameter->data().end());
    }
    return values;
}

std::vector<float> gradient_values(GPT& model) {
    std::vector<float> values;
    for (const auto& binding : mini_gpt::gpt_parameters(model)) {
        values.insert(
            values.end(), binding.gradient->data().begin(), binding.gradient->data().end());
    }
    return values;
}

void test_optimizer_updates_all_gpt_components_without_changing_gradients() {
    GPT model(12, 8, 2, 2, 16, 5, 101);
    const std::vector<std::uint32_t> inputs = {1, 2, 1, 3, 2, 1};
    const std::vector<std::uint32_t> targets = {2, 1, 3, 1, 2, 1};
    model.zero_grad();
    model.backward(inputs, 2, 3, targets);
    const auto original_parameters = parameter_values(model);
    const auto original_gradients = gradient_values(model);

    AdamWConfig config;
    config.learning_rate = 0.001F;
    AdamW optimizer(model, config);
    CHECK(optimizer.parameter_count() == 25);
    optimizer.step();
    CHECK(parameter_values(model) != original_parameters);
    CHECK(gradient_values(model) == original_gradients);

    bool found_layernorm = false;
    for (const auto& binding : optimizer.parameters()) {
        if (binding.name.find("norm") != std::string::npos) {
            found_layernorm = true;
            CHECK(!binding.weight_decay);
        }
    }
    CHECK(found_layernorm);
}

void test_training_reduces_real_loss() {
    GPT model(8, 8, 2, 1, 16, 5, 123);
    const std::vector<std::uint32_t> inputs = {1, 2, 3, 4};
    const std::vector<std::uint32_t> targets = {2, 3, 4, 5};
    const float initial_loss = mini_gpt::cross_entropy(model.forward(inputs, 1, 4), targets);

    AdamWConfig config;
    config.learning_rate = 0.003F;
    config.weight_decay = 0.0F;
    AdamW optimizer(model, config);
    for (std::size_t step = 0; step < 50; ++step) {
        model.zero_grad();
        model.backward(inputs, 1, 4, targets);
        mini_gpt::clip_gradients(optimizer.parameters(), 1.0);
        optimizer.step();
    }
    const float final_loss = mini_gpt::cross_entropy(model.forward(inputs, 1, 4), targets);
    CHECK(std::isfinite(final_loss));
    CHECK(final_loss < initial_loss * 0.8F);
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"first step and bias correction", test_first_step_and_bias_correction},
        {"multiple steps and moments", test_multiple_steps_and_moments},
        {"AdamW decoupled weight decay", test_adamw_decoupled_weight_decay},
        {"zero gradient without decay and reproducibility", test_zero_gradient_without_decay_and_reproducibility},
        {"gradient clipping", test_gradient_clipping},
        {"optimizer state persistence", test_optimizer_state_persistence},
        {"nonfinite gradient is rejected without updates", test_nonfinite_gradient_is_rejected_without_updates},
        {"optimizer updates all GPT components without changing gradients", test_optimizer_updates_all_gpt_components_without_changing_gradients},
        {"training reduces real loss", test_training_reduces_real_loss},
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

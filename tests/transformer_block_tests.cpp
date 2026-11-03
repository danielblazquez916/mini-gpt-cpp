#include "mini_gpt/transformer_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using mini_gpt::LayerNorm;
using mini_gpt::Tensor;
using mini_gpt::TransformerBlock;
using mini_gpt::gelu;

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

void check_shape(const Tensor& tensor, std::initializer_list<std::size_t> expected) {
    CHECK(tensor.shape() == Tensor::Shape(expected));
}

bool same_values(const Tensor& lhs, const Tensor& rhs) {
    return lhs.shape() == rhs.shape() &&
           std::equal(lhs.data().begin(), lhs.data().end(), rhs.data().begin());
}

bool same_bits(const Tensor& lhs, const Tensor& rhs) {
    return lhs.shape() == rhs.shape() && lhs.size() == rhs.size() &&
           (lhs.size() == 0 ||
            std::memcmp(
                lhs.data().data(), rhs.data().data(), lhs.size() * sizeof(float)) == 0);
}

void set_zero(Tensor& tensor) {
    std::fill(tensor.data().begin(), tensor.data().end(), 0.0F);
}

void set_identity(Tensor& tensor) {
    CHECK(tensor.rank() == 2);
    CHECK(tensor.shape()[0] == tensor.shape()[1]);
    set_zero(tensor);
    for (std::size_t i = 0; i < tensor.shape()[0]; ++i) {
        tensor.at({i, i}) = 1.0F;
    }
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

void test_layer_norm_known_values_and_initialization() {
    const LayerNorm norm(3);
    CHECK(norm.normalized_dim() == 3);
    CHECK_NEAR(norm.epsilon(), 1.0e-5, 1.0e-12);
    for (const float value : norm.gamma().data()) {
        CHECK(value == 1.0F);
    }
    for (const float value : norm.beta().data()) {
        CHECK(value == 0.0F);
    }

    const Tensor input({3}, std::vector<float>{1, 2, 3});
    const Tensor output = norm.forward(input);
    const double expected_scale = 1.0 / std::sqrt(2.0 / 3.0 + 1.0e-5);
    CHECK_NEAR(output.at({0}), -expected_scale, 1.0e-5);
    CHECK_NEAR(output.at({1}), 0.0, 1.0e-5);
    CHECK_NEAR(output.at({2}), expected_scale, 1.0e-5);
}

void test_layer_norm_gamma_beta_and_constant_input() {
    LayerNorm norm(3);
    norm.gamma() = Tensor({3}, std::vector<float>{2, 3, 4});
    norm.beta() = Tensor({3}, std::vector<float>{1, -1, 0.5F});
    const Tensor input({2, 3}, std::vector<float>{5, 5, 5, 2, 2, 2});

    const Tensor output = norm.forward(input);
    CHECK_NEAR(output.at({0, 0}), 1.0, 1.0e-6);
    CHECK_NEAR(output.at({0, 1}), -1.0, 1.0e-6);
    CHECK_NEAR(output.at({0, 2}), 0.5, 1.0e-6);
    CHECK_NEAR(output.at({1, 0}), 1.0, 1.0e-6);
    CHECK_NEAR(output.at({1, 1}), -1.0, 1.0e-6);
    CHECK_NEAR(output.at({1, 2}), 0.5, 1.0e-6);
}

void test_layer_norm_validation() {
    CHECK_THROWS(std::invalid_argument, LayerNorm(0));
    CHECK_THROWS(std::invalid_argument, LayerNorm(3, 0.0F));
    CHECK_THROWS(std::invalid_argument, LayerNorm(3, -1.0F));
    const LayerNorm norm(3);
    CHECK_THROWS(std::invalid_argument, norm.forward(Tensor::zeros({2, 2})));
    CHECK_THROWS(std::invalid_argument, norm.forward(Tensor::zeros({})));
}

void test_gelu_reference_values_and_extremes() {
    const Tensor input(
        {7}, std::vector<float>{-2, -1, 0, 1, 2, 100, -100});
    const Tensor output = gelu(input);
    CHECK_NEAR(output.at({0}), -0.0455002639, 1.0e-5);
    CHECK_NEAR(output.at({1}), -0.1586552539, 1.0e-5);
    CHECK_NEAR(output.at({2}), 0.0, 1.0e-6);
    CHECK_NEAR(output.at({3}), 0.8413447461, 1.0e-5);
    CHECK_NEAR(output.at({4}), 1.9544997361, 1.0e-5);
    CHECK_NEAR(output.at({5}), 100.0, 1.0e-5);
    CHECK_NEAR(output.at({6}), 0.0, 1.0e-6);
    for (const float value : output.data()) {
        CHECK(std::isfinite(value));
    }
}

void test_default_configuration_and_parameters() {
    const TransformerBlock block;
    CHECK(block.embedding_dim() == 256);
    CHECK(block.num_heads() == 4);
    CHECK(block.mlp_hidden_dim() == 1024);
    CHECK_NEAR(block.epsilon(), 1.0e-5, 1.0e-12);
    check_shape(block.fc1_weights(), {256, 1024});
    check_shape(block.fc2_weights(), {1024, 256});
    check_shape(block.attention().q_weights(), {256, 256});
    check_shape(block.norm1().gamma(), {256});
    check_shape(block.norm2().beta(), {256});
}

void test_invalid_configuration_and_input() {
    CHECK_THROWS(std::invalid_argument, TransformerBlock(4, 2, 0));
    CHECK_THROWS(std::invalid_argument, TransformerBlock(4, 2, 8, 42, 0.0F));
    CHECK_THROWS(std::invalid_argument, TransformerBlock(10, 3, 20));
    const TransformerBlock block(4, 2, 8);
    CHECK_THROWS(std::invalid_argument, block.forward(Tensor::zeros({2, 4})));
    CHECK_THROWS(std::invalid_argument, block.forward(Tensor::zeros({1, 2, 3})));
}

void test_forward_preserves_shapes_for_batches_and_sequences() {
    const TransformerBlock block(4, 2, 8, 17);
    for (const auto& dimensions : {std::pair<std::size_t, std::size_t>{1, 1},
                                   {2, 3},
                                   {3, 2}}) {
        const std::size_t batch = dimensions.first;
        const std::size_t sequence = dimensions.second;
        const Tensor input = Tensor::random({batch, sequence, 4}, 19, -1.0F, 1.0F);
        const Tensor output = block.forward(input);
        check_shape(output, {batch, sequence, 4});
        CHECK(output.size() == input.size());
    }
}

void test_empty_inputs() {
    const TransformerBlock block(4, 2, 8);
    const Tensor empty_sequence = block.forward(Tensor::zeros({1, 0, 4}));
    check_shape(empty_sequence, {1, 0, 4});
    CHECK(empty_sequence.size() == 0);

    const Tensor empty_batch = block.forward(Tensor::zeros({0, 3, 4}));
    check_shape(empty_batch, {0, 3, 4});
    CHECK(empty_batch.size() == 0);
}

void test_controlled_attention_residual() {
    TransformerBlock block(1, 1, 1, 7);
    set_zero(block.attention().q_weights());
    set_zero(block.attention().k_weights());
    set_identity(block.attention().v_weights());
    set_identity(block.attention().output_weights());
    set_zero(block.fc1_weights());
    set_zero(block.fc2_weights());
    block.norm1().beta().at({0}) = 1.0F;
    const Tensor input({1, 1, 1}, std::vector<float>{2});

    const Tensor output = block.forward(input);
    CHECK_NEAR(output.at({0, 0, 0}), 3.0, 1.0e-6);
}

void test_controlled_mlp_and_second_residual() {
    TransformerBlock block(1, 1, 1, 11);
    set_zero(block.attention().q_weights());
    set_zero(block.attention().k_weights());
    set_zero(block.attention().v_weights());
    set_zero(block.attention().output_weights());
    block.norm2().beta().at({0}) = 1.0F;
    block.fc1_weights().at({0, 0}) = 1.0F;
    block.fc2_weights().at({0, 0}) = 2.0F;
    const Tensor input({1, 1, 1}, std::vector<float>{2});

    const Tensor output = block.forward(input);
    const double expected = 2.0 + 2.0 * 0.8413447460685429;
    CHECK_NEAR(output.at({0, 0, 0}), expected, 1.0e-5);
}

void test_future_tokens_do_not_change_earlier_outputs() {
    const TransformerBlock block(4, 2, 8, 29);
    const Tensor original = Tensor::random({1, 4, 4}, 31, -1.0F, 1.0F);
    Tensor modified = original;
    for (std::size_t position = 2; position < 4; ++position) {
        for (std::size_t dimension = 0; dimension < 4; ++dimension) {
            modified.at({0, position, dimension}) +=
                static_cast<float>(position + dimension + 1);
        }
    }

    const Tensor first = block.forward(original);
    const Tensor second = block.forward(modified);
    for (std::size_t position = 0; position < 2; ++position) {
        for (std::size_t dimension = 0; dimension < 4; ++dimension) {
            CHECK_NEAR(
                first.at({0, position, dimension}),
                second.at({0, position, dimension}),
                0.0);
        }
    }
    CHECK(!same_values(first, second));
}

void test_forward_produces_finite_values() {
    const TransformerBlock block(4, 2, 8, 37);
    const Tensor input = Tensor::random({2, 3, 4}, 41, -2.0F, 2.0F);
    const Tensor output = block.forward(input);
    for (const float value : output.data()) {
        CHECK(std::isfinite(value));
    }
}

void test_initialization_is_reproducible() {
    const TransformerBlock first(4, 2, 8, 123);
    const TransformerBlock second(4, 2, 8, 123);
    const TransformerBlock different(4, 2, 8, 124);
    CHECK(same_values(first.attention().q_weights(), second.attention().q_weights()));
    CHECK(same_values(first.attention().k_weights(), second.attention().k_weights()));
    CHECK(same_values(first.attention().v_weights(), second.attention().v_weights()));
    CHECK(same_values(first.attention().output_weights(), second.attention().output_weights()));
    CHECK(same_values(first.fc1_weights(), second.fc1_weights()));
    CHECK(same_values(first.fc2_weights(), second.fc2_weights()));
    CHECK(!same_values(first.attention().q_weights(), different.attention().q_weights()));
    CHECK(!same_values(first.fc1_weights(), different.fc1_weights()));
}

void test_persistence_preserves_parameters_and_forward() {
    const ScopedFile file(
        std::filesystem::current_path() / "mini_gpt_transformer_block_persistence_test.bin");
    TransformerBlock block(4, 2, 8, 987, 1.0e-4F);
    block.norm1().gamma().data().front() = 1.25F;
    block.norm1().beta().data().back() = -0.0F;
    block.norm2().gamma().data().front() = 0.75F;
    block.norm2().beta().data().back() = 0.5F;
    block.attention().q_weights().data().front() = 0.123456789F;
    block.attention().k_weights().data().back() = -0.75F;
    block.attention().v_weights().data().front() = 0.5F;
    block.attention().output_weights().data().back() = 0.987654321F;
    block.fc1_weights().data().front() = -0.25F;
    block.fc2_weights().data().back() = 0.333333333F;
    block.save(file.path());

    const TransformerBlock loaded = TransformerBlock::load(file.path());
    CHECK(loaded.embedding_dim() == 4);
    CHECK(loaded.num_heads() == 2);
    CHECK(loaded.mlp_hidden_dim() == 8);
    CHECK_NEAR(loaded.epsilon(), 1.0e-4, 1.0e-8);
    CHECK(same_bits(loaded.norm1().gamma(), block.norm1().gamma()));
    CHECK(same_bits(loaded.norm1().beta(), block.norm1().beta()));
    CHECK(same_bits(loaded.norm2().gamma(), block.norm2().gamma()));
    CHECK(same_bits(loaded.norm2().beta(), block.norm2().beta()));
    CHECK(same_bits(loaded.attention().q_weights(), block.attention().q_weights()));
    CHECK(same_bits(loaded.attention().k_weights(), block.attention().k_weights()));
    CHECK(same_bits(loaded.attention().v_weights(), block.attention().v_weights()));
    CHECK(same_bits(loaded.attention().output_weights(), block.attention().output_weights()));
    CHECK(same_bits(loaded.fc1_weights(), block.fc1_weights()));
    CHECK(same_bits(loaded.fc2_weights(), block.fc2_weights()));

    const Tensor input = Tensor::random({2, 3, 4}, 43, -1.0F, 1.0F);
    CHECK(same_values(loaded.forward(input), block.forward(input)));
}

void test_blocks_have_independent_parameters() {
    TransformerBlock first(4, 2, 8, 55);
    const TransformerBlock second(4, 2, 8, 55);
    const float second_q = second.attention().q_weights().data().front();
    const float second_fc1 = second.fc1_weights().data().front();
    const float second_gamma = second.norm1().gamma().data().front();

    first.attention().q_weights().data().front() += 1.0F;
    first.fc1_weights().data().front() += 1.0F;
    first.norm1().gamma().data().front() = 2.0F;
    CHECK(second.attention().q_weights().data().front() == second_q);
    CHECK(second.fc1_weights().data().front() == second_fc1);
    CHECK(second.norm1().gamma().data().front() == second_gamma);
}

void test_load_rejects_invalid_files() {
    const ScopedFile file(
        std::filesystem::current_path() / "mini_gpt_transformer_block_invalid_test.bin");
    CHECK_THROWS(std::runtime_error, TransformerBlock::load(file.path()));
    {
        std::ofstream output(file.path(), std::ios::binary | std::ios::trunc);
        output << "not a block";
    }
    CHECK_THROWS(std::runtime_error, TransformerBlock::load(file.path()));

    const TransformerBlock block(4, 2, 8);
    block.save(file.path());
    std::filesystem::resize_file(file.path(), 20);
    CHECK_THROWS(std::runtime_error, TransformerBlock::load(file.path()));

    block.save(file.path());
    {
        std::fstream stream(file.path(), std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(12);
        const std::uint64_t zero = 0;
        stream.write(
            reinterpret_cast<const char*>(&zero),
            static_cast<std::streamsize>(sizeof(zero)));
    }
    CHECK_THROWS(std::runtime_error, TransformerBlock::load(file.path()));
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"LayerNorm known values and initialization", test_layer_norm_known_values_and_initialization},
        {"LayerNorm gamma beta and constant input", test_layer_norm_gamma_beta_and_constant_input},
        {"LayerNorm validation", test_layer_norm_validation},
        {"GELU reference values and extremes", test_gelu_reference_values_and_extremes},
        {"default configuration and parameters", test_default_configuration_and_parameters},
        {"invalid configuration and input", test_invalid_configuration_and_input},
        {"forward preserves shapes for batches and sequences", test_forward_preserves_shapes_for_batches_and_sequences},
        {"empty inputs", test_empty_inputs},
        {"controlled attention residual", test_controlled_attention_residual},
        {"controlled MLP and second residual", test_controlled_mlp_and_second_residual},
        {"future tokens do not change earlier outputs", test_future_tokens_do_not_change_earlier_outputs},
        {"forward produces finite values", test_forward_produces_finite_values},
        {"initialization is reproducible", test_initialization_is_reproducible},
        {"persistence preserves parameters and forward", test_persistence_preserves_parameters_and_forward},
        {"blocks have independent parameters", test_blocks_have_independent_parameters},
        {"load rejects invalid files", test_load_rejects_invalid_files},
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

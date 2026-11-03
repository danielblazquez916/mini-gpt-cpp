#include "mini_gpt/causal_self_attention.hpp"

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

using mini_gpt::AttentionForwardResult;
using mini_gpt::CausalSelfAttention;
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
            location(file, line) + ": " + actual_expression + " != " + expected_expression);
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

float probability(
    const AttentionForwardResult& result,
    std::size_t batch,
    std::size_t head,
    std::size_t query,
    std::size_t key) {
    return result.probabilities.at({batch, head, query, key});
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

void test_default_configuration() {
    const CausalSelfAttention attention;
    CHECK(attention.embedding_dim() == 256);
    CHECK(attention.num_heads() == 4);
    CHECK(attention.head_dim() == 64);
    check_shape(attention.q_weights(), {256, 256});
    check_shape(attention.k_weights(), {256, 256});
    check_shape(attention.v_weights(), {256, 256});
    check_shape(attention.output_weights(), {256, 256});
}

void test_invalid_configuration() {
    CHECK_THROWS(std::invalid_argument, CausalSelfAttention(0, 4));
    CHECK_THROWS(std::invalid_argument, CausalSelfAttention(4, 0));
    CHECK_THROWS(std::invalid_argument, CausalSelfAttention(10, 3));
}

void test_initialization_is_reproducible() {
    const CausalSelfAttention first(8, 2, 123);
    const CausalSelfAttention second(8, 2, 123);
    const CausalSelfAttention different(8, 2, 124);

    CHECK(same_values(first.q_weights(), second.q_weights()));
    CHECK(same_values(first.k_weights(), second.k_weights()));
    CHECK(same_values(first.v_weights(), second.v_weights()));
    CHECK(same_values(first.output_weights(), second.output_weights()));
    CHECK(!same_values(first.q_weights(), different.q_weights()));
    CHECK(!same_values(first.k_weights(), different.k_weights()));
    CHECK(!same_values(first.v_weights(), different.v_weights()));
    CHECK(!same_values(first.output_weights(), different.output_weights()));
}

void test_output_qkv_and_attention_shapes() {
    const CausalSelfAttention attention(8, 2, 7);
    const Tensor input = Tensor::random({2, 3, 8}, 11, -1.0F, 1.0F);
    const AttentionForwardResult result = attention.forward_with_details(input);

    check_shape(result.output, {2, 3, 8});
    check_shape(result.queries, {2, 2, 3, 4});
    check_shape(result.keys, {2, 2, 3, 4});
    check_shape(result.values, {2, 2, 3, 4});
    check_shape(result.probabilities, {2, 2, 3, 3});
    check_shape(attention.forward(input), {2, 3, 8});
}

void test_input_validation() {
    const CausalSelfAttention attention(4, 2);
    CHECK_THROWS(std::invalid_argument, attention.forward(Tensor::zeros({2, 4})));
    CHECK_THROWS(std::invalid_argument, attention.forward(Tensor::zeros({1, 2, 3})));
}

void test_projections_and_head_split() {
    CausalSelfAttention attention(4, 2);
    set_identity(attention.q_weights());
    set_identity(attention.k_weights());
    set_identity(attention.v_weights());
    set_identity(attention.output_weights());
    const Tensor input({1, 2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});

    const auto result = attention.forward_with_details(input);
    CHECK(result.queries.at({0, 0, 0, 0}) == 1.0F);
    CHECK(result.queries.at({0, 0, 0, 1}) == 2.0F);
    CHECK(result.queries.at({0, 0, 1, 0}) == 5.0F);
    CHECK(result.queries.at({0, 0, 1, 1}) == 6.0F);
    CHECK(result.queries.at({0, 1, 0, 0}) == 3.0F);
    CHECK(result.queries.at({0, 1, 0, 1}) == 4.0F);
    CHECK(result.queries.at({0, 1, 1, 0}) == 7.0F);
    CHECK(result.queries.at({0, 1, 1, 1}) == 8.0F);
    CHECK(same_values(result.queries, result.keys));
    CHECK(same_values(result.queries, result.values));
}

void test_uniform_causal_attention_and_output_projection() {
    CausalSelfAttention attention(2, 1);
    set_zero(attention.q_weights());
    set_zero(attention.k_weights());
    set_identity(attention.v_weights());
    set_identity(attention.output_weights());
    for (std::size_t i = 0; i < 2; ++i) {
        attention.output_weights().at({i, i}) = 2.0F;
    }
    const Tensor input({1, 2, 2}, std::vector<float>{1, 2, 3, 4});

    const auto result = attention.forward_with_details(input);
    CHECK_NEAR(probability(result, 0, 0, 0, 0), 1.0, 1.0e-6);
    CHECK(probability(result, 0, 0, 0, 1) == 0.0F);
    CHECK_NEAR(probability(result, 0, 0, 1, 0), 0.5, 1.0e-6);
    CHECK_NEAR(probability(result, 0, 0, 1, 1), 0.5, 1.0e-6);
    CHECK_NEAR(result.output.at({0, 0, 0}), 2.0, 1.0e-6);
    CHECK_NEAR(result.output.at({0, 0, 1}), 4.0, 1.0e-6);
    CHECK_NEAR(result.output.at({0, 1, 0}), 4.0, 1.0e-6);
    CHECK_NEAR(result.output.at({0, 1, 1}), 6.0, 1.0e-6);
}

void test_scaled_dot_product_attention() {
    CausalSelfAttention attention(4, 1);
    set_identity(attention.q_weights());
    set_identity(attention.k_weights());
    set_identity(attention.v_weights());
    set_identity(attention.output_weights());
    const Tensor input(
        {1, 2, 4}, std::vector<float>{2, 0, 0, 0, 0, 2, 0, 0});

    const auto result = attention.forward_with_details(input);
    const double expected_first = 1.0 / (1.0 + std::exp(2.0));
    const double expected_second = 1.0 - expected_first;
    CHECK_NEAR(probability(result, 0, 0, 1, 0), expected_first, 1.0e-6);
    CHECK_NEAR(probability(result, 0, 0, 1, 1), expected_second, 1.0e-6);
    CHECK_NEAR(result.output.at({0, 1, 0}), 2.0 * expected_first, 1.0e-6);
    CHECK_NEAR(result.output.at({0, 1, 1}), 2.0 * expected_second, 1.0e-6);
    CHECK_NEAR(result.output.at({0, 1, 2}), 0.0, 1.0e-6);
}

void test_softmax_is_stable_with_large_scores() {
    CausalSelfAttention attention(1, 1);
    set_identity(attention.q_weights());
    set_identity(attention.k_weights());
    set_identity(attention.v_weights());
    set_identity(attention.output_weights());
    const Tensor input({1, 2, 1}, std::vector<float>{1000, 1000});

    const auto result = attention.forward_with_details(input);
    CHECK_NEAR(probability(result, 0, 0, 1, 0), 0.5, 1.0e-6);
    CHECK_NEAR(probability(result, 0, 0, 1, 1), 0.5, 1.0e-6);
    CHECK_NEAR(result.output.at({0, 0, 0}), 1000.0, 1.0e-3);
    CHECK_NEAR(result.output.at({0, 1, 0}), 1000.0, 1.0e-3);
    for (const float value : result.output.data()) {
        CHECK(std::isfinite(value));
    }
}

void test_causal_mask_and_probability_sums() {
    const CausalSelfAttention attention(8, 2, 19);
    const Tensor input = Tensor::random({2, 5, 8}, 23, -1.0F, 1.0F);
    const auto result = attention.forward_with_details(input);

    for (std::size_t batch = 0; batch < 2; ++batch) {
        for (std::size_t head = 0; head < 2; ++head) {
            for (std::size_t query = 0; query < 5; ++query) {
                float sum = 0.0F;
                for (std::size_t key = 0; key < 5; ++key) {
                    const float value = probability(result, batch, head, query, key);
                    if (key > query) {
                        CHECK(value == 0.0F);
                    } else {
                        CHECK(value > 0.0F);
                        sum += value;
                    }
                }
                CHECK_NEAR(sum, 1.0, 1.0e-5);
            }
        }
    }
}

void test_future_tokens_do_not_change_earlier_outputs() {
    const CausalSelfAttention attention(8, 2, 29);
    const Tensor original = Tensor::random({1, 4, 8}, 31, -1.0F, 1.0F);
    Tensor modified = original;
    for (std::size_t position = 2; position < 4; ++position) {
        for (std::size_t dimension = 0; dimension < 8; ++dimension) {
            modified.at({0, position, dimension}) +=
                static_cast<float>(position * 8 + dimension + 1);
        }
    }

    const Tensor first = attention.forward(original);
    const Tensor second = attention.forward(modified);
    for (std::size_t position = 0; position < 2; ++position) {
        for (std::size_t dimension = 0; dimension < 8; ++dimension) {
            CHECK_NEAR(
                first.at({0, position, dimension}),
                second.at({0, position, dimension}),
                0.0);
        }
    }
    CHECK(!same_values(first, second));
}

void test_empty_inputs() {
    const CausalSelfAttention attention(4, 2);
    const auto empty_sequence = attention.forward_with_details(Tensor::zeros({1, 0, 4}));
    check_shape(empty_sequence.output, {1, 0, 4});
    check_shape(empty_sequence.queries, {1, 2, 0, 2});
    check_shape(empty_sequence.probabilities, {1, 2, 0, 0});
    CHECK(empty_sequence.output.size() == 0);

    const Tensor empty_batch = attention.forward(Tensor::zeros({0, 3, 4}));
    check_shape(empty_batch, {0, 3, 4});
    CHECK(empty_batch.size() == 0);
}

void test_persistence_preserves_parameters_and_results() {
    const ScopedFile file(
        std::filesystem::current_path() / "mini_gpt_attention_persistence_test.bin");
    CausalSelfAttention attention(4, 2, 987);
    attention.q_weights().data().front() = 0.123456789F;
    attention.k_weights().data().back() = -0.0F;
    attention.v_weights().data().front() = -0.75F;
    attention.output_weights().data().back() = 0.987654321F;
    attention.save(file.path());

    const CausalSelfAttention loaded = CausalSelfAttention::load(file.path());
    CHECK(loaded.embedding_dim() == 4);
    CHECK(loaded.num_heads() == 2);
    CHECK(loaded.head_dim() == 2);
    CHECK(same_bits(loaded.q_weights(), attention.q_weights()));
    CHECK(same_bits(loaded.k_weights(), attention.k_weights()));
    CHECK(same_bits(loaded.v_weights(), attention.v_weights()));
    CHECK(same_bits(loaded.output_weights(), attention.output_weights()));

    const Tensor input = Tensor::random({2, 3, 4}, 41, -1.0F, 1.0F);
    const auto expected = attention.forward_with_details(input);
    const auto actual = loaded.forward_with_details(input);
    CHECK(same_values(actual.output, expected.output));
    CHECK(same_values(actual.queries, expected.queries));
    CHECK(same_values(actual.keys, expected.keys));
    CHECK(same_values(actual.values, expected.values));
    CHECK(same_values(actual.probabilities, expected.probabilities));
}

void test_load_rejects_missing_invalid_and_truncated_files() {
    const ScopedFile file(
        std::filesystem::current_path() / "mini_gpt_attention_invalid_test.bin");
    CHECK_THROWS(std::runtime_error, CausalSelfAttention::load(file.path()));

    {
        std::ofstream output(file.path(), std::ios::binary | std::ios::trunc);
        output << "not attention";
    }
    CHECK_THROWS(std::runtime_error, CausalSelfAttention::load(file.path()));

    const CausalSelfAttention attention(4, 2);
    attention.save(file.path());
    std::filesystem::resize_file(file.path(), 20);
    CHECK_THROWS(std::runtime_error, CausalSelfAttention::load(file.path()));
}

void test_load_rejects_invalid_dimensions() {
    const ScopedFile file(
        std::filesystem::current_path() / "mini_gpt_attention_dimensions_test.bin");
    const CausalSelfAttention attention(4, 2);
    attention.save(file.path());

    {
        std::fstream stream(file.path(), std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(20);
        const std::uint64_t zero = 0;
        stream.write(
            reinterpret_cast<const char*>(&zero),
            static_cast<std::streamsize>(sizeof(zero)));
    }
    CHECK_THROWS(std::runtime_error, CausalSelfAttention::load(file.path()));
}

void test_controlled_qkv_values_logits_and_future_independence() {
    const float log3 = std::log(3.0F);
    const float log9 = std::log(9.0F);
    const Tensor queries({1, 1, 3, 1}, std::vector<float>{1, 1, 1});
    const Tensor keys({1, 1, 3, 1}, std::vector<float>{0, log3, log9});
    const Tensor values({1, 1, 3, 1}, std::vector<float>{10, 20, 30});
    const CausalSelfAttention attention(1, 1, 123);

    const auto result = attention.forward_from_qkv(queries, keys, values);
    const auto print_matrix = [](const Tensor& matrix, const char* label) {
        std::cout << "\n" << label << ":\n";
        for (std::size_t query = 0; query < 3; ++query) {
            std::cout << "  ";
            for (std::size_t key = 0; key < 3; ++key) {
                const float value = matrix.at({0, 0, query, key});
                if (std::isinf(value) && value < 0.0F) {
                    std::cout << "-inf";
                } else {
                    std::cout << value;
                }
                if (key + 1 < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << '\n';
        }
    };
    print_matrix(result.logits, "Logits controlados antes de softmax");
    print_matrix(result.probabilities, "Probabilidades controladas");
    std::cout << "Contexto antes de la proyeccion final:\n";
    for (std::size_t position = 0; position < 3; ++position) {
        std::cout << "  " << result.context.at({0, position, 0}) << '\n';
    }

    const float expected_attention[3][3] = {
        {1.0F, 0.0F, 0.0F},
        {0.25F, 0.75F, 0.0F},
        {1.0F / 13.0F, 3.0F / 13.0F, 9.0F / 13.0F},
    };
    const float expected_context[3] = {10.0F, 17.5F, 340.0F / 13.0F};
    const float expected_keys[3] = {0.0F, log3, log9};
    for (std::size_t query = 0; query < 3; ++query) {
        float row_sum = 0.0F;
        for (std::size_t key = 0; key < 3; ++key) {
            const float attention_value = probability(result, 0, 0, query, key);
            CHECK_NEAR(attention_value, expected_attention[query][key], 1.0e-5);
            const float logit = result.logits.at({0, 0, query, key});
            if (key > query) {
                CHECK(attention_value == 0.0F);
                CHECK(std::isinf(logit) && logit < 0.0F);
            } else {
                CHECK_NEAR(logit, expected_keys[key], 1.0e-5);
                row_sum += attention_value;
            }
        }
        CHECK_NEAR(row_sum, 1.0, 1.0e-5);
        CHECK_NEAR(result.context.at({0, query, 0}), expected_context[query], 1.0e-5);
    }

    Tensor modified_keys = keys;
    Tensor modified_values = values;
    modified_keys.at({0, 0, 2, 0}) = std::log(1000.0F);
    modified_values.at({0, 0, 2, 0}) = 1000.0F;
    const auto modified = attention.forward_from_qkv(queries, modified_keys, modified_values);
    for (std::size_t position = 0; position < 2; ++position) {
        CHECK(
            result.context.at({0, position, 0}) ==
            modified.context.at({0, position, 0}));
    }
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"controlled QKV values logits and future independence", test_controlled_qkv_values_logits_and_future_independence},
        {"default configuration", test_default_configuration},
        {"invalid configuration", test_invalid_configuration},
        {"initialization is reproducible", test_initialization_is_reproducible},
        {"output QKV and attention shapes", test_output_qkv_and_attention_shapes},
        {"input validation", test_input_validation},
        {"projections and head split", test_projections_and_head_split},
        {"uniform causal attention and output projection", test_uniform_causal_attention_and_output_projection},
        {"scaled dot-product attention", test_scaled_dot_product_attention},
        {"softmax is stable with large scores", test_softmax_is_stable_with_large_scores},
        {"causal mask and probability sums", test_causal_mask_and_probability_sums},
        {"future tokens do not change earlier outputs", test_future_tokens_do_not_change_earlier_outputs},
        {"empty inputs", test_empty_inputs},
        {"persistence preserves parameters and results", test_persistence_preserves_parameters_and_results},
        {"load rejects missing invalid and truncated files", test_load_rejects_missing_invalid_and_truncated_files},
        {"load rejects invalid dimensions", test_load_rejects_invalid_dimensions},
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

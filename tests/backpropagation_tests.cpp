#include "mini_gpt/backpropagation.hpp"
#include "mini_gpt/cross_entropy.hpp"
#include "mini_gpt/gpt.hpp"

#include <algorithm>
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

void test_linear_backward_known_values() {
    const Tensor input({1, 2, 2}, std::vector<float>{1, 2, 3, 4});
    const Tensor weights({2, 2}, std::vector<float>{1, 2, 3, 4});
    const Tensor output_gradient = Tensor::ones({1, 2, 2});
    const auto result = mini_gpt::linear_backward(input, weights, output_gradient);

    CHECK(result.input_gradient.shape() == input.shape());
    CHECK(result.weight_gradient.shape() == weights.shape());
    CHECK_NEAR(result.weight_gradient.at({0, 0}), 4.0, 1.0e-6);
    CHECK_NEAR(result.weight_gradient.at({0, 1}), 4.0, 1.0e-6);
    CHECK_NEAR(result.weight_gradient.at({1, 0}), 6.0, 1.0e-6);
    CHECK_NEAR(result.weight_gradient.at({1, 1}), 6.0, 1.0e-6);
    CHECK_NEAR(result.input_gradient.at({0, 0, 0}), 3.0, 1.0e-6);
    CHECK_NEAR(result.input_gradient.at({0, 0, 1}), 7.0, 1.0e-6);
    CHECK_NEAR(result.input_gradient.at({0, 1, 0}), 3.0, 1.0e-6);
    CHECK_NEAR(result.input_gradient.at({0, 1, 1}), 7.0, 1.0e-6);
}

void test_cross_entropy_backward_uniform_and_average() {
    const Tensor logits = Tensor::zeros({1, 2, 3});
    const std::vector<std::uint32_t> targets = {1, 2};
    const Tensor gradient = mini_gpt::cross_entropy_backward(logits, targets);
    CHECK(gradient.shape() == logits.shape());
    for (std::size_t position = 0; position < 2; ++position) {
        for (std::size_t token = 0; token < 3; ++token) {
            const double expected = token == targets[position] ? -2.0 / 6.0 : 1.0 / 6.0;
            CHECK_NEAR(gradient.at({0, position, token}), expected, 1.0e-6);
        }
    }
}

void test_cross_entropy_backward_mask_and_extreme_logits() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const Tensor logits(
        {1, 2, 3}, std::vector<float>{10000, 9999, -10000, nan, nan, nan});
    const std::vector<std::uint32_t> targets = {1, 99};
    const std::vector<std::uint8_t> mask = {1, 0};
    const Tensor gradient = mini_gpt::cross_entropy_backward(logits, targets, mask);

    CHECK_NEAR(gradient.at({0, 0, 0}), 0.7310585786, 1.0e-5);
    CHECK_NEAR(gradient.at({0, 0, 1}), -0.7310585786, 1.0e-5);
    CHECK_NEAR(gradient.at({0, 0, 2}), 0.0, 1.0e-5);
    for (std::size_t token = 0; token < 3; ++token) {
        CHECK(gradient.at({0, 1, token}) == 0.0F);
    }
}

void test_embedding_backward_accumulates_repeated_ids() {
    const mini_gpt::Embedding embedding(5, 3, 4);
    const std::vector<std::uint32_t> ids = {2, 2, 2};
    const Tensor output_gradient = Tensor::ones({1, 3, 3});
    const auto gradients = mini_gpt::embedding_backward(embedding, ids, 1, 3, output_gradient);

    for (std::size_t i = 0; i < 3; ++i) {
        CHECK_NEAR(gradients.token_embeddings.at({2, i}), 3.0, 1.0e-6);
        for (std::size_t position = 0; position < 3; ++position) {
            CHECK_NEAR(gradients.positional_embeddings.at({position, i}), 1.0, 1.0e-6);
        }
    }
    CHECK(gradients.token_embeddings.at({1, 0}) == 0.0F);
}

void test_layer_norm_backward_parameters_and_input() {
    const mini_gpt::LayerNorm norm(3);
    const Tensor input({1, 3}, std::vector<float>{1, 2, 3});
    const Tensor output_gradient({1, 3}, std::vector<float>{0.5F, -1.0F, 2.0F});
    const auto result = mini_gpt::layer_norm_backward(norm, input, output_gradient);
    const Tensor normalized = norm.forward(input);

    for (std::size_t i = 0; i < 3; ++i) {
        CHECK_NEAR(
            result.parameters.gamma.at({i}),
            output_gradient.at({0, i}) * normalized.at({0, i}),
            1.0e-5);
        CHECK_NEAR(result.parameters.beta.at({i}), output_gradient.at({0, i}), 1.0e-6);
        CHECK(std::isfinite(result.input_gradient.at({0, i})));
    }
    CHECK_NEAR(
        result.input_gradient.at({0, 1}), -1.5 * normalized.at({0, 2}), 1.0e-5);
}

void test_gelu_backward_reference_values() {
    const Tensor input({3}, std::vector<float>{0, 1, -1});
    const Tensor output_gradient({3}, std::vector<float>{2, 2, 2});
    const Tensor gradient = mini_gpt::gelu_backward(input, output_gradient);
    CHECK_NEAR(gradient.at({0}), 1.0, 1.0e-6);
    CHECK_NEAR(gradient.at({1}), 2.0 * 1.0833154708, 1.0e-5);
    CHECK_NEAR(gradient.at({2}), 2.0 * -0.0833154708, 1.0e-5);
}

void test_attention_backward_blocks_future_positions() {
    const mini_gpt::CausalSelfAttention attention(4, 2, 71);
    const Tensor input = Tensor::random({1, 3, 4}, 73, -1.0F, 1.0F);
    Tensor output_gradient = Tensor::zeros({1, 3, 4});
    for (std::size_t i = 0; i < 4; ++i) {
        output_gradient.at({0, 0, i}) = 1.0F;
    }

    const auto result = mini_gpt::attention_backward(attention, input, output_gradient);
    for (std::size_t position = 1; position < 3; ++position) {
        for (std::size_t dimension = 0; dimension < 4; ++dimension) {
            CHECK(result.input_gradient.at({0, position, dimension}) == 0.0F);
        }
    }
}

double attention_scalar_loss(
    const mini_gpt::CausalSelfAttention& attention,
    const Tensor& input,
    const Tensor& output_gradient) {
    const Tensor output = attention.forward(input);
    double loss = 0.0;
    for (std::size_t i = 0; i < output.size(); ++i) {
        loss += static_cast<double>(output.data()[i]) * output_gradient.data()[i];
    }
    return loss;
}

void check_attention_gradient(
    const mini_gpt::CausalSelfAttention& attention,
    const Tensor& input,
    const Tensor& output_gradient,
    float& parameter,
    double analytical) {
    constexpr float epsilon = 1.0e-3F;
    const float original = parameter;
    parameter = original + epsilon;
    const double plus = attention_scalar_loss(attention, input, output_gradient);
    parameter = original - epsilon;
    const double minus = attention_scalar_loss(attention, input, output_gradient);
    parameter = original;
    const double numerical = (plus - minus) / (2.0 * epsilon);
    const double tolerance = 3.0e-3 + 3.0e-2 * std::max(std::fabs(numerical), std::fabs(analytical));
    CHECK_NEAR(analytical, numerical, tolerance);
}

void test_attention_backward_gradient_check() {
    mini_gpt::CausalSelfAttention attention(4, 2, 79);
    Tensor input = Tensor::random({1, 3, 4}, 83, -1.0F, 1.0F);
    const Tensor output_gradient = Tensor::random({1, 3, 4}, 89, -1.0F, 1.0F);
    const auto result = mini_gpt::attention_backward(attention, input, output_gradient);

    check_attention_gradient(
        attention, input, output_gradient,
        attention.q_weights().at({0, 0}), result.parameters.q_weights.at({0, 0}));
    check_attention_gradient(
        attention, input, output_gradient,
        attention.k_weights().at({0, 0}), result.parameters.k_weights.at({0, 0}));
    check_attention_gradient(
        attention, input, output_gradient,
        attention.v_weights().at({0, 0}), result.parameters.v_weights.at({0, 0}));
    check_attention_gradient(
        attention, input, output_gradient,
        attention.output_weights().at({0, 0}), result.parameters.output_weights.at({0, 0}));
    check_attention_gradient(
        attention, input, output_gradient,
        input.at({0, 0, 0}), result.input_gradient.at({0, 0, 0}));
}

std::vector<float> model_parameter_values(const mini_gpt::GPT& model) {
    std::vector<float> values;
    const auto append = [&](const Tensor& tensor) {
        values.insert(values.end(), tensor.data().begin(), tensor.data().end());
    };
    append(model.embedding().token_embeddings());
    append(model.embedding().positional_embeddings());
    for (const auto& block : model.blocks()) {
        for (const Tensor* tensor : {
                 &block.norm1().gamma(), &block.norm1().beta(),
                 &block.norm2().gamma(), &block.norm2().beta(),
                 &block.attention().q_weights(), &block.attention().k_weights(),
                 &block.attention().v_weights(), &block.attention().output_weights(),
                 &block.fc1_weights(), &block.fc2_weights()}) {
            append(*tensor);
        }
    }
    append(model.final_norm().gamma());
    append(model.final_norm().beta());
    append(model.output_weights());
    return values;
}

bool nonzero_gradient(const Tensor& tensor) {
    return std::any_of(tensor.data().begin(), tensor.data().end(), [](float value) {
        return value != 0.0F;
    });
}

void check_gradients(const mini_gpt::GPT& model, bool require_zero) {
    const auto check_tensor = [&](const Tensor& gradient, const Tensor& parameter) {
        CHECK(gradient.shape() == parameter.shape());
        for (const float value : gradient.data()) {
            CHECK(std::isfinite(value));
            if (require_zero) {
                CHECK(value == 0.0F);
            }
        }
    };
    const auto check_block = [&](const mini_gpt::TransformerBlockGradients& gradient,
                                 const mini_gpt::TransformerBlock& block) {
        check_tensor(gradient.norm1.gamma, block.norm1().gamma());
        check_tensor(gradient.norm1.beta, block.norm1().beta());
        check_tensor(gradient.norm2.gamma, block.norm2().gamma());
        check_tensor(gradient.norm2.beta, block.norm2().beta());
        check_tensor(gradient.attention.q_weights, block.attention().q_weights());
        check_tensor(gradient.attention.k_weights, block.attention().k_weights());
        check_tensor(gradient.attention.v_weights, block.attention().v_weights());
        check_tensor(gradient.attention.output_weights, block.attention().output_weights());
        check_tensor(gradient.fc1_weights, block.fc1_weights());
        check_tensor(gradient.fc2_weights, block.fc2_weights());
    };

    const auto& gradients = model.gradients();
    check_tensor(gradients.embedding.token_embeddings, model.embedding().token_embeddings());
    check_tensor(gradients.embedding.positional_embeddings, model.embedding().positional_embeddings());
    CHECK(gradients.blocks.size() == model.blocks().size());
    for (std::size_t i = 0; i < gradients.blocks.size(); ++i) {
        check_block(gradients.blocks[i], model.blocks()[i]);
    }
    check_tensor(gradients.final_norm.gamma, model.final_norm().gamma());
    check_tensor(gradients.final_norm.beta, model.final_norm().beta());
    check_tensor(gradients.output_weights, model.output_weights());
}

void test_gpt_backward_accumulation_zeroing_and_parameter_integrity() {
    mini_gpt::GPT model(12, 8, 2, 2, 16, 5, 101);
    const std::vector<std::uint32_t> inputs = {1, 2, 1, 3, 2, 1};
    const std::vector<std::uint32_t> targets = {2, 1, 3, 1, 2, 1};
    const auto original_parameters = model_parameter_values(model);

    model.zero_grad();
    const float first_loss = model.backward(inputs, 2, 3, targets);
    CHECK(std::isfinite(first_loss));
    check_gradients(model, false);
    CHECK(nonzero_gradient(model.gradients().output_weights));
    CHECK(nonzero_gradient(model.gradients().embedding.token_embeddings));
    CHECK(nonzero_gradient(model.gradients().embedding.positional_embeddings));
    CHECK(nonzero_gradient(model.gradients().blocks[0].attention.q_weights));
    CHECK(nonzero_gradient(model.gradients().blocks[1].fc1_weights));
    CHECK(nonzero_gradient(model.gradients().final_norm.gamma));

    const Tensor first_output_gradient = model.gradients().output_weights;
    const Tensor first_embedding_gradient = model.gradients().embedding.token_embeddings;
    const float second_loss = model.backward(inputs, 2, 3, targets);
    CHECK_NEAR(second_loss, first_loss, 0.0);
    for (std::size_t i = 0; i < first_output_gradient.size(); ++i) {
        CHECK_NEAR(
            model.gradients().output_weights.data()[i],
            2.0F * first_output_gradient.data()[i],
            1.0e-5);
    }
    for (std::size_t i = 0; i < first_embedding_gradient.size(); ++i) {
        CHECK_NEAR(
            model.gradients().embedding.token_embeddings.data()[i],
            2.0F * first_embedding_gradient.data()[i],
            1.0e-5);
    }

    CHECK(model_parameter_values(model) == original_parameters);
    model.zero_grad();
    check_gradients(model, true);
    CHECK(model_parameter_values(model) == original_parameters);
}

double gpt_reference_loss(
    const mini_gpt::GPT& model,
    std::span<const std::uint32_t> inputs,
    std::size_t batch_size,
    std::size_t seq_len,
    std::span<const std::uint32_t> targets,
    std::span<const std::uint8_t> mask) {
    const Tensor logits = model.forward(inputs, batch_size, seq_len);
    const std::size_t vocab_size = model.vocab_size();
    const std::size_t positions = batch_size * seq_len;
    std::size_t valid_positions = 0;
    for (std::size_t position = 0; position < positions; ++position) {
        if (mask.empty() || mask[position] != 0) {
            ++valid_positions;
        }
    }

    double loss = 0.0;
    for (std::size_t position = 0; position < positions; ++position) {
        if (!mask.empty() && mask[position] == 0) {
            continue;
        }
        const std::size_t offset = position * vocab_size;
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t token = 0; token < vocab_size; ++token) {
            maximum = std::max(maximum, static_cast<double>(logits.data()[offset + token]));
        }
        double denominator = 0.0;
        for (std::size_t token = 0; token < vocab_size; ++token) {
            denominator += std::exp(logits.data()[offset + token] - maximum);
        }
        loss += maximum + std::log(denominator) - logits.data()[offset + targets[position]];
    }
    return loss / valid_positions;
}

void check_gpt_gradient(
    mini_gpt::GPT& model,
    std::span<const std::uint32_t> inputs,
    std::size_t batch_size,
    std::size_t seq_len,
    std::span<const std::uint32_t> targets,
    std::span<const std::uint8_t> mask,
    float& parameter,
    double analytical) {
    constexpr float epsilon = 1.0e-3F;
    const float original = parameter;
    parameter = original + epsilon;
    const double plus = gpt_reference_loss(model, inputs, batch_size, seq_len, targets, mask);
    parameter = original - epsilon;
    const double minus = gpt_reference_loss(model, inputs, batch_size, seq_len, targets, mask);
    parameter = original;
    const double numerical = (plus - minus) / (2.0 * epsilon);
    CHECK(std::isfinite(numerical));
    CHECK(std::isfinite(analytical));
    const double tolerance = 3.0e-3 + 3.0e-2 * std::max(std::fabs(numerical), std::fabs(analytical));
    CHECK_NEAR(analytical, numerical, tolerance);
}

void test_gpt_full_gradient_check() {
    mini_gpt::GPT model(12, 8, 2, 2, 16, 5, 131);
    const std::vector<std::uint32_t> inputs = {1, 1, 2};
    const std::vector<std::uint32_t> targets = {2, 3, 4};
    const std::vector<std::uint8_t> mask = {1, 1, 0};
    model.zero_grad();
    model.backward(inputs, 1, 3, targets, mask);

    const auto check = [&](float& parameter, double analytical) {
        check_gpt_gradient(model, inputs, 1, 3, targets, mask, parameter, analytical);
    };
    check(model.embedding().token_embeddings().at({1, 0}),
          model.gradients().embedding.token_embeddings.at({1, 0}));
    check(model.embedding().positional_embeddings().at({0, 0}),
          model.gradients().embedding.positional_embeddings.at({0, 0}));

    auto& block0 = model.blocks()[0];
    const auto& gradient0 = model.gradients().blocks[0];
    check(block0.norm1().gamma().at({0}), gradient0.norm1.gamma.at({0}));
    check(block0.norm1().beta().at({0}), gradient0.norm1.beta.at({0}));
    check(block0.norm2().gamma().at({0}), gradient0.norm2.gamma.at({0}));
    check(block0.norm2().beta().at({0}), gradient0.norm2.beta.at({0}));
    check(block0.attention().q_weights().at({0, 0}), gradient0.attention.q_weights.at({0, 0}));
    check(block0.attention().k_weights().at({0, 0}), gradient0.attention.k_weights.at({0, 0}));
    check(block0.attention().v_weights().at({0, 0}), gradient0.attention.v_weights.at({0, 0}));
    check(block0.attention().output_weights().at({0, 0}), gradient0.attention.output_weights.at({0, 0}));
    check(block0.fc1_weights().at({0, 0}), gradient0.fc1_weights.at({0, 0}));
    check(block0.fc2_weights().at({0, 0}), gradient0.fc2_weights.at({0, 0}));

    check(model.blocks()[1].attention().q_weights().at({0, 0}),
          model.gradients().blocks[1].attention.q_weights.at({0, 0}));
    check(model.final_norm().gamma().at({0}), model.gradients().final_norm.gamma.at({0}));
    check(model.final_norm().beta().at({0}), model.gradients().final_norm.beta.at({0}));
    check(model.output_weights().at({0, 0}), model.gradients().output_weights.at({0, 0}));
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"GPT backward accumulation zeroing and parameter integrity", test_gpt_backward_accumulation_zeroing_and_parameter_integrity},
        {"GPT full gradient check", test_gpt_full_gradient_check},
        {"attention backward blocks future positions", test_attention_backward_blocks_future_positions},
        {"attention backward gradient check", test_attention_backward_gradient_check},
        {"embedding backward accumulates repeated IDs", test_embedding_backward_accumulates_repeated_ids},
        {"LayerNorm backward parameters and input", test_layer_norm_backward_parameters_and_input},
        {"GELU backward reference values", test_gelu_backward_reference_values},
        {"linear backward known values", test_linear_backward_known_values},
        {"Cross-Entropy backward uniform and average", test_cross_entropy_backward_uniform_and_average},
        {"Cross-Entropy backward mask and extreme logits", test_cross_entropy_backward_mask_and_extreme_logits},
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

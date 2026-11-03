#include "mini_gpt/gpt.hpp"

#include "mini_gpt/bpe_tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using mini_gpt::BPETokenizer;
using mini_gpt::GPT;
using mini_gpt::GenerationConfig;
using mini_gpt::SamplingStrategy;
using mini_gpt::StopCondition;
using mini_gpt::Tensor;
using mini_gpt::TransformerBlock;
using TokenId = GPT::TokenId;

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

void force_greedy_token(GPT& model, TokenId id) {
    set_zero(model.final_norm().gamma());
    std::fill(
        model.final_norm().beta().data().begin(),
        model.final_norm().beta().data().end(),
        1.0F);
    set_zero(model.output_weights());
    for (std::size_t dimension = 0; dimension < model.embedding_dim(); ++dimension) {
        model.output_weights().at({dimension, id}) = 1.0F;
    }
}

void check_block_bits(const TransformerBlock& lhs, const TransformerBlock& rhs) {
    CHECK(same_bits(lhs.norm1().gamma(), rhs.norm1().gamma()));
    CHECK(same_bits(lhs.norm1().beta(), rhs.norm1().beta()));
    CHECK(same_bits(lhs.norm2().gamma(), rhs.norm2().gamma()));
    CHECK(same_bits(lhs.norm2().beta(), rhs.norm2().beta()));
    CHECK(same_bits(lhs.attention().q_weights(), rhs.attention().q_weights()));
    CHECK(same_bits(lhs.attention().k_weights(), rhs.attention().k_weights()));
    CHECK(same_bits(lhs.attention().v_weights(), rhs.attention().v_weights()));
    CHECK(same_bits(lhs.attention().output_weights(), rhs.attention().output_weights()));
    CHECK(same_bits(lhs.fc1_weights(), rhs.fc1_weights()));
    CHECK(same_bits(lhs.fc2_weights(), rhs.fc2_weights()));
}

void check_model_bits(const GPT& lhs, const GPT& rhs) {
    CHECK(same_bits(lhs.embedding().token_embeddings(), rhs.embedding().token_embeddings()));
    CHECK(same_bits(
        lhs.embedding().positional_embeddings(), rhs.embedding().positional_embeddings()));
    CHECK(lhs.blocks().size() == rhs.blocks().size());
    for (std::size_t i = 0; i < lhs.blocks().size(); ++i) {
        check_block_bits(lhs.blocks()[i], rhs.blocks()[i]);
    }
    CHECK(same_bits(lhs.final_norm().gamma(), rhs.final_norm().gamma()));
    CHECK(same_bits(lhs.final_norm().beta(), rhs.final_norm().beta()));
    CHECK(same_bits(lhs.output_weights(), rhs.output_weights()));
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
    const GPT model(300);
    CHECK(model.vocab_size() == 300);
    CHECK(model.embedding_dim() == 256);
    CHECK(model.num_heads() == 4);
    CHECK(model.num_layers() == 4);
    CHECK(model.mlp_hidden_dim() == 1024);
    CHECK(model.max_seq_len() == 256);
    CHECK(model.blocks().size() == 4);
    check_shape(model.output_weights(), {256, 300});
    check_shape(model.embedding().token_embeddings(), {300, 256});
    check_shape(model.embedding().positional_embeddings(), {256, 256});
}

void test_logits_shapes_for_batches_sequences_and_empty_inputs() {
    const GPT model(8, 4, 2, 2, 8, 4, 17);
    for (const auto& dimensions : {std::pair<std::size_t, std::size_t>{1, 1}, {2, 3}}) {
        const std::size_t batch = dimensions.first;
        const std::size_t sequence = dimensions.second;
        const std::vector<TokenId> ids(batch * sequence, 1);
        const Tensor logits = model.forward(ids, batch, sequence);
        check_shape(logits, {batch, sequence, 8});
        CHECK(logits.size() == batch * sequence * 8);
    }

    const std::vector<TokenId> empty;
    check_shape(model.forward(empty, 1, 0), {1, 0, 8});
    check_shape(model.forward(empty, 0, 2), {0, 2, 8});
}

void test_multiple_blocks_have_independent_parameters() {
    GPT model(8, 4, 2, 3, 8, 4, 23);
    CHECK(model.blocks().size() == 3);
    CHECK(!same_values(
        model.blocks()[0].attention().q_weights(),
        model.blocks()[1].attention().q_weights()));
    const float second_layer_value = model.blocks()[1].attention().q_weights().data().front();
    model.blocks()[0].attention().q_weights().data().front() += 1.0F;
    CHECK(model.blocks()[1].attention().q_weights().data().front() == second_layer_value);

    const std::vector<TokenId> ids = {1, 2};
    check_shape(model.forward(ids, 1, 2), {1, 2, 8});
}

void test_controlled_vocabulary_projection() {
    GPT model(5, 4, 2, 1, 8, 4, 29);
    const std::vector<TokenId> ids = {1, 2, 3};
    Tensor features = model.embedding().forward(ids, 1, 3);
    for (const TransformerBlock& block : model.blocks()) {
        features = block.forward(features);
    }
    features = model.final_norm().forward(features);

    set_zero(model.output_weights());
    model.output_weights().at({0, 2}) = 1.0F;
    const Tensor logits = model.forward(ids, 1, 3);
    check_shape(logits, {1, 3, 5});
    for (std::size_t position = 0; position < 3; ++position) {
        CHECK_NEAR(logits.at({0, position, 2}), features.at({0, position, 0}), 1.0e-6);
        for (std::size_t token = 0; token < 5; ++token) {
            if (token != 2) {
                CHECK(logits.at({0, position, token}) == 0.0F);
            }
        }
    }
}

void test_forward_is_causal() {
    const GPT model(10, 4, 2, 2, 8, 4, 31);
    const std::vector<TokenId> original = {1, 2, 3, 4};
    const std::vector<TokenId> modified = {1, 2, 7, 8};
    const Tensor first = model.forward(original, 1, 4);
    const Tensor second = model.forward(modified, 1, 4);

    for (std::size_t position = 0; position < 2; ++position) {
        for (std::size_t token = 0; token < 10; ++token) {
            CHECK_NEAR(
                first.at({0, position, token}),
                second.at({0, position, token}),
                0.0);
        }
    }
    CHECK(!same_values(first, second));
}

void test_controlled_greedy_selection() {
    const std::vector<float> logits = {0.5F, 3.0F, 3.0F, -1.0F};
    std::mt19937 generator(1);
    CHECK(GPT::select_token(logits, 1.0F, SamplingStrategy::greedy, generator) == 1);
    const std::vector<float> empty;
    CHECK_THROWS(
        std::invalid_argument,
        GPT::select_token(empty, 1.0F, SamplingStrategy::greedy, generator));
}

void test_softmax_sampling_temperature_and_reproducibility() {
    const std::vector<float> logits = {10000.0F, 9999.0F, -10000.0F};
    const auto probabilities = GPT::softmax(logits, 1.0F);
    CHECK_NEAR(probabilities[0], 0.7310585786, 1.0e-5);
    CHECK_NEAR(probabilities[1], 0.2689414214, 1.0e-5);
    CHECK_NEAR(probabilities[2], 0.0, 1.0e-6);
    CHECK_NEAR(probabilities[0] + probabilities[1] + probabilities[2], 1.0, 1.0e-5);

    std::mt19937 low_temperature_generator(7);
    CHECK(GPT::select_token(logits, 1.0e-3F, SamplingStrategy::sampling, low_temperature_generator) == 0);
    std::mt19937 first(123);
    std::mt19937 second(123);
    for (std::size_t i = 0; i < 20; ++i) {
        CHECK(
            GPT::select_token(logits, 0.7F, SamplingStrategy::sampling, first) ==
            GPT::select_token(logits, 0.7F, SamplingStrategy::sampling, second));
    }

    const std::vector<float> binary_logits = {0.0F, 1.0F};
    std::mt19937 high_temperature_generator(99);
    bool saw_zero = false;
    bool saw_one = false;
    for (std::size_t i = 0; i < 100; ++i) {
        const TokenId selected = GPT::select_token(
            binary_logits, 1.0e6F, SamplingStrategy::sampling, high_temperature_generator);
        saw_zero = saw_zero || selected == 0;
        saw_one = saw_one || selected == 1;
    }
    CHECK(saw_zero);
    CHECK(saw_one);
    CHECK_THROWS(std::invalid_argument, GPT::softmax(logits, 0.0F));
}

void test_initialization_is_reproducible() {
    const GPT first(8, 4, 2, 2, 8, 4, 123);
    const GPT second(8, 4, 2, 2, 8, 4, 123);
    const GPT different(8, 4, 2, 2, 8, 4, 124);
    CHECK(same_values(first.embedding().token_embeddings(), second.embedding().token_embeddings()));
    CHECK(same_values(first.blocks()[0].attention().q_weights(), second.blocks()[0].attention().q_weights()));
    CHECK(same_values(first.blocks()[1].fc1_weights(), second.blocks()[1].fc1_weights()));
    CHECK(same_values(first.output_weights(), second.output_weights()));
    CHECK(!same_values(first.output_weights(), different.output_weights()));
}

void test_generate_multiple_tokens() {
    GPT model(260, 4, 2, 1, 8, 4, 41);
    force_greedy_token(model, 5);
    const std::vector<TokenId> prompt = {1, 2};
    GenerationConfig config;
    config.max_new_tokens = 4;
    config.strategy = SamplingStrategy::greedy;
    const auto generated = model.generate(prompt, config);
    CHECK(generated == std::vector<TokenId>({5, 5, 5, 5}));

    config.max_new_tokens = 0;
    CHECK(model.generate(prompt, config).empty());
}

void test_generation_respects_context_limit() {
    GPT model(260, 4, 2, 1, 8, 3, 43);
    force_greedy_token(model, 7);
    const std::vector<TokenId> prompt = {1, 2, 3, 4, 5};
    GenerationConfig config;
    config.max_new_tokens = 5;
    const auto generated = model.generate(prompt, config);
    CHECK(generated == std::vector<TokenId>({7, 7, 7, 7, 7}));
}

void test_generation_stop_conditions() {
    GPT model(260, 4, 2, 1, 8, 4, 47);
    const std::vector<TokenId> prompt = {1};
    GenerationConfig config;
    config.max_new_tokens = 4;
    config.stop_condition = StopCondition::end_of_turn;
    force_greedy_token(model, BPETokenizer::eot_id);
    CHECK(model.generate(prompt, config) == std::vector<TokenId>({BPETokenizer::eot_id}));

    config.max_new_tokens = 3;
    config.stop_condition = StopCondition::end_of_conversation;
    CHECK(model.generate(prompt, config) == std::vector<TokenId>(
        {BPETokenizer::eot_id, BPETokenizer::eot_id, BPETokenizer::eot_id}));

    force_greedy_token(model, BPETokenizer::end_id);
    CHECK(model.generate(prompt, config) == std::vector<TokenId>({BPETokenizer::end_id}));
}

void test_persistence_preserves_parameters_and_predictions() {
    const ScopedFile file(std::filesystem::current_path() / "mini_gpt_model_persistence_test.bin");
    GPT model(10, 4, 2, 2, 8, 4, 987, 1.0e-4F);
    model.embedding().token_embeddings().data().front() = 0.123456789F;
    model.embedding().positional_embeddings().data().back() = -0.0F;
    model.blocks()[0].norm1().gamma().data().front() = 1.25F;
    model.blocks()[0].attention().q_weights().data().front() = -0.75F;
    model.blocks()[1].fc1_weights().data().front() = 0.5F;
    model.blocks()[1].fc2_weights().data().back() = 0.987654321F;
    model.final_norm().gamma().data().front() = 0.75F;
    model.final_norm().beta().data().back() = 0.25F;
    model.output_weights().data().front() = 0.333333333F;
    model.save(file.path());

    const GPT loaded = GPT::load(file.path());
    CHECK(loaded.vocab_size() == 10);
    CHECK(loaded.embedding_dim() == 4);
    CHECK(loaded.num_heads() == 2);
    CHECK(loaded.num_layers() == 2);
    CHECK(loaded.mlp_hidden_dim() == 8);
    CHECK(loaded.max_seq_len() == 4);
    CHECK_NEAR(loaded.epsilon(), 1.0e-4, 1.0e-8);
    check_model_bits(loaded, model);

    const std::vector<TokenId> ids = {1, 2, 3, 4, 3, 2};
    CHECK(same_values(loaded.forward(ids, 2, 3), model.forward(ids, 2, 3)));
    GenerationConfig config;
    config.max_new_tokens = 3;
    config.strategy = SamplingStrategy::sampling;
    config.temperature = 0.8F;
    config.seed = 77;
    CHECK(loaded.generate(ids, config) == model.generate(ids, config));
}

void test_parameter_count() {
    const GPT model(8, 4, 2, 2, 8, 3, 53);
    const auto count = model.parameter_count();
    CHECK(count.embeddings == 44);
    CHECK(count.transformer_blocks == 288);
    CHECK(count.final_layer_norm == 8);
    CHECK(count.output_projection == 32);
    CHECK(count.total() == 372);
}

void test_invalid_configuration_and_inputs() {
    CHECK_THROWS(std::invalid_argument, GPT(0));
    CHECK_THROWS(std::invalid_argument, GPT(8, 4, 2, 0, 8));
    CHECK_THROWS(std::invalid_argument, GPT(8, 10, 3, 1, 20));
    const GPT model(8, 4, 2, 1, 8, 2);
    const std::vector<TokenId> empty;
    GenerationConfig config;
    CHECK_THROWS(std::invalid_argument, model.generate(empty, config));
    const std::vector<TokenId> too_long = {1, 1, 1};
    CHECK_THROWS(std::invalid_argument, model.forward(too_long, 1, 3));
    const std::vector<TokenId> invalid = {8};
    CHECK_THROWS(std::out_of_range, model.forward(invalid, 1, 1));
}

void test_load_rejects_invalid_files() {
    const ScopedFile file(std::filesystem::current_path() / "mini_gpt_model_invalid_test.bin");
    CHECK_THROWS(std::runtime_error, GPT::load(file.path()));
    {
        std::ofstream output(file.path(), std::ios::binary | std::ios::trunc);
        output << "not a model";
    }
    CHECK_THROWS(std::runtime_error, GPT::load(file.path()));

    const GPT model(8, 4, 2, 1, 8, 3);
    model.save(file.path());
    std::filesystem::resize_file(file.path(), 20);
    CHECK_THROWS(std::runtime_error, GPT::load(file.path()));

    model.save(file.path());
    {
        std::fstream stream(file.path(), std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(12);
        const std::uint64_t zero = 0;
        stream.write(
            reinterpret_cast<const char*>(&zero),
            static_cast<std::streamsize>(sizeof(zero)));
    }
    CHECK_THROWS(std::runtime_error, GPT::load(file.path()));
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"default configuration", test_default_configuration},
        {"logits shapes for batches sequences and empty inputs", test_logits_shapes_for_batches_sequences_and_empty_inputs},
        {"multiple blocks have independent parameters", test_multiple_blocks_have_independent_parameters},
        {"controlled vocabulary projection", test_controlled_vocabulary_projection},
        {"forward is causal", test_forward_is_causal},
        {"controlled greedy selection", test_controlled_greedy_selection},
        {"softmax sampling temperature and reproducibility", test_softmax_sampling_temperature_and_reproducibility},
        {"initialization is reproducible", test_initialization_is_reproducible},
        {"generate multiple tokens", test_generate_multiple_tokens},
        {"generation respects context limit", test_generation_respects_context_limit},
        {"generation stop conditions", test_generation_stop_conditions},
        {"persistence preserves parameters and predictions", test_persistence_preserves_parameters_and_predictions},
        {"parameter count", test_parameter_count},
        {"invalid configuration and inputs", test_invalid_configuration_and_inputs},
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

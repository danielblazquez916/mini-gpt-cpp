#include "mini_gpt/embedding.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using mini_gpt::Embedding;
using mini_gpt::Tensor;
using TokenId = Embedding::TokenId;

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

void check_vector(
    const Tensor& tensor,
    std::size_t batch,
    std::size_t position,
    std::initializer_list<float> expected) {
    const std::size_t dimension = expected.size();
    CHECK(tensor.rank() == 3);
    CHECK(tensor.shape()[2] == dimension);
    const std::size_t offset =
        (batch * tensor.shape()[1] + position) * dimension;
    std::size_t index = 0;
    for (const float value : expected) {
        CHECK(tensor.data()[offset + index] == value);
        ++index;
    }
}

void set_token_row(
    Embedding& embedding,
    TokenId id,
    std::initializer_list<float> values) {
    CHECK(values.size() == embedding.embedding_dim());
    std::copy(
        values.begin(),
        values.end(),
        embedding.token_embeddings().data().begin() +
            static_cast<std::size_t>(id) * embedding.embedding_dim());
}

void set_position_row(
    Embedding& embedding,
    std::size_t position,
    std::initializer_list<float> values) {
    CHECK(values.size() == embedding.embedding_dim());
    std::copy(
        values.begin(),
        values.end(),
        embedding.positional_embeddings().data().begin() +
            position * embedding.embedding_dim());
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

void test_default_configuration_and_shapes() {
    const Embedding embedding(5);
    CHECK(embedding.vocab_size() == 5);
    CHECK(embedding.embedding_dim() == 256);
    CHECK(embedding.max_seq_len() == 1024);
    check_shape(embedding.token_embeddings(), {5, 256});
    check_shape(embedding.positional_embeddings(), {1024, 256});
    CHECK(embedding.token_embeddings().strides() == Tensor::Shape({256, 1}));
    CHECK(embedding.positional_embeddings().strides() == Tensor::Shape({256, 1}));
}

void test_invalid_dimensions() {
    CHECK_THROWS(std::invalid_argument, Embedding(0, 3, 4));
    CHECK_THROWS(std::invalid_argument, Embedding(4, 0, 4));
    CHECK_THROWS(std::invalid_argument, Embedding(4, 3, 0));
}

void test_initialization_is_reproducible_and_small() {
    const Embedding first(8, 4, 6, 123);
    const Embedding second(8, 4, 6, 123);
    const Embedding different(8, 4, 6, 124);

    CHECK(same_values(first.token_embeddings(), second.token_embeddings()));
    CHECK(same_values(first.positional_embeddings(), second.positional_embeddings()));
    CHECK(!same_values(first.token_embeddings(), different.token_embeddings()));
    CHECK(!same_values(first.positional_embeddings(), different.positional_embeddings()));

    for (const float value : first.token_embeddings().data()) {
        CHECK(value >= -0.02F);
        CHECK(value < 0.02F);
    }
    for (const float value : first.positional_embeddings().data()) {
        CHECK(value >= -0.02F);
        CHECK(value < 0.02F);
    }
}

void test_token_lookup_and_repeated_ids() {
    Embedding embedding(4, 3, 5, 7);
    std::iota(embedding.token_embeddings().data().begin(),
              embedding.token_embeddings().data().end(),
              0.0F);
    const std::vector<TokenId> ids = {2, 1, 2};

    const Tensor output = embedding.token_forward(ids, 1, 3);
    check_shape(output, {1, 3, 3});
    CHECK(output.strides() == Tensor::Shape({9, 3, 1}));
    check_vector(output, 0, 0, {6, 7, 8});
    check_vector(output, 0, 1, {3, 4, 5});
    check_vector(output, 0, 2, {6, 7, 8});
}

void test_batch_output_shapes() {
    const Embedding embedding(4, 2, 4, 9);
    const std::vector<TokenId> ids = {0, 1, 2, 3, 3, 2, 1, 0};

    const Tensor token_output = embedding.token_forward(ids, 2, 4);
    const Tensor combined_output = embedding.forward(ids, 2, 4);
    check_shape(token_output, {2, 4, 2});
    check_shape(combined_output, {2, 4, 2});
    CHECK(token_output.strides() == Tensor::Shape({8, 2, 1}));
    CHECK(combined_output.size() == 16);
}

void test_positions_are_added_and_reset_per_batch() {
    Embedding embedding(3, 2, 4, 1);
    set_token_row(embedding, 1, {1, 2});
    set_position_row(embedding, 0, {10, 20});
    set_position_row(embedding, 1, {30, 40});
    set_position_row(embedding, 2, {50, 60});
    set_position_row(embedding, 3, {70, 80});
    const std::vector<TokenId> ids = {1, 1, 1, 1};

    const Tensor tokens = embedding.token_forward(ids, 2, 2);
    check_vector(tokens, 0, 0, {1, 2});
    check_vector(tokens, 0, 1, {1, 2});
    check_vector(tokens, 1, 0, {1, 2});
    check_vector(tokens, 1, 1, {1, 2});

    const Tensor combined = embedding.forward(ids, 2, 2);
    check_vector(combined, 0, 0, {11, 22});
    check_vector(combined, 0, 1, {31, 42});
    check_vector(combined, 1, 0, {11, 22});
    check_vector(combined, 1, 1, {31, 42});
}

void test_batch_size_mismatch() {
    const Embedding embedding(4, 2, 4);
    const std::vector<TokenId> ids = {0, 1, 2};
    const std::vector<TokenId> empty;
    CHECK_THROWS(std::invalid_argument, embedding.token_forward(ids, 2, 2));
    CHECK_THROWS(
        std::invalid_argument,
        embedding.token_forward(empty, std::numeric_limits<std::size_t>::max(), 2));
}

void test_out_of_vocabulary_ids() {
    const Embedding embedding(4, 2, 4);
    const std::vector<TokenId> ids = {0, 4, 1};
    CHECK_THROWS(std::out_of_range, embedding.token_forward(ids, 1, 3));
    CHECK_THROWS(std::out_of_range, embedding.forward(ids, 1, 3));
}

void test_sequence_too_long() {
    const Embedding embedding(4, 2, 2);
    const std::vector<TokenId> ids = {0, 0, 0};
    CHECK_THROWS(std::invalid_argument, embedding.token_forward(ids, 1, 3));
    CHECK_THROWS(std::invalid_argument, embedding.forward(ids, 1, 3));
}

void test_empty_inputs() {
    const Embedding embedding(4, 3, 4);
    const std::vector<TokenId> ids;

    const Tensor empty_sequence = embedding.token_forward(ids, 2, 0);
    check_shape(empty_sequence, {2, 0, 3});
    CHECK(empty_sequence.size() == 0);

    const Tensor empty_batch = embedding.forward(ids, 0, 3);
    check_shape(empty_batch, {0, 3, 3});
    CHECK(empty_batch.size() == 0);
}

void test_persistence_preserves_exact_values() {
    const ScopedFile file(
        std::filesystem::current_path() / "mini_gpt_embedding_persistence_test.bin");
    Embedding embedding(5, 3, 4, 987);
    embedding.token_embeddings().data().front() = 0.123456789F;
    embedding.token_embeddings().data().back() = -0.0F;
    embedding.positional_embeddings().data().front() = -0.75F;
    embedding.positional_embeddings().data().back() = 0.987654321F;
    embedding.save(file.path());

    const Embedding loaded = Embedding::load(file.path());
    CHECK(loaded.vocab_size() == embedding.vocab_size());
    CHECK(loaded.embedding_dim() == embedding.embedding_dim());
    CHECK(loaded.max_seq_len() == embedding.max_seq_len());
    CHECK(same_bits(loaded.token_embeddings(), embedding.token_embeddings()));
    CHECK(same_bits(loaded.positional_embeddings(), embedding.positional_embeddings()));

    const std::vector<TokenId> ids = {4, 0, 2, 4};
    CHECK(same_values(
        loaded.token_forward(ids, 2, 2), embedding.token_forward(ids, 2, 2)));
    CHECK(same_values(loaded.forward(ids, 2, 2), embedding.forward(ids, 2, 2)));
}

void test_load_rejects_missing_and_invalid_files() {
    const ScopedFile file(
        std::filesystem::current_path() / "mini_gpt_embedding_invalid_test.bin");
    CHECK_THROWS(std::runtime_error, Embedding::load(file.path()));

    {
        std::ofstream output(file.path(), std::ios::binary | std::ios::trunc);
        output << "not an embedding";
    }
    CHECK_THROWS(std::runtime_error, Embedding::load(file.path()));
}

void test_load_rejects_invalid_dimensions() {
    const ScopedFile file(
        std::filesystem::current_path() / "mini_gpt_embedding_dimensions_test.bin");
    const Embedding embedding(3, 2, 4);
    embedding.save(file.path());

    {
        std::fstream file_stream(
            file.path(), std::ios::binary | std::ios::in | std::ios::out);
        file_stream.seekp(20);
        const std::uint64_t zero = 0;
        file_stream.write(
            reinterpret_cast<const char*>(&zero),
            static_cast<std::streamsize>(sizeof(zero)));
    }
    CHECK_THROWS(std::runtime_error, Embedding::load(file.path()));
}

void test_load_rejects_truncated_file() {
    const ScopedFile file(
        std::filesystem::current_path() / "mini_gpt_embedding_truncated_test.bin");
    const Embedding embedding(3, 2, 4);
    embedding.save(file.path());
    std::filesystem::resize_file(file.path(), 20);
    CHECK_THROWS(std::runtime_error, Embedding::load(file.path()));
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"default configuration and shapes", test_default_configuration_and_shapes},
        {"invalid dimensions", test_invalid_dimensions},
        {"initialization is reproducible and small", test_initialization_is_reproducible_and_small},
        {"token lookup and repeated IDs", test_token_lookup_and_repeated_ids},
        {"batch output shapes", test_batch_output_shapes},
        {"positions are added and reset per batch", test_positions_are_added_and_reset_per_batch},
        {"batch size mismatch", test_batch_size_mismatch},
        {"out-of-vocabulary IDs", test_out_of_vocabulary_ids},
        {"sequence too long", test_sequence_too_long},
        {"empty inputs", test_empty_inputs},
        {"persistence preserves exact values", test_persistence_preserves_exact_values},
        {"load rejects missing and invalid files", test_load_rejects_missing_and_invalid_files},
        {"load rejects invalid dimensions", test_load_rejects_invalid_dimensions},
        {"load rejects truncated file", test_load_rejects_truncated_file},
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

#include "mini_gpt/data_loader.hpp"
#include "mini_gpt/adamw.hpp"
#include "mini_gpt/cross_entropy.hpp"
#include "mini_gpt/gpt.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using mini_gpt::AdamW;
using mini_gpt::BPETokenizer;
using mini_gpt::GPT;
using mini_gpt::TextDataLoader;

class TestFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void check(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        throw TestFailure(std::string(file) + ':' + std::to_string(line) + ": " + expression);
    }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

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

void test_utf8_split_preserves_text_and_boundaries() {
    const std::string text = "áb😀ñeérguiópqrstuvwx";
    const std::size_t count = mini_gpt::utf8_codepoint_count(text);
    const auto split = mini_gpt::split_corpus_utf8(text, 0.2);
    CHECK(split.training.size() + split.validation.size() == text.size());
    CHECK(split.training + split.validation == text);
    CHECK(split.training != split.validation);
    CHECK(mini_gpt::utf8_codepoint_count(split.training) == count - 4);
    CHECK(mini_gpt::utf8_codepoint_count(split.validation) == 4);

    bool invalid_rejected = false;
    try {
        static_cast<void>(mini_gpt::utf8_codepoint_count(std::string_view("\xF0\x9F", 2)));
    } catch (const std::invalid_argument&) {
        invalid_rejected = true;
    }
    CHECK(invalid_rejected);
}

void test_batches_shift_padding_and_final_partial_batch() {
    const BPETokenizer tokenizer(260);
    TextDataLoader loader(tokenizer, "abcdef", "xyz", 2, 2, 7, false);
    CHECK(loader.training_sample_count() == 3);
    CHECK(loader.validation_sample_count() == 1);
    CHECK(loader.training_dropped_tokens() == 0);

    const auto first = loader.next_batch();
    CHECK(first.has_value());
    CHECK(first->batch_size == 2);
    CHECK(first->seq_len == 2);
    CHECK(first->real_tokens == 4);
    CHECK(first->inputs == std::vector<std::uint32_t>({97, 98, 99, 100}));
    CHECK(first->targets == std::vector<std::uint32_t>({98, 99, 100, 101}));
    CHECK(std::all_of(first->mask.begin(), first->mask.end(), [](std::uint8_t value) {
        return value == 1;
    }));

    const auto second = loader.next_batch();
    CHECK(second.has_value());
    CHECK(second->batch_size == 1);
    CHECK(second->real_tokens == 1);
    CHECK(second->inputs == std::vector<std::uint32_t>({101, 0}));
    CHECK(second->targets == std::vector<std::uint32_t>({102, 0}));
    CHECK(second->mask == std::vector<std::uint8_t>({1, 0}));
    CHECK(!loader.next_batch().has_value());

    const auto validation = loader.next_validation_batch();
    CHECK(validation.has_value());
    CHECK(validation->inputs == std::vector<std::uint32_t>({120, 121}));
    CHECK(validation->targets == std::vector<std::uint32_t>({121, 122}));
}

void test_train_validation_do_not_overlap() {
    const BPETokenizer tokenizer(260);
    TextDataLoader loader(tokenizer, "aaaa", "bbbb", 3, 1, 11);
    const auto training = loader.next_batch();
    const auto validation = loader.next_validation_batch();
    CHECK(training.has_value() && validation.has_value());
    CHECK(std::all_of(training->targets.begin(), training->targets.end(), [](std::uint32_t id) {
        return id == 97 || id == 0;
    }));
    CHECK(std::all_of(validation->targets.begin(), validation->targets.end(), [](std::uint32_t id) {
        return id == 98 || id == 0;
    }));
}

void test_shuffling_is_reproducible_between_epochs() {
    const BPETokenizer tokenizer(260);
    TextDataLoader first(tokenizer, "abcdefghijkl", "", 2, 1, 123);
    TextDataLoader second(tokenizer, "abcdefghijkl", "", 2, 1, 123);
    const auto first_batch = first.next_batch();
    const auto second_batch = second.next_batch();
    CHECK(first_batch->inputs == second_batch->inputs);

    first.reset_epoch();
    second.reset_epoch();
    const auto next_first = first.next_batch();
    const auto next_second = second.next_batch();
    CHECK(next_first->inputs == next_second->inputs);
    CHECK(next_first->inputs != first_batch->inputs);
}

std::vector<float> parameters(GPT& model) {
    std::vector<float> values;
    for (const auto& binding : mini_gpt::gpt_parameters(model)) {
        values.insert(values.end(), binding.parameter->data().begin(), binding.parameter->data().end());
    }
    return values;
}

void test_training_checkpoint_resume() {
    const ScopedFile model_file(std::filesystem::current_path() / "mini_gpt_resume_model.bin");
    const ScopedFile optimizer_file(std::filesystem::current_path() / "mini_gpt_resume_adamw.bin");
    const std::vector<std::uint32_t> inputs = {1, 2, 3, 4};
    const std::vector<std::uint32_t> targets = {2, 3, 4, 5};
    GPT original(8, 8, 2, 1, 16, 5, 123);
    mini_gpt::AdamWConfig config;
    config.learning_rate = 0.001F;
    AdamW original_optimizer(original, config);
    for (int i = 0; i < 2; ++i) {
        original.zero_grad();
        original.backward(inputs, 1, 4, targets);
        original_optimizer.step();
    }
    original.save(model_file.path());
    original_optimizer.save(optimizer_file.path());

    GPT restored = GPT::load(model_file.path());
    AdamW restored_optimizer = AdamW::load(optimizer_file.path(), restored);
    original.zero_grad();
    original.backward(inputs, 1, 4, targets);
    original_optimizer.step();
    restored.zero_grad();
    restored.backward(inputs, 1, 4, targets);
    restored_optimizer.step();
    CHECK(parameters(original) == parameters(restored));
    CHECK(restored_optimizer.step_count() == 3);
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"UTF-8 split preserves text and boundaries", test_utf8_split_preserves_text_and_boundaries},
        {"batches shift padding and final partial batch", test_batches_shift_padding_and_final_partial_batch},
        {"train validation do not overlap", test_train_validation_do_not_overlap},
        {"shuffling is reproducible between epochs", test_shuffling_is_reproducible_between_epochs},
        {"training checkpoint resume", test_training_checkpoint_resume},
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
    std::cout << "\n" << (tests->name ? (sizeof(tests) / sizeof(tests[0]) - failures) : 0)
              << '/' << sizeof(tests) / sizeof(tests[0]) << " tests passed\n";
    return failures == 0 ? 0 : 1;
}

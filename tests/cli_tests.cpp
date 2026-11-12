#include "mini_gpt/chat_mode.hpp"
#include "mini_gpt/cli.hpp"
#include "mini_gpt/data_loader.hpp"
#include "mini_gpt/training_state.hpp"

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

using mini_gpt::BPETokenizer;
using mini_gpt::TextDataLoader;
using mini_gpt::TrainingState;

class TestFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void check(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        throw TestFailure(std::string(file) + ':' + std::to_string(line) + ": " + expression);
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
#define CHECK_THROWS(exception_type, ...) \
    check_throws<exception_type>([&]() { static_cast<void>((__VA_ARGS__)); })

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

void test_argument_parsing_and_validation() {
    const mini_gpt::cli::Arguments args{"program", "--steps", "5", "--lr", "0.001", "--force"};
    mini_gpt::cli::validate(args, {"--force"}, {"--steps", "--lr"});
    CHECK(mini_gpt::cli::has(args, "--force"));
    CHECK(mini_gpt::cli::size_value(args, "--steps", 20) == 5);
    CHECK(mini_gpt::cli::float_value(args, "--lr", 0.0003F) > 0.0009F);
    CHECK(mini_gpt::cli::float_value(args, "--epsilon", 1.0e-8F) > 0.0F);
    CHECK(mini_gpt::cli::size_value(args, "--batch-size", 2) == 2);

    CHECK_THROWS(std::invalid_argument, mini_gpt::cli::size_value(args, "--steps", 20, 6));
    const mini_gpt::cli::Arguments duplicate{"program", "--steps", "1", "--steps", "2"};
    CHECK_THROWS(std::invalid_argument, mini_gpt::cli::size_value(duplicate, "--steps", 20));
    const mini_gpt::cli::Arguments unknown{"program", "--unknown"};
    CHECK_THROWS(std::invalid_argument, mini_gpt::cli::validate(unknown, {}, {}));
}

void test_output_overwrite_protection() {
    const ScopedFile file(std::filesystem::current_path() / "mini_gpt_cli_output_test.tmp");
    {
        std::ofstream output(file.path());
        output << "existing";
    }
    CHECK_THROWS(std::runtime_error, mini_gpt::cli::require_output_file(file.path(), false));
    mini_gpt::cli::require_output_file(file.path(), true);

    const std::filesystem::path missing =
        std::filesystem::current_path() / "missing_directory" / "file.bin";
    CHECK_THROWS(std::runtime_error, mini_gpt::cli::require_output_file(missing, false));
}

void test_data_loader_state_restores_next_batch() {
    const BPETokenizer tokenizer(260);
    TextDataLoader original(tokenizer, "abcdefghij", "", 2, 1, 123);
    static_cast<void>(original.next_batch());
    const auto state = original.state();
    const auto expected = original.next_batch();

    TextDataLoader restored(tokenizer, "abcdefghij", "", 2, 1, 999);
    restored.restore_state(state);
    const auto actual = restored.next_batch();
    CHECK(actual.has_value() && expected.has_value());
    CHECK(actual->inputs == expected->inputs);
    CHECK(actual->targets == expected->targets);
    CHECK(actual->mask == expected->mask);
}

void test_training_state_roundtrip() {
    const ScopedFile file(std::filesystem::current_path() / "mini_gpt_training_state_test.data");
    const BPETokenizer tokenizer(260);
    TextDataLoader loader(tokenizer, "abcdefghij", "", 2, 1, 77);
    static_cast<void>(loader.next_batch());
    TrainingState expected{"corpus.txt", "tokenizer.bpe", 3, 1234, loader.state()};
    mini_gpt::save_training_state(file.path(), expected);

    const TrainingState restored = mini_gpt::load_training_state(file.path());
    CHECK(restored.corpus_path == expected.corpus_path);
    CHECK(restored.tokenizer_path == expected.tokenizer_path);
    CHECK(restored.epoch == 3);
    CHECK(restored.processed_tokens == 1234);
    CHECK(restored.data.training_order == expected.data.training_order);
    CHECK(restored.data.validation_order.empty());
    CHECK(restored.data.training_cursor == expected.data.training_cursor);
    CHECK(restored.data.generator_state == expected.data.generator_state);

    TextDataLoader restored_loader(tokenizer, "abcdefghij", "", 2, 1, 1);
    restored_loader.restore_state(restored.data);
    const auto actual = restored_loader.next_batch();
    const auto wanted = loader.next_batch();
    CHECK(actual->inputs == wanted->inputs);
}

void test_completion_and_chat_prompts() {
    CHECK(mini_gpt::parse_chat_mode("chat") == mini_gpt::ChatMode::chat);
    CHECK(mini_gpt::parse_chat_mode("completion") == mini_gpt::ChatMode::completion);
    CHECK(mini_gpt::parse_chat_mode("continue") == mini_gpt::ChatMode::completion);

    const std::string poem = "Mientras por competir con tu cabello";
    CHECK(mini_gpt::build_chat_prompt(mini_gpt::ChatMode::completion, poem) == poem);
    const std::string utf8 = "  línea uno\nñ y 😀\t";
    CHECK(mini_gpt::build_chat_prompt(mini_gpt::ChatMode::completion, utf8) == utf8);
    CHECK(mini_gpt::build_chat_prompt(mini_gpt::ChatMode::completion, utf8).find("<|user|>") ==
          std::string::npos);
    CHECK(mini_gpt::build_chat_prompt(mini_gpt::ChatMode::completion, utf8).find("<|assistant|>") ==
          std::string::npos);
    CHECK(mini_gpt::build_chat_prompt(mini_gpt::ChatMode::completion, utf8).find("<|eot|>") ==
          std::string::npos);

    CHECK(mini_gpt::build_chat_prompt(mini_gpt::ChatMode::chat, "hola") ==
          "<|user|> hola <|eot|>\n<|assistant|>");
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"completion and chat prompts", test_completion_and_chat_prompts},
        {"argument parsing and validation", test_argument_parsing_and_validation},
        {"output overwrite protection", test_output_overwrite_protection},
        {"DataLoader state restores next batch", test_data_loader_state_restores_next_batch},
        {"training state roundtrip", test_training_state_roundtrip},
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

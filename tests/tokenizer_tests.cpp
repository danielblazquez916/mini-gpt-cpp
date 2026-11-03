#include "mini_gpt/bpe_tokenizer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using mini_gpt::BPETokenizer;
using Merge = BPETokenizer::Merge;
using TokenId = BPETokenizer::TokenId;

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

void check_ids(
    const std::vector<TokenId>& actual,
    std::initializer_list<TokenId> expected) {
    CHECK(actual.size() == expected.size());
    auto actual_it = actual.begin();
    auto expected_it = expected.begin();
    for (; actual_it != actual.end() && expected_it != expected.end();
         ++actual_it, ++expected_it) {
        CHECK(*actual_it == *expected_it);
    }
}

void check_merge(const Merge& merge, TokenId left, TokenId right) {
    CHECK(merge.left == left);
    CHECK(merge.right == right);
}

std::string corpus() {
    const std::filesystem::path path =
        std::filesystem::path(MINI_GPT_TEST_DATA_DIR) / "conversations.txt";
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw TestFailure("Cannot open test corpus");
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

BPETokenizer trained_tokenizer(std::size_t max_vocab_size = 300) {
    BPETokenizer tokenizer(max_vocab_size);
    tokenizer.train(corpus());
    return tokenizer;
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

void test_base_vocabulary_and_special_ids() {
    const BPETokenizer tokenizer;
    CHECK(tokenizer.vocab_size() == 260);
    CHECK(tokenizer.max_vocab_size() == 4000);
    CHECK(tokenizer.merges().empty());
    CHECK(tokenizer.token(0).size() == 1);
    CHECK(static_cast<unsigned char>(tokenizer.token(0)[0]) == 0);
    CHECK(tokenizer.token(65) == "A");
    CHECK(static_cast<unsigned char>(tokenizer.token(255)[0]) == 255);
    CHECK(tokenizer.token(BPETokenizer::user_id) == "<|user|>");
    CHECK(tokenizer.token(BPETokenizer::assistant_id) == "<|assistant|>");
    CHECK(tokenizer.token(BPETokenizer::eot_id) == "<|eot|>");
    CHECK(tokenizer.token(BPETokenizer::end_id) == "<|end|>");
}

void test_vocabulary_size_validation() {
    CHECK_THROWS(std::invalid_argument, BPETokenizer(259));
    const BPETokenizer tokenizer(260);
    CHECK(tokenizer.vocab_size() == 260);
}

void test_empty_input() {
    BPETokenizer tokenizer;
    tokenizer.train("");
    CHECK(tokenizer.vocab_size() == 260);
    const std::vector<TokenId> ids = tokenizer.encode("");
    CHECK(ids.empty());
    CHECK(tokenizer.decode(ids).empty());
}

void test_untrained_encoding_uses_bytes() {
    const BPETokenizer tokenizer(260);
    const std::string text = "Añ😀";
    const auto ids = tokenizer.encode(text);
    check_ids(ids, {65, 0xC3, 0xB1, 0xF0, 0x9F, 0x98, 0x80});
    CHECK(tokenizer.decode(ids) == text);
}

void test_training_learns_merges_and_roundtrips_corpus() {
    const BPETokenizer tokenizer = trained_tokenizer();
    const std::string text = corpus();
    const auto ids = tokenizer.encode(text);

    CHECK(tokenizer.vocab_size() == 300);
    CHECK(tokenizer.merges().size() == 40);
    CHECK(ids.size() < text.size());
    CHECK(tokenizer.decode(ids) == text);
    CHECK(std::count(ids.begin(), ids.end(), BPETokenizer::user_id) == 2);
    CHECK(std::count(ids.begin(), ids.end(), BPETokenizer::assistant_id) == 2);
    CHECK(std::count(ids.begin(), ids.end(), BPETokenizer::eot_id) == 4);
    CHECK(std::count(ids.begin(), ids.end(), BPETokenizer::end_id) == 1);

    const auto& vocabulary = tokenizer.vocabulary();
    for (std::size_t i = 0; i < tokenizer.merges().size(); ++i) {
        const Merge& merge = tokenizer.merges()[i];
        const std::size_t result_id = BPETokenizer::base_vocabulary_size + i;
        CHECK(vocabulary[result_id] ==
              vocabulary[merge.left] + vocabulary[merge.right]);
    }
}

void test_vocabulary_limit_is_respected() {
    const BPETokenizer tokenizer = trained_tokenizer(261);
    CHECK(tokenizer.vocab_size() == 261);
    CHECK(tokenizer.merges().size() == 1);
}

void test_training_is_deterministic() {
    const BPETokenizer first = trained_tokenizer();
    const BPETokenizer second = trained_tokenizer();
    CHECK(first.vocabulary() == second.vocabulary());
    CHECK(first.merges() == second.merges());
    CHECK(first.encode(corpus()) == second.encode(corpus()));
}

void test_tie_break_and_merge_order() {
    BPETokenizer tokenizer(263);
    tokenizer.train("abba");

    CHECK(tokenizer.merges().size() == 3);
    check_merge(tokenizer.merges()[0], 97, 98);
    check_merge(tokenizer.merges()[1], 98, 97);
    check_merge(tokenizer.merges()[2], 260, 261);
    CHECK(tokenizer.token(262) == "abba");
    check_ids(tokenizer.encode("abba"), {262});
    CHECK(tokenizer.decode(tokenizer.encode("abba")) == "abba");
}

void test_overlapping_pairs() {
    BPETokenizer tokenizer(262);
    tokenizer.train("aaaa");
    check_merge(tokenizer.merges()[0], 97, 97);
    check_merge(tokenizer.merges()[1], 260, 260);

    const auto ids = tokenizer.encode("aaaaa");
    check_ids(ids, {261, 97});
    CHECK(tokenizer.decode(ids) == "aaaaa");
}

void test_special_tokens_are_atomic() {
    const BPETokenizer tokenizer = trained_tokenizer();
    const std::string specials = "<|user|><|assistant|><|eot|><|end|>";
    const auto special_ids = tokenizer.encode(specials);
    check_ids(
        special_ids,
        {BPETokenizer::user_id,
         BPETokenizer::assistant_id,
         BPETokenizer::eot_id,
         BPETokenizer::end_id});
    CHECK(tokenizer.decode(special_ids) == specials);

    const auto surrounding_ids = tokenizer.encode("x<|user|>y");
    check_ids(surrounding_ids, {120, BPETokenizer::user_id, 121});
    CHECK(tokenizer.decode(surrounding_ids) == "x<|user|>y");

    for (const Merge& merge : tokenizer.merges()) {
        CHECK(merge.left < BPETokenizer::user_id || merge.left > BPETokenizer::end_id);
        CHECK(merge.right < BPETokenizer::user_id || merge.right > BPETokenizer::end_id);
    }
}

void test_specials_do_not_create_merges() {
    BPETokenizer tokenizer;
    tokenizer.train("a<|user|>b<|eot|>c");
    CHECK(tokenizer.vocab_size() == 260);
    CHECK(tokenizer.merges().empty());
    check_ids(
        tokenizer.encode("a<|user|>b<|eot|>c"),
        {97, BPETokenizer::user_id, 98, BPETokenizer::eot_id, 99});
}

void test_utf8_roundtrip() {
    const BPETokenizer tokenizer = trained_tokenizer();
    const std::string text =
        "  ¡Hola, niño! ¿Cómo estás? Corazón y pingüino 😀🚀\n\tTexto nuevo.";
    const auto ids = tokenizer.encode(text);
    CHECK(!ids.empty());
    CHECK(tokenizer.decode(ids) == text);
}

void test_arbitrary_bytes_roundtrip() {
    const BPETokenizer tokenizer;
    const std::string bytes("a\0b\xFF", 4);
    const auto ids = tokenizer.encode(bytes);
    check_ids(ids, {97, 0, 98, 255});
    CHECK(tokenizer.decode(ids) == bytes);
}

void test_retraining_resets_learned_state() {
    BPETokenizer tokenizer = trained_tokenizer();
    const auto expected_vocabulary = tokenizer.vocabulary();
    const auto expected_merges = tokenizer.merges();
    const auto expected_ids = tokenizer.encode(corpus());

    tokenizer.train("");
    CHECK(tokenizer.vocab_size() == 260);
    CHECK(tokenizer.merges().empty());

    tokenizer.train(corpus());
    CHECK(tokenizer.vocabulary() == expected_vocabulary);
    CHECK(tokenizer.merges() == expected_merges);
    CHECK(tokenizer.encode(corpus()) == expected_ids);
}

void test_persistence_roundtrip() {
    const ScopedFile file(
        std::filesystem::current_path() / "mini_gpt_tokenizer_persistence_test.bpe");
    const BPETokenizer tokenizer = trained_tokenizer();
    const std::string sample = "<|user|> ¿qué tal, niño? 😀 <|eot|>";
    const auto original_ids = tokenizer.encode(sample);
    tokenizer.save(file.path());

    const BPETokenizer loaded = BPETokenizer::load(file.path());
    CHECK(loaded.max_vocab_size() == tokenizer.max_vocab_size());
    CHECK(loaded.vocabulary() == tokenizer.vocabulary());
    CHECK(loaded.merges() == tokenizer.merges());
    CHECK(loaded.encode(sample) == original_ids);
    CHECK(loaded.decode(original_ids) == sample);
    CHECK(loaded.token(BPETokenizer::user_id) == "<|user|>");
}

void test_load_rejects_missing_and_invalid_files() {
    const ScopedFile file(
        std::filesystem::current_path() / "mini_gpt_tokenizer_invalid_test.bpe");
    CHECK_THROWS(std::runtime_error, BPETokenizer::load(file.path()));

    {
        std::ofstream output(file.path(), std::ios::binary | std::ios::trunc);
        output << "not a tokenizer";
    }
    CHECK_THROWS(std::runtime_error, BPETokenizer::load(file.path()));
}

void test_invalid_token_ids_are_rejected() {
    const BPETokenizer tokenizer(260);
    const std::vector<TokenId> invalid_ids = {260};
    CHECK_THROWS(std::out_of_range, tokenizer.decode(invalid_ids));
    CHECK_THROWS(std::out_of_range, tokenizer.token(260));
}

struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"base vocabulary and special IDs", test_base_vocabulary_and_special_ids},
        {"vocabulary size validation", test_vocabulary_size_validation},
        {"empty input", test_empty_input},
        {"untrained encoding uses bytes", test_untrained_encoding_uses_bytes},
        {"training learns merges and roundtrips corpus", test_training_learns_merges_and_roundtrips_corpus},
        {"vocabulary limit is respected", test_vocabulary_limit_is_respected},
        {"training is deterministic", test_training_is_deterministic},
        {"tie break and merge order", test_tie_break_and_merge_order},
        {"overlapping pairs", test_overlapping_pairs},
        {"special tokens are atomic", test_special_tokens_are_atomic},
        {"specials do not create merges", test_specials_do_not_create_merges},
        {"UTF-8 roundtrip", test_utf8_roundtrip},
        {"arbitrary bytes roundtrip", test_arbitrary_bytes_roundtrip},
        {"retraining resets learned state", test_retraining_resets_learned_state},
        {"persistence roundtrip", test_persistence_roundtrip},
        {"load rejects missing and invalid files", test_load_rejects_missing_and_invalid_files},
        {"invalid token IDs are rejected", test_invalid_token_ids_are_rejected},
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

#ifndef MINI_GPT_DATA_LOADER_HPP
#define MINI_GPT_DATA_LOADER_HPP

#include "mini_gpt/bpe_tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace mini_gpt {

struct CorpusSplit {
    std::string training;
    std::string validation;
};

struct DataLoaderState {
    std::vector<std::size_t> training_order;
    std::vector<std::size_t> validation_order;
    std::size_t training_cursor = 0;
    std::size_t validation_cursor = 0;
    std::string generator_state;
};

struct SequenceBatch {
    std::vector<std::uint32_t> inputs;
    std::vector<std::uint32_t> targets;
    std::vector<std::uint8_t> mask;
    std::size_t batch_size = 0;
    std::size_t seq_len = 0;
    std::size_t real_tokens = 0;
};

[[nodiscard]] std::size_t utf8_codepoint_count(std::string_view text);
[[nodiscard]] CorpusSplit split_corpus_utf8(std::string_view text, double validation_ratio = 0.1);
[[nodiscard]] std::string read_utf8_file(const std::filesystem::path& path);

class TextDataLoader {
public:
    TextDataLoader(
        const BPETokenizer& tokenizer,
        std::string_view training_text,
        std::string_view validation_text,
        std::size_t seq_len = 128,
        std::size_t batch_size = 2,
        std::uint32_t seed = 42,
        bool shuffle_training = true);

    void reset_epoch();
    void reset_validation();
    [[nodiscard]] std::optional<SequenceBatch> next_batch();
    [[nodiscard]] std::optional<SequenceBatch> next_validation_batch();

    [[nodiscard]] DataLoaderState state() const;
    void restore_state(const DataLoaderState& state);

    [[nodiscard]] std::size_t training_sample_count() const noexcept;
    [[nodiscard]] std::size_t validation_sample_count() const noexcept;
    [[nodiscard]] std::size_t training_token_count() const noexcept;
    [[nodiscard]] std::size_t validation_token_count() const noexcept;
    [[nodiscard]] std::size_t training_dropped_tokens() const noexcept;
    [[nodiscard]] std::size_t validation_dropped_tokens() const noexcept;
    [[nodiscard]] std::size_t seq_len() const noexcept;
    [[nodiscard]] std::size_t batch_size() const noexcept;

private:
    struct Sequence {
        std::size_t offset;
        std::size_t length;
    };

    [[nodiscard]] std::vector<Sequence> make_sequences(
        const std::vector<std::uint32_t>& tokens,
        std::size_t& dropped_tokens) const;
    [[nodiscard]] std::optional<SequenceBatch> make_batch(
        const std::vector<Sequence>& sequences,
        const std::vector<std::uint32_t>& tokens,
        const std::vector<std::size_t>& order,
        std::size_t& cursor) const;

    std::size_t seq_len_;
    std::size_t batch_size_;
    std::vector<std::uint32_t> training_tokens_;
    std::vector<std::uint32_t> validation_tokens_;
    std::vector<Sequence> training_sequences_;
    std::vector<Sequence> validation_sequences_;
    std::vector<std::size_t> training_order_;
    std::vector<std::size_t> validation_order_;
    std::size_t training_cursor_ = 0;
    std::size_t validation_cursor_ = 0;
    std::size_t training_dropped_tokens_ = 0;
    std::size_t validation_dropped_tokens_ = 0;
    bool shuffle_training_ = true;
    std::mt19937 generator_;
};

}  // namespace mini_gpt

#endif

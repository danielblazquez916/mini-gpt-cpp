#include "mini_gpt/data_loader.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace mini_gpt {
namespace {

std::size_t continuation_count(unsigned char lead) {
    if (lead <= 0x7F) return 0;
    if (lead >= 0xC2 && lead <= 0xDF) return 1;
    if (lead >= 0xE0 && lead <= 0xEF) return 2;
    if (lead >= 0xF0 && lead <= 0xF4) return 3;
    throw std::invalid_argument("Invalid UTF-8 leading byte");
}

}  // namespace

std::size_t utf8_codepoint_count(std::string_view text) {
    std::size_t count = 0;
    std::size_t index = 0;
    while (index < text.size()) {
        const auto lead = static_cast<unsigned char>(text[index]);
        const std::size_t continuations = continuation_count(lead);
        if (index + continuations >= text.size()) {
            throw std::invalid_argument("Incomplete UTF-8 sequence");
        }

        std::uint32_t code_point = lead;
        if (continuations == 1) code_point = lead & 0x1FU;
        if (continuations == 2) code_point = lead & 0x0FU;
        if (continuations == 3) code_point = lead & 0x07U;
        for (std::size_t i = 1; i <= continuations; ++i) {
            const auto byte = static_cast<unsigned char>(text[index + i]);
            if ((byte & 0xC0U) != 0x80U) {
                throw std::invalid_argument("Invalid UTF-8 continuation byte");
            }
            code_point = (code_point << 6) | (byte & 0x3FU);
        }

        const std::uint32_t minimum = continuations == 0 ? 0x00
            : continuations == 1 ? 0x80
            : continuations == 2 ? 0x800
                                 : 0x10000;
        if (code_point < minimum || code_point > 0x10FFFF ||
            (code_point >= 0xD800 && code_point <= 0xDFFF)) {
            throw std::invalid_argument("Invalid UTF-8 code point");
        }
        index += continuations + 1;
        ++count;
    }
    return count;
}

CorpusSplit split_corpus_utf8(std::string_view text, double validation_ratio) {
    if (!std::isfinite(validation_ratio) || validation_ratio <= 0.0 || validation_ratio >= 1.0) {
        throw std::invalid_argument("Validation ratio must be between zero and one");
    }
    const std::size_t count = utf8_codepoint_count(text);
    const std::size_t training_count = static_cast<std::size_t>(
        static_cast<double>(count) * (1.0 - validation_ratio));
    if (training_count == 0 || training_count == count) {
        throw std::invalid_argument("Both corpus splits must be non-empty");
    }

    std::size_t byte_offset = 0;
    for (std::size_t i = 0; i < training_count; ++i) {
        const std::size_t continuations = continuation_count(
            static_cast<unsigned char>(text[byte_offset]));
        byte_offset += continuations + 1;
    }
    return CorpusSplit{
        std::string(text.substr(0, byte_offset)),
        std::string(text.substr(byte_offset))};
}

std::string read_utf8_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open corpus: " + path.string());
    }
    const std::string text{
        std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    static_cast<void>(utf8_codepoint_count(text));
    return text;
}

TextDataLoader::TextDataLoader(
    const BPETokenizer& tokenizer,
    std::string_view training_text,
    std::string_view validation_text,
    std::size_t seq_len,
    std::size_t batch_size,
    std::uint32_t seed,
    bool shuffle_training)
    : seq_len_(seq_len),
      batch_size_(batch_size),
      training_tokens_(tokenizer.encode(training_text)),
      validation_tokens_(tokenizer.encode(validation_text)),
      shuffle_training_(shuffle_training),
      generator_(seed) {
    if (seq_len_ == 0 || batch_size_ == 0) {
        throw std::invalid_argument("Sequence and batch sizes must be greater than zero");
    }
    training_sequences_ = make_sequences(training_tokens_, training_dropped_tokens_);
    validation_sequences_ = make_sequences(validation_tokens_, validation_dropped_tokens_);
    training_order_.resize(training_sequences_.size());
    validation_order_.resize(validation_sequences_.size());
    std::iota(training_order_.begin(), training_order_.end(), 0);
    std::iota(validation_order_.begin(), validation_order_.end(), 0);
    reset_epoch();
    reset_validation();
}

void TextDataLoader::reset_epoch() {
    training_cursor_ = 0;
    if (shuffle_training_) {
        std::shuffle(training_order_.begin(), training_order_.end(), generator_);
    }
}

void TextDataLoader::reset_validation() {
    validation_cursor_ = 0;
}

std::optional<SequenceBatch> TextDataLoader::next_batch() {
    return make_batch(training_sequences_, training_tokens_, training_order_, training_cursor_);
}

std::optional<SequenceBatch> TextDataLoader::next_validation_batch() {
    return make_batch(
        validation_sequences_, validation_tokens_, validation_order_, validation_cursor_);
}

DataLoaderState TextDataLoader::state() const {
    std::ostringstream generator;
    generator << generator_;
    return DataLoaderState{
        training_order_,
        validation_order_,
        training_cursor_,
        validation_cursor_,
        generator.str()};
}

void TextDataLoader::restore_state(const DataLoaderState& state) {
    const auto valid_order = [](const std::vector<std::size_t>& order, std::size_t expected) {
        if (order.size() != expected) {
            return false;
        }
        std::vector<bool> seen(expected, false);
        for (const std::size_t index : order) {
            if (index >= expected || seen[index]) {
                return false;
            }
            seen[index] = true;
        }
        return true;
    };
    if (!valid_order(state.training_order, training_sequences_.size()) ||
        !valid_order(state.validation_order, validation_sequences_.size()) ||
        state.training_cursor > training_order_.size() ||
        state.validation_cursor > validation_order_.size()) {
        throw std::invalid_argument("DataLoader state is incompatible with the current dataset");
    }

    std::mt19937 generator;
    std::istringstream input(state.generator_state);
    input >> generator;
    if (input.fail()) {
        throw std::invalid_argument("Invalid DataLoader generator state");
    }
    training_order_ = state.training_order;
    validation_order_ = state.validation_order;
    training_cursor_ = state.training_cursor;
    validation_cursor_ = state.validation_cursor;
    generator_ = generator;
}

std::size_t TextDataLoader::training_sample_count() const noexcept {
    return training_sequences_.size();
}

std::size_t TextDataLoader::validation_sample_count() const noexcept {
    return validation_sequences_.size();
}

std::size_t TextDataLoader::training_token_count() const noexcept {
    return training_tokens_.size();
}

std::size_t TextDataLoader::validation_token_count() const noexcept {
    return validation_tokens_.size();
}

std::size_t TextDataLoader::training_dropped_tokens() const noexcept {
    return training_dropped_tokens_;
}

std::size_t TextDataLoader::validation_dropped_tokens() const noexcept {
    return validation_dropped_tokens_;
}

std::size_t TextDataLoader::seq_len() const noexcept {
    return seq_len_;
}

std::size_t TextDataLoader::batch_size() const noexcept {
    return batch_size_;
}

std::vector<TextDataLoader::Sequence> TextDataLoader::make_sequences(
    const std::vector<std::uint32_t>& tokens,
    std::size_t& dropped_tokens) const {
    std::vector<Sequence> sequences;
    dropped_tokens = 0;
    std::size_t offset = 0;
    while (offset + 1 < tokens.size()) {
        const std::size_t length = std::min(seq_len_, tokens.size() - 1 - offset);
        sequences.push_back(Sequence{offset, length});
        offset += length;
    }
    if (sequences.empty() && tokens.size() == 1) {
        dropped_tokens = 1;
    }
    return sequences;
}

std::optional<SequenceBatch> TextDataLoader::make_batch(
    const std::vector<Sequence>& sequences,
    const std::vector<std::uint32_t>& tokens,
    const std::vector<std::size_t>& order,
    std::size_t& cursor) const {
    if (cursor >= order.size()) {
        return std::nullopt;
    }

    SequenceBatch batch;
    batch.batch_size = std::min(batch_size_, order.size() - cursor);
    batch.seq_len = seq_len_;
    batch.inputs.assign(batch.batch_size * seq_len_, 0);
    batch.targets.assign(batch.batch_size * seq_len_, 0);
    batch.mask.assign(batch.batch_size * seq_len_, 0);

    for (std::size_t sample = 0; sample < batch.batch_size; ++sample) {
        const Sequence& sequence = sequences[order[cursor + sample]];
        for (std::size_t position = 0; position < sequence.length; ++position) {
            const std::size_t destination = sample * seq_len_ + position;
            batch.inputs[destination] = tokens[sequence.offset + position];
            batch.targets[destination] = tokens[sequence.offset + position + 1];
            batch.mask[destination] = 1;
        }
        batch.real_tokens += sequence.length;
    }
    cursor += batch.batch_size;
    return batch;
}

}  // namespace mini_gpt

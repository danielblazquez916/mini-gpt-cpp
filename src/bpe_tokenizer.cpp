#include "mini_gpt/bpe_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace mini_gpt {
namespace {

constexpr std::array<std::string_view, 4> special_tokens = {
    "<|user|>",
    "<|assistant|>",
    "<|eot|>",
    "<|end|>",
};
constexpr std::string_view file_magic = "MINIGPTB";
constexpr std::uint32_t file_version = 1;

using TokenId = BPETokenizer::TokenId;
using Merge = BPETokenizer::Merge;

std::vector<std::string> make_base_vocabulary() {
    std::vector<std::string> vocabulary(256);
    for (std::size_t i = 0; i < 256; ++i) {
        vocabulary[i] = std::string(1, static_cast<char>(i));
    }
    for (const std::string_view special : special_tokens) {
        vocabulary.emplace_back(special);
    }
    return vocabulary;
}

bool is_special(TokenId id) {
    return id >= BPETokenizer::user_id && id <= BPETokenizer::end_id;
}

template <typename NormalCallback, typename SpecialCallback>
void split_specials(
    std::string_view text,
    NormalCallback&& normal_callback,
    SpecialCallback&& special_callback) {
    std::size_t cursor = 0;
    while (cursor < text.size()) {
        std::size_t next_position = std::string_view::npos;
        std::size_t next_special = 0;
        for (std::size_t i = 0; i < special_tokens.size(); ++i) {
            const std::size_t position = text.find(special_tokens[i], cursor);
            if (position < next_position) {
                next_position = position;
                next_special = i;
            }
        }

        if (next_position == std::string_view::npos) {
            normal_callback(text.substr(cursor));
            return;
        }
        if (next_position > cursor) {
            normal_callback(text.substr(cursor, next_position - cursor));
        }

        special_callback(static_cast<TokenId>(BPETokenizer::user_id + next_special));
        cursor = next_position + special_tokens[next_special].size();
    }
}

std::vector<TokenId> byte_sequence(std::string_view text) {
    std::vector<TokenId> sequence;
    sequence.reserve(text.size());
    for (const unsigned char byte : text) {
        sequence.push_back(static_cast<TokenId>(byte));
    }
    return sequence;
}

void apply_merge(std::vector<TokenId>& sequence, const Merge& merge, TokenId result_id) {
    std::size_t read = 0;
    std::size_t write = 0;
    while (read < sequence.size()) {
        if (read + 1 < sequence.size() && sequence[read] == merge.left &&
            sequence[read + 1] == merge.right) {
            sequence[write++] = result_id;
            read += 2;
        } else {
            sequence[write++] = sequence[read++];
        }
    }
    sequence.resize(write);
}

void write_u32(std::ofstream& file, std::uint32_t value) {
    std::array<char, 4> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((value >> (i * 8)) & 0xFFU);
    }
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to write tokenizer file");
    }
}

void write_u64(std::ofstream& file, std::uint64_t value) {
    write_u32(file, static_cast<std::uint32_t>(value & 0xFFFFFFFFU));
    write_u32(file, static_cast<std::uint32_t>(value >> 32));
}

void write_string(std::ofstream& file, const std::string& value) {
    if (value.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::length_error("Token is too large to save");
    }
    write_u64(file, static_cast<std::uint64_t>(value.size()));
    if (!value.empty()) {
        file.write(value.data(), static_cast<std::streamsize>(value.size()));
        if (!file) {
            throw std::runtime_error("Failed to write tokenizer file");
        }
    }
}

std::uint32_t read_u32(std::ifstream& file, std::uint64_t& remaining) {
    if (remaining < 4) {
        throw std::runtime_error("Unexpected end of tokenizer file");
    }

    std::array<unsigned char, 4> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to read tokenizer file");
    }
    remaining -= 4;

    std::uint32_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<std::uint32_t>(bytes[i]) << (i * 8);
    }
    return value;
}

std::uint64_t read_u64(std::ifstream& file, std::uint64_t& remaining) {
    const std::uint64_t low = read_u32(file, remaining);
    const std::uint64_t high = read_u32(file, remaining);
    return low | (high << 32);
}

std::string read_string(std::ifstream& file, std::uint64_t& remaining) {
    const std::uint64_t length = read_u64(file, remaining);
    if (length > remaining ||
        length > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        length > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Invalid token length in tokenizer file");
    }

    std::string value(static_cast<std::size_t>(length), '\0');
    if (length != 0) {
        file.read(value.data(), static_cast<std::streamsize>(length));
        if (!file) {
            throw std::runtime_error("Failed to read token from tokenizer file");
        }
        remaining -= length;
    }
    return value;
}

}  // namespace

BPETokenizer::BPETokenizer(std::size_t max_vocab_size)
    : max_vocab_size_(validate_max_vocab_size(max_vocab_size)),
      vocabulary_(make_base_vocabulary()) {}

void BPETokenizer::train(std::string_view corpus) {
    reset_to_base();
    if (corpus.empty() || max_vocab_size_ == base_vocabulary_size) {
        return;
    }

    std::vector<std::vector<TokenId>> sequences;
    split_specials(
        corpus,
        [&](std::string_view text) { sequences.push_back(byte_sequence(text)); },
        [](TokenId) {});

    while (vocabulary_.size() < max_vocab_size_) {
        std::map<Pair, std::uint64_t> pair_counts;
        for (const auto& sequence : sequences) {
            for (std::size_t i = 0; i + 1 < sequence.size(); ++i) {
                ++pair_counts[{sequence[i], sequence[i + 1]}];
            }
        }
        if (pair_counts.empty()) {
            break;
        }

        auto best = pair_counts.begin();
        for (auto it = std::next(best); it != pair_counts.end(); ++it) {
            if (it->second > best->second) {
                best = it;
            }
        }

        const Merge merge{best->first.first, best->first.second};
        const TokenId result_id = static_cast<TokenId>(vocabulary_.size());
        std::string merged_token = vocabulary_[merge.left];
        merged_token += vocabulary_[merge.right];

        vocabulary_.push_back(std::move(merged_token));
        merge_ranks_.emplace(best->first, merges_.size());
        merges_.push_back(merge);

        for (auto& sequence : sequences) {
            apply_merge(sequence, merge, result_id);
        }
    }
}

std::vector<BPETokenizer::TokenId> BPETokenizer::encode(std::string_view text) const {
    std::vector<TokenId> output;
    output.reserve(text.size());
    split_specials(
        text,
        [&](std::string_view normal) { append_encoded(normal, output); },
        [&](TokenId special) { output.push_back(special); });
    return output;
}

std::string BPETokenizer::decode(std::span<const TokenId> ids) const {
    std::string output;
    output.reserve(ids.size());
    for (const TokenId id : ids) {
        if (id >= vocabulary_.size()) {
            throw std::out_of_range("Token ID is out of range");
        }
        output += vocabulary_[id];
    }
    return output;
}

void BPETokenizer::save(const std::filesystem::path& path) const {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot open tokenizer file for writing");
    }

    file.write(file_magic.data(), static_cast<std::streamsize>(file_magic.size()));
    if (!file) {
        throw std::runtime_error("Failed to write tokenizer file");
    }
    write_u32(file, file_version);
    write_u32(file, static_cast<std::uint32_t>(max_vocab_size_));
    write_u32(file, static_cast<std::uint32_t>(vocabulary_.size()));
    write_u32(file, static_cast<std::uint32_t>(merges_.size()));
    for (const std::string& value : vocabulary_) {
        write_string(file, value);
    }
    for (const Merge& merge : merges_) {
        write_u32(file, merge.left);
        write_u32(file, merge.right);
    }

    file.close();
    if (!file) {
        throw std::runtime_error("Failed to close tokenizer file");
    }
}

BPETokenizer BPETokenizer::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open tokenizer file for reading");
    }

    const std::streamoff file_size = file.tellg();
    if (file_size < 0) {
        throw std::runtime_error("Cannot determine tokenizer file size");
    }
    std::uint64_t remaining = static_cast<std::uint64_t>(file_size);
    file.seekg(0, std::ios::beg);

    if (remaining < file_magic.size()) {
        throw std::runtime_error("Invalid tokenizer file header");
    }
    std::array<char, 8> magic{};
    file.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!file || std::string_view(magic.data(), magic.size()) != file_magic) {
        throw std::runtime_error("Invalid tokenizer file magic");
    }
    remaining -= magic.size();

    if (read_u32(file, remaining) != file_version) {
        throw std::runtime_error("Unsupported tokenizer file version");
    }
    const std::size_t max_vocab_size = read_u32(file, remaining);
    const std::size_t vocab_count = read_u32(file, remaining);
    const std::size_t merge_count = read_u32(file, remaining);
    if (vocab_count < base_vocabulary_size || vocab_count > max_vocab_size ||
        merge_count != vocab_count - base_vocabulary_size) {
        throw std::runtime_error("Invalid tokenizer vocabulary counts");
    }

    BPETokenizer tokenizer(max_vocab_size);
    for (std::size_t i = 0; i < vocab_count; ++i) {
        std::string value = read_string(file, remaining);
        if (i < base_vocabulary_size) {
            if (value != tokenizer.vocabulary_[i]) {
                throw std::runtime_error("Tokenizer base vocabulary does not match its IDs");
            }
        } else {
            tokenizer.vocabulary_.push_back(std::move(value));
        }
    }

    for (std::size_t i = 0; i < merge_count; ++i) {
        const Merge merge{read_u32(file, remaining), read_u32(file, remaining)};
        const TokenId result_id = static_cast<TokenId>(base_vocabulary_size + i);
        if (merge.left >= result_id || merge.right >= result_id || is_special(merge.left) ||
            is_special(merge.right)) {
            throw std::runtime_error("Invalid merge in tokenizer file");
        }
        if (tokenizer.vocabulary_[result_id] !=
            tokenizer.vocabulary_[merge.left] + tokenizer.vocabulary_[merge.right]) {
            throw std::runtime_error("Tokenizer merge and vocabulary are inconsistent");
        }

        tokenizer.merges_.push_back(merge);
        if (!tokenizer.merge_ranks_.emplace(
                Pair{merge.left, merge.right}, i).second) {
            throw std::runtime_error("Duplicate merge in tokenizer file");
        }
    }

    if (remaining != 0) {
        throw std::runtime_error("Unexpected trailing data in tokenizer file");
    }
    return tokenizer;
}

std::size_t BPETokenizer::vocab_size() const noexcept {
    return vocabulary_.size();
}

std::size_t BPETokenizer::max_vocab_size() const noexcept {
    return max_vocab_size_;
}

const std::vector<std::string>& BPETokenizer::vocabulary() const noexcept {
    return vocabulary_;
}

const std::vector<BPETokenizer::Merge>& BPETokenizer::merges() const noexcept {
    return merges_;
}

std::string_view BPETokenizer::token(TokenId id) const {
    if (id >= vocabulary_.size()) {
        throw std::out_of_range("Token ID is out of range");
    }
    return vocabulary_[id];
}

std::size_t BPETokenizer::validate_max_vocab_size(std::size_t size) {
    if (size < base_vocabulary_size ||
        size > static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
        throw std::invalid_argument("Vocabulary size must be between 260 and uint32_t max");
    }
    return size;
}

void BPETokenizer::reset_to_base() {
    vocabulary_.resize(base_vocabulary_size);
    merges_.clear();
    merge_ranks_.clear();
}

void BPETokenizer::append_encoded(std::string_view text, std::vector<TokenId>& output) const {
    std::vector<TokenId> sequence = byte_sequence(text);
    while (sequence.size() >= 2) {
        std::size_t best_rank = merges_.size();
        Pair best_pair{};
        for (std::size_t i = 0; i + 1 < sequence.size(); ++i) {
            const Pair pair{sequence[i], sequence[i + 1]};
            const auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pair = pair;
            }
        }
        if (best_rank == merges_.size()) {
            break;
        }
        apply_merge(
            sequence,
            Merge{best_pair.first, best_pair.second},
            static_cast<TokenId>(base_vocabulary_size + best_rank));
    }

    output.insert(output.end(), sequence.begin(), sequence.end());
}

std::size_t BPETokenizer::PairHash::operator()(const Pair& pair) const noexcept {
    const std::uint64_t key =
        (static_cast<std::uint64_t>(pair.first) << 32) | pair.second;
    return std::hash<std::uint64_t>{}(key);
}

}  // namespace mini_gpt

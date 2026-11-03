#include "mini_gpt/embedding.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mini_gpt {
namespace {

constexpr float initialization_scale = 0.02F;
constexpr std::string_view file_magic = "MINIEMBD";
constexpr std::uint32_t file_version = 1;
static_assert(sizeof(float) == sizeof(std::uint32_t));

std::size_t require_positive(std::size_t value, const char* name) {
    if (value == 0) {
        throw std::invalid_argument(std::string(name) + " must be greater than zero");
    }
    return value;
}

void validate_batch(
    std::size_t token_count,
    std::size_t batch_size,
    std::size_t seq_len,
    std::size_t max_seq_len) {
    if (seq_len > max_seq_len) {
        throw std::invalid_argument("Sequence length exceeds max_seq_len");
    }
    if (batch_size != 0 && seq_len > std::numeric_limits<std::size_t>::max() / batch_size) {
        throw std::invalid_argument("Batch dimensions exceed size_t range");
    }
    if (token_count != batch_size * seq_len) {
        throw std::invalid_argument("Token count does not match batch_size * seq_len");
    }
}

void write_u32(std::ofstream& file, std::uint32_t value) {
    std::array<char, 4> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((value >> (i * 8)) & 0xFFU);
    }
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to write embedding file");
    }
}

void write_u64(std::ofstream& file, std::uint64_t value) {
    write_u32(file, static_cast<std::uint32_t>(value & 0xFFFFFFFFU));
    write_u32(file, static_cast<std::uint32_t>(value >> 32));
}

void write_float(std::ofstream& file, float value) {
    write_u32(file, std::bit_cast<std::uint32_t>(value));
}

std::uint32_t read_u32(std::ifstream& file, std::uint64_t& remaining) {
    if (remaining < 4) {
        throw std::runtime_error("Unexpected end of embedding file");
    }

    std::array<unsigned char, 4> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to read embedding file");
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

float read_float(std::ifstream& file, std::uint64_t& remaining) {
    return std::bit_cast<float>(read_u32(file, remaining));
}

std::size_t read_dimension(std::ifstream& file, std::uint64_t& remaining) {
    const std::uint64_t value = read_u64(file, remaining);
    if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Invalid dimension in embedding file");
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_element_count(std::size_t rows, std::size_t columns) {
    if (rows > std::numeric_limits<std::size_t>::max() / columns) {
        throw std::runtime_error("Embedding dimensions exceed size_t range");
    }
    return rows * columns;
}

std::uint64_t checked_byte_count(std::size_t element_count) {
    if (element_count > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
        throw std::runtime_error("Embedding data size exceeds file format range");
    }
    return static_cast<std::uint64_t>(element_count) * sizeof(float);
}

std::vector<float> read_values(
    std::ifstream& file,
    std::uint64_t& remaining,
    std::size_t count) {
    std::vector<float> values(count);
    for (float& value : values) {
        value = read_float(file, remaining);
    }
    return values;
}

}  // namespace

Embedding::Embedding(
    std::size_t vocab_size,
    std::size_t embedding_dim,
    std::size_t max_seq_len,
    std::uint32_t seed)
    : vocab_size_(require_positive(vocab_size, "vocab_size")),
      embedding_dim_(require_positive(embedding_dim, "embedding_dim")),
      max_seq_len_(require_positive(max_seq_len, "max_seq_len")),
      token_embeddings_(Tensor::random(
          {vocab_size_, embedding_dim_}, seed, -initialization_scale, initialization_scale)),
      positional_embeddings_(Tensor::random(
          {max_seq_len_, embedding_dim_},
          seed + 1U,
          -initialization_scale,
          initialization_scale)) {}

Embedding::Embedding(Tensor token_embeddings, Tensor positional_embeddings)
    : vocab_size_(0),
      embedding_dim_(0),
      max_seq_len_(0),
      token_embeddings_(std::move(token_embeddings)),
      positional_embeddings_(std::move(positional_embeddings)) {
    if (token_embeddings_.rank() != 2 || positional_embeddings_.rank() != 2 ||
        token_embeddings_.shape()[1] != positional_embeddings_.shape()[1] ||
        token_embeddings_.shape()[0] == 0 || token_embeddings_.shape()[1] == 0 ||
        positional_embeddings_.shape()[0] == 0) {
        throw std::invalid_argument("Invalid embedding table shapes");
    }
    vocab_size_ = token_embeddings_.shape()[0];
    embedding_dim_ = token_embeddings_.shape()[1];
    max_seq_len_ = positional_embeddings_.shape()[0];
}

Tensor Embedding::token_forward(
    std::span<const TokenId> token_ids,
    std::size_t batch_size,
    std::size_t seq_len) const {
    validate_shapes();
    validate_batch(token_ids.size(), batch_size, seq_len, max_seq_len_);

    Tensor output = Tensor::zeros({batch_size, seq_len, embedding_dim_});
    const float* weights = token_embeddings_.data().data();
    float* output_data = output.data().data();
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < seq_len; ++position) {
            const std::size_t token_index = batch * seq_len + position;
            const TokenId id = token_ids[token_index];
            if (id >= vocab_size_) {
                throw std::out_of_range("Token ID is outside the embedding vocabulary");
            }

            const float* source = weights + static_cast<std::size_t>(id) * embedding_dim_;
            float* destination = output_data + token_index * embedding_dim_;
            std::copy_n(source, embedding_dim_, destination);
        }
    }
    return output;
}

Tensor Embedding::forward(
    std::span<const TokenId> token_ids,
    std::size_t batch_size,
    std::size_t seq_len) const {
    Tensor output = token_forward(token_ids, batch_size, seq_len);
    const float* positions = positional_embeddings_.data().data();
    float* output_data = output.data().data();

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < seq_len; ++position) {
            const std::size_t output_offset =
                (batch * seq_len + position) * embedding_dim_;
            const std::size_t position_offset = position * embedding_dim_;
            for (std::size_t dimension = 0; dimension < embedding_dim_; ++dimension) {
                output_data[output_offset + dimension] +=
                    positions[position_offset + dimension];
            }
        }
    }
    return output;
}

void Embedding::save(const std::filesystem::path& path) const {
    validate_shapes();
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot open embedding file for writing");
    }

    file.write(file_magic.data(), static_cast<std::streamsize>(file_magic.size()));
    if (!file) {
        throw std::runtime_error("Failed to write embedding file");
    }
    write_u32(file, file_version);
    write_u64(file, vocab_size_);
    write_u64(file, embedding_dim_);
    write_u64(file, max_seq_len_);
    for (const float value : token_embeddings_.data()) {
        write_float(file, value);
    }
    for (const float value : positional_embeddings_.data()) {
        write_float(file, value);
    }

    file.close();
    if (!file) {
        throw std::runtime_error("Failed to close embedding file");
    }
}

Embedding Embedding::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open embedding file for reading");
    }

    const std::streamoff file_size = file.tellg();
    if (file_size < 0) {
        throw std::runtime_error("Cannot determine embedding file size");
    }
    std::uint64_t remaining = static_cast<std::uint64_t>(file_size);
    file.seekg(0, std::ios::beg);

    if (remaining < file_magic.size()) {
        throw std::runtime_error("Invalid embedding file header");
    }
    std::array<char, 8> magic{};
    file.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!file || std::string_view(magic.data(), magic.size()) != file_magic) {
        throw std::runtime_error("Invalid embedding file magic");
    }
    remaining -= magic.size();

    if (read_u32(file, remaining) != file_version) {
        throw std::runtime_error("Unsupported embedding file version");
    }
    const std::size_t vocab_size = read_dimension(file, remaining);
    const std::size_t embedding_dim = read_dimension(file, remaining);
    const std::size_t max_seq_len = read_dimension(file, remaining);
    const std::size_t token_count = checked_element_count(vocab_size, embedding_dim);
    const std::size_t position_count = checked_element_count(max_seq_len, embedding_dim);
    const std::uint64_t token_bytes = checked_byte_count(token_count);
    const std::uint64_t position_bytes = checked_byte_count(position_count);
    if (token_bytes > remaining || position_bytes != remaining - token_bytes) {
        throw std::runtime_error("Embedding dimensions do not match file size");
    }

    Tensor token_embeddings(
        {vocab_size, embedding_dim}, read_values(file, remaining, token_count));
    Tensor positional_embeddings(
        {max_seq_len, embedding_dim}, read_values(file, remaining, position_count));
    return Embedding(std::move(token_embeddings), std::move(positional_embeddings));
}

std::size_t Embedding::vocab_size() const noexcept {
    return vocab_size_;
}

std::size_t Embedding::embedding_dim() const noexcept {
    return embedding_dim_;
}

std::size_t Embedding::max_seq_len() const noexcept {
    return max_seq_len_;
}

Tensor& Embedding::token_embeddings() noexcept {
    return token_embeddings_;
}

const Tensor& Embedding::token_embeddings() const noexcept {
    return token_embeddings_;
}

Tensor& Embedding::positional_embeddings() noexcept {
    return positional_embeddings_;
}

const Tensor& Embedding::positional_embeddings() const noexcept {
    return positional_embeddings_;
}

void Embedding::validate_shapes() const {
    if (token_embeddings_.shape() != Tensor::Shape{vocab_size_, embedding_dim_} ||
        positional_embeddings_.shape() != Tensor::Shape{max_seq_len_, embedding_dim_}) {
        throw std::logic_error("Embedding table shapes do not match the layer configuration");
    }
}

}  // namespace mini_gpt

#include "mini_gpt/causal_self_attention.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
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
constexpr std::string_view file_magic = "MINIATTN";
constexpr std::uint32_t file_version = 1;
static_assert(sizeof(float) == sizeof(std::uint32_t));

std::size_t require_positive(std::size_t value, const char* name) {
    if (value == 0) {
        throw std::invalid_argument(std::string(name) + " must be greater than zero");
    }
    return value;
}

std::size_t validated_head_dim(std::size_t embedding_dim, std::size_t num_heads) {
    if (embedding_dim % num_heads != 0) {
        throw std::invalid_argument("embedding_dim must be divisible by num_heads");
    }
    return embedding_dim / num_heads;
}

Tensor linear(const Tensor& input, const Tensor& weights) {
    const std::size_t batch_size = input.shape()[0];
    const std::size_t seq_len = input.shape()[1];
    const std::size_t input_dim = input.shape()[2];
    const std::size_t output_dim = weights.shape()[1];
    const std::size_t rows = batch_size * seq_len;

    Tensor output = Tensor::zeros({batch_size, seq_len, output_dim});
    const float* input_data = input.data().data();
    const float* weight_data = weights.data().data();
    float* output_data = output.data().data();
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t inner = 0; inner < input_dim; ++inner) {
            const float input_value = input_data[row * input_dim + inner];
            for (std::size_t column = 0; column < output_dim; ++column) {
                output_data[row * output_dim + column] +=
                    input_value * weight_data[inner * output_dim + column];
            }
        }
    }
    return output;
}

Tensor split_heads(
    const Tensor& input,
    std::size_t num_heads,
    std::size_t head_dim) {
    const std::size_t batch_size = input.shape()[0];
    const std::size_t seq_len = input.shape()[1];
    const std::size_t embedding_dim = input.shape()[2];
    Tensor output = Tensor::zeros({batch_size, num_heads, seq_len, head_dim});
    const float* input_data = input.data().data();
    float* output_data = output.data().data();

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t head = 0; head < num_heads; ++head) {
            for (std::size_t position = 0; position < seq_len; ++position) {
                const float* source =
                    input_data + (batch * seq_len + position) * embedding_dim + head * head_dim;
                float* destination =
                    output_data + ((batch * num_heads + head) * seq_len + position) * head_dim;
                std::copy_n(source, head_dim, destination);
            }
        }
    }
    return output;
}

void write_u32(std::ofstream& file, std::uint32_t value) {
    std::array<char, 4> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((value >> (i * 8)) & 0xFFU);
    }
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to write attention file");
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
        throw std::runtime_error("Unexpected end of attention file");
    }
    std::array<unsigned char, 4> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to read attention file");
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
        throw std::runtime_error("Invalid dimension in attention file");
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_matrix_size(std::size_t embedding_dim) {
    if (embedding_dim > std::numeric_limits<std::size_t>::max() / embedding_dim) {
        throw std::runtime_error("Attention dimensions exceed size_t range");
    }
    return embedding_dim * embedding_dim;
}

std::uint64_t checked_byte_count(std::size_t element_count) {
    if (element_count > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
        throw std::runtime_error("Attention data size exceeds file format range");
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

CausalSelfAttention::CausalSelfAttention(
    std::size_t embedding_dim,
    std::size_t num_heads,
    std::uint32_t seed)
    : embedding_dim_(require_positive(embedding_dim, "embedding_dim")),
      num_heads_(require_positive(num_heads, "num_heads")),
      head_dim_(validated_head_dim(embedding_dim_, num_heads_)),
      q_weights_(Tensor::random(
          {embedding_dim_, embedding_dim_}, seed, -initialization_scale, initialization_scale)),
      k_weights_(Tensor::random(
          {embedding_dim_, embedding_dim_},
          seed + 1U,
          -initialization_scale,
          initialization_scale)),
      v_weights_(Tensor::random(
          {embedding_dim_, embedding_dim_},
          seed + 2U,
          -initialization_scale,
          initialization_scale)),
      output_weights_(Tensor::random(
          {embedding_dim_, embedding_dim_},
          seed + 3U,
          -initialization_scale,
          initialization_scale)) {}

CausalSelfAttention::CausalSelfAttention(
    std::size_t num_heads,
    Tensor q_weights,
    Tensor k_weights,
    Tensor v_weights,
    Tensor output_weights)
    : embedding_dim_(0),
      num_heads_(num_heads),
      head_dim_(0),
      q_weights_(std::move(q_weights)),
      k_weights_(std::move(k_weights)),
      v_weights_(std::move(v_weights)),
      output_weights_(std::move(output_weights)) {
    if (q_weights_.rank() != 2 || k_weights_.rank() != 2 || v_weights_.rank() != 2 ||
        output_weights_.rank() != 2) {
        throw std::invalid_argument("Attention parameters must be matrices");
    }

    const std::size_t embedding_dim = q_weights_.shape()[0];
    const Tensor::Shape expected{embedding_dim, embedding_dim};
    if (embedding_dim == 0 || num_heads_ == 0 || q_weights_.shape() != expected ||
        k_weights_.shape() != expected || v_weights_.shape() != expected ||
        output_weights_.shape() != expected || embedding_dim % num_heads_ != 0) {
        throw std::invalid_argument("Invalid attention parameter dimensions");
    }
    embedding_dim_ = embedding_dim;
    head_dim_ = embedding_dim_ / num_heads_;
}

Tensor CausalSelfAttention::forward(const Tensor& input) const {
    AttentionForwardResult result = forward_with_details(input);
    return std::move(result.output);
}

AttentionForwardResult CausalSelfAttention::forward_with_details(const Tensor& input) const {
    validate_parameter_shapes();
    if (input.rank() != 3 || input.shape()[2] != embedding_dim_) {
        throw std::invalid_argument(
            "Attention input must have shape [batch_size, seq_len, embedding_dim]");
    }

    Tensor queries = split_heads(linear(input, q_weights_), num_heads_, head_dim_);
    Tensor keys = split_heads(linear(input, k_weights_), num_heads_, head_dim_);
    Tensor values = split_heads(linear(input, v_weights_), num_heads_, head_dim_);
    return attend(std::move(queries), std::move(keys), std::move(values));
}

AttentionForwardResult CausalSelfAttention::forward_from_qkv(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values) const {
    validate_parameter_shapes();
    if (queries.rank() != 4 || keys.rank() != 4 || values.rank() != 4 ||
        queries.shape() != keys.shape() || queries.shape() != values.shape() ||
        queries.shape()[1] != num_heads_ || queries.shape()[3] != head_dim_) {
        throw std::invalid_argument(
            "Q, K and V must have matching shape [batch_size, num_heads, seq_len, head_dim]");
    }
    return attend(queries, keys, values);
}

AttentionForwardResult CausalSelfAttention::attend(
    Tensor queries,
    Tensor keys,
    Tensor values) const {
    const std::size_t batch_size = queries.shape()[0];
    const std::size_t seq_len = queries.shape()[2];
    Tensor logits = Tensor::zeros({batch_size, num_heads_, seq_len, seq_len});
    std::fill(
        logits.data().begin(),
        logits.data().end(),
        -std::numeric_limits<float>::infinity());
    Tensor probabilities = Tensor::zeros({batch_size, num_heads_, seq_len, seq_len});
    Tensor combined = Tensor::zeros({batch_size, seq_len, embedding_dim_});

    const float* query_data = queries.data().data();
    const float* key_data = keys.data().data();
    const float* value_data = values.data().data();
    float* logit_data = logits.data().data();
    float* probability_data = probabilities.data().data();
    float* combined_data = combined.data().data();
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
    std::vector<float> scores(seq_len);

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t head = 0; head < num_heads_; ++head) {
            for (std::size_t query = 0; query < seq_len; ++query) {
                const std::size_t query_offset =
                    ((batch * num_heads_ + head) * seq_len + query) * head_dim_;
                const std::size_t probability_offset =
                    ((batch * num_heads_ + head) * seq_len + query) * seq_len;

                for (std::size_t key = 0; key <= query; ++key) {
                    const std::size_t key_offset =
                        ((batch * num_heads_ + head) * seq_len + key) * head_dim_;
                    float score = 0.0F;
                    for (std::size_t dimension = 0; dimension < head_dim_; ++dimension) {
                        score += query_data[query_offset + dimension] *
                                 key_data[key_offset + dimension];
                    }
                    scores[key] = score * scale;
                    logit_data[probability_offset + key] = scores[key];
                }

                float maximum = scores[0];
                for (std::size_t key = 1; key <= query; ++key) {
                    maximum = std::max(maximum, scores[key]);
                }
                float denominator = 0.0F;
                for (std::size_t key = 0; key <= query; ++key) {
                    const float weight = std::exp(scores[key] - maximum);
                    probability_data[probability_offset + key] = weight;
                    denominator += weight;
                }
                for (std::size_t key = 0; key <= query; ++key) {
                    const float weight =
                        probability_data[probability_offset + key] / denominator;
                    probability_data[probability_offset + key] = weight;
                    const std::size_t value_offset =
                        ((batch * num_heads_ + head) * seq_len + key) * head_dim_;
                    const std::size_t output_offset =
                        (batch * seq_len + query) * embedding_dim_ + head * head_dim_;
                    for (std::size_t dimension = 0; dimension < head_dim_; ++dimension) {
                        combined_data[output_offset + dimension] +=
                            weight * value_data[value_offset + dimension];
                    }
                }
            }
        }
    }

    Tensor output = linear(combined, output_weights_);
    return AttentionForwardResult{
        std::move(output),
        std::move(combined),
        std::move(queries),
        std::move(keys),
        std::move(values),
        std::move(logits),
        std::move(probabilities)};
}

void CausalSelfAttention::save(const std::filesystem::path& path) const {
    validate_parameter_shapes();
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot open attention file for writing");
    }

    file.write(file_magic.data(), static_cast<std::streamsize>(file_magic.size()));
    if (!file) {
        throw std::runtime_error("Failed to write attention file");
    }
    write_u32(file, file_version);
    write_u64(file, embedding_dim_);
    write_u64(file, num_heads_);
    for (const Tensor* weights : {&q_weights_, &k_weights_, &v_weights_, &output_weights_}) {
        for (const float value : weights->data()) {
            write_float(file, value);
        }
    }

    file.close();
    if (!file) {
        throw std::runtime_error("Failed to close attention file");
    }
}

CausalSelfAttention CausalSelfAttention::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open attention file for reading");
    }
    const std::streamoff file_size = file.tellg();
    if (file_size < 0) {
        throw std::runtime_error("Cannot determine attention file size");
    }
    std::uint64_t remaining = static_cast<std::uint64_t>(file_size);
    file.seekg(0, std::ios::beg);

    if (remaining < file_magic.size()) {
        throw std::runtime_error("Invalid attention file header");
    }
    std::array<char, 8> magic{};
    file.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!file || std::string_view(magic.data(), magic.size()) != file_magic) {
        throw std::runtime_error("Invalid attention file magic");
    }
    remaining -= magic.size();

    if (read_u32(file, remaining) != file_version) {
        throw std::runtime_error("Unsupported attention file version");
    }
    const std::size_t embedding_dim = read_dimension(file, remaining);
    const std::size_t num_heads = read_dimension(file, remaining);
    if (embedding_dim % num_heads != 0) {
        throw std::runtime_error("Invalid attention head configuration");
    }
    const std::size_t matrix_size = checked_matrix_size(embedding_dim);
    const std::uint64_t matrix_bytes = checked_byte_count(matrix_size);
    if (matrix_bytes > remaining / 4 || remaining != matrix_bytes * 4) {
        throw std::runtime_error("Attention dimensions do not match file size");
    }

    Tensor q_weights({embedding_dim, embedding_dim}, read_values(file, remaining, matrix_size));
    Tensor k_weights({embedding_dim, embedding_dim}, read_values(file, remaining, matrix_size));
    Tensor v_weights({embedding_dim, embedding_dim}, read_values(file, remaining, matrix_size));
    Tensor output_weights(
        {embedding_dim, embedding_dim}, read_values(file, remaining, matrix_size));
    return CausalSelfAttention(
        num_heads,
        std::move(q_weights),
        std::move(k_weights),
        std::move(v_weights),
        std::move(output_weights));
}

std::size_t CausalSelfAttention::embedding_dim() const noexcept {
    return embedding_dim_;
}

std::size_t CausalSelfAttention::num_heads() const noexcept {
    return num_heads_;
}

std::size_t CausalSelfAttention::head_dim() const noexcept {
    return head_dim_;
}

Tensor& CausalSelfAttention::q_weights() noexcept {
    return q_weights_;
}

const Tensor& CausalSelfAttention::q_weights() const noexcept {
    return q_weights_;
}

Tensor& CausalSelfAttention::k_weights() noexcept {
    return k_weights_;
}

const Tensor& CausalSelfAttention::k_weights() const noexcept {
    return k_weights_;
}

Tensor& CausalSelfAttention::v_weights() noexcept {
    return v_weights_;
}

const Tensor& CausalSelfAttention::v_weights() const noexcept {
    return v_weights_;
}

Tensor& CausalSelfAttention::output_weights() noexcept {
    return output_weights_;
}

const Tensor& CausalSelfAttention::output_weights() const noexcept {
    return output_weights_;
}

void CausalSelfAttention::validate_parameter_shapes() const {
    const Tensor::Shape expected{embedding_dim_, embedding_dim_};
    if (q_weights_.shape() != expected || k_weights_.shape() != expected ||
        v_weights_.shape() != expected || output_weights_.shape() != expected) {
        throw std::logic_error("Attention parameter shapes do not match the layer configuration");
    }
}

}  // namespace mini_gpt

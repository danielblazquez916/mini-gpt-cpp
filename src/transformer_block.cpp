#include "mini_gpt/transformer_block.hpp"

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
constexpr std::string_view file_magic = "MINITRFB";
constexpr std::uint32_t file_version = 1;
static_assert(sizeof(float) == sizeof(std::uint32_t));

std::size_t require_positive(std::size_t value) {
    if (value == 0) {
        throw std::invalid_argument("mlp_hidden_dim must be greater than zero");
    }
    return value;
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

void write_u32(std::ofstream& file, std::uint32_t value) {
    std::array<char, 4> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((value >> (i * 8)) & 0xFFU);
    }
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to write Transformer Block file");
    }
}

void write_u64(std::ofstream& file, std::uint64_t value) {
    write_u32(file, static_cast<std::uint32_t>(value & 0xFFFFFFFFU));
    write_u32(file, static_cast<std::uint32_t>(value >> 32));
}

void write_float(std::ofstream& file, float value) {
    write_u32(file, std::bit_cast<std::uint32_t>(value));
}

void write_tensor(std::ofstream& file, const Tensor& tensor) {
    for (const float value : tensor.data()) {
        write_float(file, value);
    }
}

std::uint32_t read_u32(std::ifstream& file, std::uint64_t& remaining) {
    if (remaining < 4) {
        throw std::runtime_error("Unexpected end of Transformer Block file");
    }
    std::array<unsigned char, 4> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to read Transformer Block file");
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
        throw std::runtime_error("Invalid dimension in Transformer Block file");
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::runtime_error("Transformer Block dimensions exceed size_t range");
    }
    return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::runtime_error("Transformer Block parameter count exceeds size_t range");
    }
    return lhs + rhs;
}

std::uint64_t checked_byte_count(std::size_t element_count) {
    if (element_count > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
        throw std::runtime_error("Transformer Block data size exceeds file format range");
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

Tensor read_tensor(
    std::ifstream& file,
    std::uint64_t& remaining,
    const Tensor::Shape& shape,
    std::size_t count) {
    return Tensor(shape, read_values(file, remaining, count));
}

}  // namespace

Tensor gelu(const Tensor& input) {
    Tensor output = Tensor::zeros(input.shape());
    for (std::size_t i = 0; i < input.size(); ++i) {
        const double value = input.data()[i];
        const double activated =
            0.5 * value * (1.0 + std::erf(value / std::sqrt(2.0)));
        output.data()[i] = static_cast<float>(activated);
    }
    return output;
}

TransformerBlock::TransformerBlock(
    std::size_t embedding_dim,
    std::size_t num_heads,
    std::size_t mlp_hidden_dim,
    std::uint32_t seed,
    float epsilon)
    : mlp_hidden_dim_(require_positive(mlp_hidden_dim)),
      attention_(embedding_dim, num_heads, seed),
      norm1_(embedding_dim, epsilon),
      norm2_(embedding_dim, epsilon),
      fc1_weights_(Tensor::random(
          {embedding_dim, mlp_hidden_dim_},
          seed + 4U,
          -initialization_scale,
          initialization_scale)),
      fc2_weights_(Tensor::random(
          {mlp_hidden_dim_, embedding_dim},
          seed + 5U,
          -initialization_scale,
          initialization_scale)) {}

Tensor TransformerBlock::forward(const Tensor& input) const {
    validate_parameters();
    if (input.rank() != 3 || input.shape()[2] != embedding_dim()) {
        throw std::invalid_argument(
            "Transformer Block input must have shape [batch_size, seq_len, embedding_dim]");
    }

    const Tensor normalized_attention = norm1_.forward(input);
    const Tensor attention_output = attention_.forward(normalized_attention);
    const Tensor residual = input + attention_output;

    const Tensor normalized_mlp = norm2_.forward(residual);
    const Tensor hidden = linear(normalized_mlp, fc1_weights_);
    const Tensor activated = gelu(hidden);
    const Tensor mlp_output = linear(activated, fc2_weights_);
    return residual + mlp_output;
}

void TransformerBlock::save(const std::filesystem::path& path) const {
    validate_parameters();
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot open Transformer Block file for writing");
    }

    file.write(file_magic.data(), static_cast<std::streamsize>(file_magic.size()));
    if (!file) {
        throw std::runtime_error("Failed to write Transformer Block file");
    }
    write_u32(file, file_version);
    write_u64(file, embedding_dim());
    write_u64(file, num_heads());
    write_u64(file, mlp_hidden_dim_);
    write_float(file, epsilon());
    for (const Tensor* tensor : {
             &norm1_.gamma(), &norm1_.beta(), &norm2_.gamma(), &norm2_.beta(),
             &attention_.q_weights(), &attention_.k_weights(),
             &attention_.v_weights(), &attention_.output_weights(),
             &fc1_weights_, &fc2_weights_}) {
        write_tensor(file, *tensor);
    }

    file.close();
    if (!file) {
        throw std::runtime_error("Failed to close Transformer Block file");
    }
}

TransformerBlock TransformerBlock::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open Transformer Block file for reading");
    }
    const std::streamoff file_size = file.tellg();
    if (file_size < 0) {
        throw std::runtime_error("Cannot determine Transformer Block file size");
    }
    std::uint64_t remaining = static_cast<std::uint64_t>(file_size);
    file.seekg(0, std::ios::beg);

    if (remaining < file_magic.size()) {
        throw std::runtime_error("Invalid Transformer Block file header");
    }
    std::array<char, 8> magic{};
    file.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!file || std::string_view(magic.data(), magic.size()) != file_magic) {
        throw std::runtime_error("Invalid Transformer Block file magic");
    }
    remaining -= magic.size();

    if (read_u32(file, remaining) != file_version) {
        throw std::runtime_error("Unsupported Transformer Block file version");
    }
    const std::size_t embedding_dim = read_dimension(file, remaining);
    const std::size_t num_heads = read_dimension(file, remaining);
    const std::size_t mlp_hidden_dim = read_dimension(file, remaining);
    if (embedding_dim % num_heads != 0) {
        throw std::runtime_error("Invalid Transformer Block head configuration");
    }
    const float epsilon = read_float(file, remaining);
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::runtime_error("Invalid epsilon in Transformer Block file");
    }

    const std::size_t matrix_size = checked_multiply(embedding_dim, embedding_dim);
    const std::size_t mlp_matrix_size = checked_multiply(embedding_dim, mlp_hidden_dim);
    std::size_t total_values = checked_multiply(4, embedding_dim);
    total_values = checked_add(total_values, checked_multiply(4, matrix_size));
    total_values = checked_add(total_values, checked_multiply(2, mlp_matrix_size));
    if (checked_byte_count(total_values) != remaining) {
        throw std::runtime_error("Transformer Block dimensions do not match file size");
    }

    TransformerBlock block(embedding_dim, num_heads, mlp_hidden_dim, 0, epsilon);
    block.norm1().gamma() = read_tensor(file, remaining, {embedding_dim}, embedding_dim);
    block.norm1().beta() = read_tensor(file, remaining, {embedding_dim}, embedding_dim);
    block.norm2().gamma() = read_tensor(file, remaining, {embedding_dim}, embedding_dim);
    block.norm2().beta() = read_tensor(file, remaining, {embedding_dim}, embedding_dim);
    block.attention().q_weights() =
        read_tensor(file, remaining, {embedding_dim, embedding_dim}, matrix_size);
    block.attention().k_weights() =
        read_tensor(file, remaining, {embedding_dim, embedding_dim}, matrix_size);
    block.attention().v_weights() =
        read_tensor(file, remaining, {embedding_dim, embedding_dim}, matrix_size);
    block.attention().output_weights() =
        read_tensor(file, remaining, {embedding_dim, embedding_dim}, matrix_size);
    block.fc1_weights() =
        read_tensor(file, remaining, {embedding_dim, mlp_hidden_dim}, mlp_matrix_size);
    block.fc2_weights() =
        read_tensor(file, remaining, {mlp_hidden_dim, embedding_dim}, mlp_matrix_size);
    return block;
}

std::size_t TransformerBlock::embedding_dim() const noexcept {
    return attention_.embedding_dim();
}

std::size_t TransformerBlock::num_heads() const noexcept {
    return attention_.num_heads();
}

std::size_t TransformerBlock::mlp_hidden_dim() const noexcept {
    return mlp_hidden_dim_;
}

float TransformerBlock::epsilon() const noexcept {
    return norm1_.epsilon();
}

CausalSelfAttention& TransformerBlock::attention() noexcept {
    return attention_;
}

const CausalSelfAttention& TransformerBlock::attention() const noexcept {
    return attention_;
}

LayerNorm& TransformerBlock::norm1() noexcept {
    return norm1_;
}

const LayerNorm& TransformerBlock::norm1() const noexcept {
    return norm1_;
}

LayerNorm& TransformerBlock::norm2() noexcept {
    return norm2_;
}

const LayerNorm& TransformerBlock::norm2() const noexcept {
    return norm2_;
}

Tensor& TransformerBlock::fc1_weights() noexcept {
    return fc1_weights_;
}

const Tensor& TransformerBlock::fc1_weights() const noexcept {
    return fc1_weights_;
}

Tensor& TransformerBlock::fc2_weights() noexcept {
    return fc2_weights_;
}

const Tensor& TransformerBlock::fc2_weights() const noexcept {
    return fc2_weights_;
}

void TransformerBlock::validate_parameters() const {
    const std::size_t dimension = embedding_dim();
    const Tensor::Shape vector_shape{dimension};
    const Tensor::Shape attention_shape{dimension, dimension};
    if (norm1_.gamma().shape() != vector_shape || norm1_.beta().shape() != vector_shape ||
        norm2_.gamma().shape() != vector_shape || norm2_.beta().shape() != vector_shape ||
        attention_.q_weights().shape() != attention_shape ||
        attention_.k_weights().shape() != attention_shape ||
        attention_.v_weights().shape() != attention_shape ||
        attention_.output_weights().shape() != attention_shape ||
        fc1_weights_.shape() != Tensor::Shape{dimension, mlp_hidden_dim_} ||
        fc2_weights_.shape() != Tensor::Shape{mlp_hidden_dim_, dimension}) {
        throw std::logic_error("Transformer Block parameters have incompatible shapes");
    }
}

}  // namespace mini_gpt

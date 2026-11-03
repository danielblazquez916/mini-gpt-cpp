#include "mini_gpt/gpt.hpp"

#include "mini_gpt/bpe_tokenizer.hpp"

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
constexpr std::string_view file_magic = "MINIGPTM";
constexpr std::uint32_t file_version = 1;
static_assert(sizeof(float) == sizeof(std::uint32_t));

std::size_t require_positive_layers(std::size_t value) {
    if (value == 0) {
        throw std::invalid_argument("num_layers must be greater than zero");
    }
    return value;
}

std::uint32_t component_seed(std::uint32_t seed, std::size_t component) {
    return seed + static_cast<std::uint32_t>(6U * (component + 1U));
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
        throw std::runtime_error("Failed to write GPT file");
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
        throw std::runtime_error("Unexpected end of GPT file");
    }
    std::array<unsigned char, 4> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to read GPT file");
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
        throw std::runtime_error("Invalid dimension in GPT file");
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::runtime_error("GPT dimensions exceed size_t range");
    }
    return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::runtime_error("GPT parameter count exceeds size_t range");
    }
    return lhs + rhs;
}

std::size_t checked_parameter_total(
    std::size_t vocab_size,
    std::size_t embedding_dim,
    std::size_t num_layers,
    std::size_t mlp_hidden_dim,
    std::size_t max_seq_len) {
    const std::size_t token_values = checked_multiply(vocab_size, embedding_dim);
    const std::size_t position_values = checked_multiply(max_seq_len, embedding_dim);
    const std::size_t attention_values = checked_multiply(
        4, checked_multiply(embedding_dim, embedding_dim));
    const std::size_t mlp_values = checked_multiply(
        2, checked_multiply(embedding_dim, mlp_hidden_dim));
    std::size_t block_values = checked_multiply(4, embedding_dim);
    block_values = checked_add(block_values, attention_values);
    block_values = checked_add(block_values, mlp_values);

    std::size_t total = checked_add(token_values, position_values);
    total = checked_add(total, checked_multiply(num_layers, block_values));
    total = checked_add(total, checked_multiply(2, embedding_dim));
    total = checked_add(total, checked_multiply(embedding_dim, vocab_size));
    return total;
}

std::uint64_t checked_byte_count(std::size_t element_count) {
    if (element_count > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
        throw std::runtime_error("GPT data size exceeds file format range");
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

void write_block(std::ofstream& file, const TransformerBlock& block) {
    for (const Tensor* tensor : {
             &block.norm1().gamma(), &block.norm1().beta(),
             &block.norm2().gamma(), &block.norm2().beta(),
             &block.attention().q_weights(), &block.attention().k_weights(),
             &block.attention().v_weights(), &block.attention().output_weights(),
             &block.fc1_weights(), &block.fc2_weights()}) {
        write_tensor(file, *tensor);
    }
}

void read_block(
    std::ifstream& file,
    std::uint64_t& remaining,
    TransformerBlock& block,
    std::size_t embedding_dim,
    std::size_t hidden_dim) {
    const std::size_t attention_values = embedding_dim * embedding_dim;
    const std::size_t mlp_values = embedding_dim * hidden_dim;
    block.norm1().gamma() = read_tensor(file, remaining, {embedding_dim}, embedding_dim);
    block.norm1().beta() = read_tensor(file, remaining, {embedding_dim}, embedding_dim);
    block.norm2().gamma() = read_tensor(file, remaining, {embedding_dim}, embedding_dim);
    block.norm2().beta() = read_tensor(file, remaining, {embedding_dim}, embedding_dim);
    block.attention().q_weights() =
        read_tensor(file, remaining, {embedding_dim, embedding_dim}, attention_values);
    block.attention().k_weights() =
        read_tensor(file, remaining, {embedding_dim, embedding_dim}, attention_values);
    block.attention().v_weights() =
        read_tensor(file, remaining, {embedding_dim, embedding_dim}, attention_values);
    block.attention().output_weights() =
        read_tensor(file, remaining, {embedding_dim, embedding_dim}, attention_values);
    block.fc1_weights() =
        read_tensor(file, remaining, {embedding_dim, hidden_dim}, mlp_values);
    block.fc2_weights() =
        read_tensor(file, remaining, {hidden_dim, embedding_dim}, mlp_values);
}

}  // namespace

GPT::GPT(
    std::size_t vocab_size,
    std::size_t embedding_dim,
    std::size_t num_heads,
    std::size_t num_layers,
    std::size_t mlp_hidden_dim,
    std::size_t max_seq_len,
    std::uint32_t seed,
    float epsilon)
    : embedding_(vocab_size, embedding_dim, max_seq_len, seed),
      final_norm_(embedding_dim, epsilon),
      output_weights_(Tensor::random(
          {embedding_dim, vocab_size},
          component_seed(seed, num_layers),
          -initialization_scale,
          initialization_scale)) {
    require_positive_layers(num_layers);
    blocks_.reserve(num_layers);
    for (std::size_t layer = 0; layer < num_layers; ++layer) {
        blocks_.emplace_back(
            embedding_dim,
            num_heads,
            mlp_hidden_dim,
            component_seed(seed, layer),
            epsilon);
    }
}

Tensor GPT::forward(
    std::span<const TokenId> token_ids,
    std::size_t batch_size,
    std::size_t seq_len) const {
    validate_parameters();
    return linear(forward_features(token_ids, batch_size, seq_len), output_weights_);
}

std::vector<GPT::TokenId> GPT::generate(
    std::span<const TokenId> prompt,
    const GenerationConfig& config) const {
    if (prompt.empty()) {
        throw std::invalid_argument("Generation requires a non-empty prompt");
    }
    for (const TokenId id : prompt) {
        if (id >= vocab_size()) {
            throw std::out_of_range("Prompt token ID is outside the model vocabulary");
        }
    }

    std::vector<TokenId> generated;
    generated.reserve(config.max_new_tokens);
    if (config.max_new_tokens == 0) {
        return generated;
    }

    std::vector<TokenId> context(prompt.begin(), prompt.end());
    std::mt19937 generator(config.seed);
    for (std::size_t step = 0; step < config.max_new_tokens; ++step) {
        if (context.size() > max_seq_len()) {
            const std::size_t excess = context.size() - max_seq_len();
            context.erase(context.begin(), context.begin() + excess);
        }

        const Tensor logits = forward(context, 1, context.size());
        const auto last_logits = logits.data().subspan(
            (context.size() - 1) * vocab_size(), vocab_size());
        const TokenId next = select_token(
            last_logits, config.temperature, config.strategy, generator);
        context.push_back(next);
        generated.push_back(next);

        const TokenId stop_id = config.stop_condition == StopCondition::end_of_turn
            ? BPETokenizer::eot_id
            : BPETokenizer::end_id;
        if (next == stop_id) {
            break;
        }
    }
    return generated;
}

std::vector<float> GPT::softmax(std::span<const float> logits, float temperature) {
    if (logits.empty()) {
        throw std::invalid_argument("softmax requires at least one logit");
    }
    if (!std::isfinite(temperature) || temperature <= 0.0F) {
        throw std::invalid_argument("Temperature must be finite and greater than zero");
    }

    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities(logits.size());
    double denominator = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) {
            throw std::invalid_argument("Logits must be finite");
        }
        const double weight = std::exp(
            (static_cast<double>(logits[i]) - maximum) /
            static_cast<double>(temperature));
        probabilities[i] = static_cast<float>(weight);
        denominator += weight;
    }
    for (float& probability : probabilities) {
        probability = static_cast<float>(probability / denominator);
    }
    return probabilities;
}

GPT::TokenId GPT::select_token(
    std::span<const float> logits,
    float temperature,
    SamplingStrategy strategy,
    std::mt19937& generator) {
    if (logits.empty()) {
        throw std::invalid_argument("Token selection requires at least one logit");
    }
    if (logits.size() > static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
        throw std::invalid_argument("Logit count exceeds TokenId range");
    }
    if (strategy == SamplingStrategy::greedy) {
        return static_cast<TokenId>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    if (strategy != SamplingStrategy::sampling) {
        throw std::invalid_argument("Unknown sampling strategy");
    }

    const std::vector<float> probabilities = softmax(logits, temperature);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const double draw = distribution(generator);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (draw < cumulative) {
            return static_cast<TokenId>(i);
        }
    }
    return static_cast<TokenId>(probabilities.size() - 1);
}

void GPT::save(const std::filesystem::path& path) const {
    validate_parameters();
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot open GPT file for writing");
    }

    file.write(file_magic.data(), static_cast<std::streamsize>(file_magic.size()));
    if (!file) {
        throw std::runtime_error("Failed to write GPT file");
    }
    write_u32(file, file_version);
    write_u64(file, vocab_size());
    write_u64(file, embedding_dim());
    write_u64(file, num_heads());
    write_u64(file, num_layers());
    write_u64(file, mlp_hidden_dim());
    write_u64(file, max_seq_len());
    write_float(file, epsilon());
    write_tensor(file, embedding_.token_embeddings());
    write_tensor(file, embedding_.positional_embeddings());
    for (const TransformerBlock& block : blocks_) {
        write_block(file, block);
    }
    write_tensor(file, final_norm_.gamma());
    write_tensor(file, final_norm_.beta());
    write_tensor(file, output_weights_);

    file.close();
    if (!file) {
        throw std::runtime_error("Failed to close GPT file");
    }
}

GPT GPT::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open GPT file for reading");
    }
    const std::streamoff file_size = file.tellg();
    if (file_size < 0) {
        throw std::runtime_error("Cannot determine GPT file size");
    }
    std::uint64_t remaining = static_cast<std::uint64_t>(file_size);
    file.seekg(0, std::ios::beg);

    if (remaining < file_magic.size()) {
        throw std::runtime_error("Invalid GPT file header");
    }
    std::array<char, 8> magic{};
    file.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!file || std::string_view(magic.data(), magic.size()) != file_magic) {
        throw std::runtime_error("Invalid GPT file magic");
    }
    remaining -= magic.size();

    if (read_u32(file, remaining) != file_version) {
        throw std::runtime_error("Unsupported GPT file version");
    }
    const std::size_t vocab_size = read_dimension(file, remaining);
    const std::size_t embedding_dim = read_dimension(file, remaining);
    const std::size_t num_heads = read_dimension(file, remaining);
    const std::size_t num_layers = read_dimension(file, remaining);
    const std::size_t mlp_hidden_dim = read_dimension(file, remaining);
    const std::size_t max_seq_len = read_dimension(file, remaining);
    if (embedding_dim % num_heads != 0) {
        throw std::runtime_error("Invalid GPT head configuration");
    }
    const float epsilon = read_float(file, remaining);
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::runtime_error("Invalid epsilon in GPT file");
    }
    const std::size_t total_values = checked_parameter_total(
        vocab_size, embedding_dim, num_layers, mlp_hidden_dim, max_seq_len);
    if (checked_byte_count(total_values) != remaining) {
        throw std::runtime_error("GPT dimensions do not match file size");
    }

    GPT model(
        vocab_size,
        embedding_dim,
        num_heads,
        num_layers,
        mlp_hidden_dim,
        max_seq_len,
        0,
        epsilon);
    model.embedding().token_embeddings() = read_tensor(
        file,
        remaining,
        {vocab_size, embedding_dim},
        checked_multiply(vocab_size, embedding_dim));
    model.embedding().positional_embeddings() = read_tensor(
        file,
        remaining,
        {max_seq_len, embedding_dim},
        checked_multiply(max_seq_len, embedding_dim));
    for (TransformerBlock& block : model.blocks_) {
        read_block(file, remaining, block, embedding_dim, mlp_hidden_dim);
    }
    model.final_norm().gamma() = read_tensor(file, remaining, {embedding_dim}, embedding_dim);
    model.final_norm().beta() = read_tensor(file, remaining, {embedding_dim}, embedding_dim);
    model.output_weights() = read_tensor(
        file,
        remaining,
        {embedding_dim, vocab_size},
        checked_multiply(embedding_dim, vocab_size));
    return model;
}

std::size_t GPT::vocab_size() const noexcept {
    return embedding_.vocab_size();
}

std::size_t GPT::embedding_dim() const noexcept {
    return embedding_.embedding_dim();
}

std::size_t GPT::num_heads() const noexcept {
    return blocks_.front().num_heads();
}

std::size_t GPT::num_layers() const noexcept {
    return blocks_.size();
}

std::size_t GPT::mlp_hidden_dim() const noexcept {
    return blocks_.front().mlp_hidden_dim();
}

std::size_t GPT::max_seq_len() const noexcept {
    return embedding_.max_seq_len();
}

float GPT::epsilon() const noexcept {
    return final_norm_.epsilon();
}

GPTParameterCount GPT::parameter_count() const {
    validate_parameters();
    GPTParameterCount count;
    count.embeddings = embedding_.token_embeddings().size() +
                       embedding_.positional_embeddings().size();
    for (const TransformerBlock& block : blocks_) {
        count.transformer_blocks += block.norm1().gamma().size() + block.norm1().beta().size() +
                                    block.norm2().gamma().size() + block.norm2().beta().size() +
                                    block.attention().q_weights().size() +
                                    block.attention().k_weights().size() +
                                    block.attention().v_weights().size() +
                                    block.attention().output_weights().size() +
                                    block.fc1_weights().size() + block.fc2_weights().size();
    }
    count.final_layer_norm = final_norm_.gamma().size() + final_norm_.beta().size();
    count.output_projection = output_weights_.size();
    return count;
}

Embedding& GPT::embedding() noexcept {
    return embedding_;
}

const Embedding& GPT::embedding() const noexcept {
    return embedding_;
}

std::vector<TransformerBlock>& GPT::blocks() noexcept {
    return blocks_;
}

const std::vector<TransformerBlock>& GPT::blocks() const noexcept {
    return blocks_;
}

LayerNorm& GPT::final_norm() noexcept {
    return final_norm_;
}

const LayerNorm& GPT::final_norm() const noexcept {
    return final_norm_;
}

Tensor& GPT::output_weights() noexcept {
    return output_weights_;
}

const Tensor& GPT::output_weights() const noexcept {
    return output_weights_;
}

Tensor GPT::forward_features(
    std::span<const TokenId> token_ids,
    std::size_t batch_size,
    std::size_t seq_len) const {
    Tensor hidden = embedding_.forward(token_ids, batch_size, seq_len);
    for (const TransformerBlock& block : blocks_) {
        hidden = block.forward(hidden);
    }
    return final_norm_.forward(hidden);
}

void GPT::validate_parameters() const {
    const std::size_t vocab_size = embedding_.vocab_size();
    const std::size_t embedding_dim = embedding_.embedding_dim();
    if (blocks_.empty() || embedding_.token_embeddings().shape() !=
            Tensor::Shape{vocab_size, embedding_dim} ||
        embedding_.positional_embeddings().shape() !=
            Tensor::Shape{embedding_.max_seq_len(), embedding_dim} ||
        final_norm_.gamma().shape() != Tensor::Shape{embedding_dim} ||
        final_norm_.beta().shape() != Tensor::Shape{embedding_dim} ||
        output_weights_.shape() != Tensor::Shape{embedding_dim, vocab_size}) {
        throw std::logic_error("GPT parameters have incompatible shapes");
    }

    const std::size_t hidden_dim = blocks_.front().mlp_hidden_dim();
    const std::size_t num_heads = blocks_.front().num_heads();
    for (const TransformerBlock& block : blocks_) {
        if (block.embedding_dim() != embedding_dim || block.num_heads() != num_heads ||
            block.mlp_hidden_dim() != hidden_dim ||
            block.norm1().gamma().shape() != Tensor::Shape{embedding_dim} ||
            block.norm1().beta().shape() != Tensor::Shape{embedding_dim} ||
            block.norm2().gamma().shape() != Tensor::Shape{embedding_dim} ||
            block.norm2().beta().shape() != Tensor::Shape{embedding_dim} ||
            block.attention().q_weights().shape() !=
                Tensor::Shape{embedding_dim, embedding_dim} ||
            block.attention().k_weights().shape() !=
                Tensor::Shape{embedding_dim, embedding_dim} ||
            block.attention().v_weights().shape() !=
                Tensor::Shape{embedding_dim, embedding_dim} ||
            block.attention().output_weights().shape() !=
                Tensor::Shape{embedding_dim, embedding_dim} ||
            block.fc1_weights().shape() != Tensor::Shape{embedding_dim, hidden_dim} ||
            block.fc2_weights().shape() != Tensor::Shape{hidden_dim, embedding_dim}) {
            throw std::logic_error("GPT Transformer Block parameters are incompatible");
        }
    }
}

}  // namespace mini_gpt

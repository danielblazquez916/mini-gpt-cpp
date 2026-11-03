#ifndef MINI_GPT_GPT_HPP
#define MINI_GPT_GPT_HPP

#include "mini_gpt/embedding.hpp"
#include "mini_gpt/layer_norm.hpp"
#include "mini_gpt/tensor.hpp"
#include "mini_gpt/transformer_block.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <span>
#include <vector>

namespace mini_gpt {

enum class SamplingStrategy {
    greedy,
    sampling,
};

enum class StopCondition {
    end_of_turn,
    end_of_conversation,
};

struct GenerationConfig {
    std::size_t max_new_tokens = 16;
    float temperature = 1.0F;
    SamplingStrategy strategy = SamplingStrategy::greedy;
    StopCondition stop_condition = StopCondition::end_of_conversation;
    std::uint32_t seed = 42;
};

struct GPTParameterCount {
    std::size_t embeddings = 0;
    std::size_t transformer_blocks = 0;
    std::size_t final_layer_norm = 0;
    std::size_t output_projection = 0;

    [[nodiscard]] std::size_t total() const noexcept {
        return embeddings + transformer_blocks + final_layer_norm + output_projection;
    }
};

class GPT {
public:
    using TokenId = std::uint32_t;

    GPT(
        std::size_t vocab_size,
        std::size_t embedding_dim = 256,
        std::size_t num_heads = 4,
        std::size_t num_layers = 4,
        std::size_t mlp_hidden_dim = 1024,
        std::size_t max_seq_len = 256,
        std::uint32_t seed = 42,
        float epsilon = 1.0e-5F);

    [[nodiscard]] Tensor forward(
        std::span<const TokenId> token_ids,
        std::size_t batch_size,
        std::size_t seq_len) const;

    // Devuelve solo los tokens nuevos, incluido el especial que detenga la generación.
    [[nodiscard]] std::vector<TokenId> generate(
        std::span<const TokenId> prompt,
        const GenerationConfig& config) const;

    [[nodiscard]] static std::vector<float> softmax(
        std::span<const float> logits,
        float temperature = 1.0F);
    [[nodiscard]] static TokenId select_token(
        std::span<const float> logits,
        float temperature,
        SamplingStrategy strategy,
        std::mt19937& generator);

    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static GPT load(const std::filesystem::path& path);

    [[nodiscard]] std::size_t vocab_size() const noexcept;
    [[nodiscard]] std::size_t embedding_dim() const noexcept;
    [[nodiscard]] std::size_t num_heads() const noexcept;
    [[nodiscard]] std::size_t num_layers() const noexcept;
    [[nodiscard]] std::size_t mlp_hidden_dim() const noexcept;
    [[nodiscard]] std::size_t max_seq_len() const noexcept;
    [[nodiscard]] float epsilon() const noexcept;
    [[nodiscard]] GPTParameterCount parameter_count() const;

    [[nodiscard]] Embedding& embedding() noexcept;
    [[nodiscard]] const Embedding& embedding() const noexcept;
    [[nodiscard]] std::vector<TransformerBlock>& blocks() noexcept;
    [[nodiscard]] const std::vector<TransformerBlock>& blocks() const noexcept;
    [[nodiscard]] LayerNorm& final_norm() noexcept;
    [[nodiscard]] const LayerNorm& final_norm() const noexcept;
    [[nodiscard]] Tensor& output_weights() noexcept;
    [[nodiscard]] const Tensor& output_weights() const noexcept;

private:
    [[nodiscard]] Tensor forward_features(
        std::span<const TokenId> token_ids,
        std::size_t batch_size,
        std::size_t seq_len) const;
    void validate_parameters() const;

    Embedding embedding_;
    std::vector<TransformerBlock> blocks_;
    LayerNorm final_norm_;
    Tensor output_weights_;
};

}  // namespace mini_gpt

#endif

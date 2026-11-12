#ifndef MINI_GPT_BACKPROPAGATION_HPP
#define MINI_GPT_BACKPROPAGATION_HPP

#include "mini_gpt/causal_self_attention.hpp"
#include "mini_gpt/embedding.hpp"
#include "mini_gpt/layer_norm.hpp"
#include "mini_gpt/tensor.hpp"
#include "mini_gpt/transformer_block.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mini_gpt {

struct LinearBackwardResult {
    Tensor input_gradient = Tensor::zeros({0});
    Tensor weight_gradient = Tensor::zeros({0});
};

struct LayerNormGradients {
    Tensor gamma = Tensor::zeros({0});
    Tensor beta = Tensor::zeros({0});
};

struct LayerNormBackwardResult {
    Tensor input_gradient = Tensor::zeros({0});
    LayerNormGradients parameters;
};

struct EmbeddingGradients {
    Tensor token_embeddings = Tensor::zeros({0});
    Tensor positional_embeddings = Tensor::zeros({0});
};

struct AttentionGradients {
    Tensor q_weights = Tensor::zeros({0});
    Tensor k_weights = Tensor::zeros({0});
    Tensor v_weights = Tensor::zeros({0});
    Tensor output_weights = Tensor::zeros({0});
};

struct AttentionBackwardResult {
    Tensor input_gradient = Tensor::zeros({0});
    AttentionGradients parameters;
};

struct TransformerBlockGradients {
    LayerNormGradients norm1;
    LayerNormGradients norm2;
    AttentionGradients attention;
    Tensor fc1_weights = Tensor::zeros({0});
    Tensor fc2_weights = Tensor::zeros({0});
};

struct TransformerBlockBackwardResult {
    Tensor input_gradient = Tensor::zeros({0});
    TransformerBlockGradients parameters;
};

struct GPTGradients {
    EmbeddingGradients embedding;
    std::vector<TransformerBlockGradients> blocks;
    LayerNormGradients final_norm;
    Tensor output_weights = Tensor::zeros({0});
};

[[nodiscard]] LinearBackwardResult linear_backward(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& output_gradient);

[[nodiscard]] Tensor gelu_backward(const Tensor& input, const Tensor& output_gradient);

[[nodiscard]] LayerNormBackwardResult layer_norm_backward(
    const LayerNorm& layer_norm,
    const Tensor& input,
    const Tensor& output_gradient);

[[nodiscard]] EmbeddingGradients embedding_backward(
    const Embedding& embedding,
    std::span<const std::uint32_t> token_ids,
    std::size_t batch_size,
    std::size_t seq_len,
    const Tensor& output_gradient);

[[nodiscard]] AttentionBackwardResult attention_backward(
    const CausalSelfAttention& attention,
    const Tensor& input,
    const Tensor& output_gradient);

[[nodiscard]] TransformerBlockBackwardResult transformer_block_backward(
    const TransformerBlock& block,
    const Tensor& input,
    const Tensor& output_gradient);

}  // namespace mini_gpt

#endif

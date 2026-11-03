#ifndef MINI_GPT_TRANSFORMER_BLOCK_HPP
#define MINI_GPT_TRANSFORMER_BLOCK_HPP

#include "mini_gpt/causal_self_attention.hpp"
#include "mini_gpt/layer_norm.hpp"
#include "mini_gpt/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace mini_gpt {

// GELU exacta: 0.5 * x * (1 + erf(x / sqrt(2))).
[[nodiscard]] Tensor gelu(const Tensor& input);

class TransformerBlock {
public:
    TransformerBlock(
        std::size_t embedding_dim = 256,
        std::size_t num_heads = 4,
        std::size_t mlp_hidden_dim = 1024,
        std::uint32_t seed = 42,
        float epsilon = 1.0e-5F);

    [[nodiscard]] Tensor forward(const Tensor& input) const;

    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static TransformerBlock load(const std::filesystem::path& path);

    [[nodiscard]] std::size_t embedding_dim() const noexcept;
    [[nodiscard]] std::size_t num_heads() const noexcept;
    [[nodiscard]] std::size_t mlp_hidden_dim() const noexcept;
    [[nodiscard]] float epsilon() const noexcept;

    [[nodiscard]] CausalSelfAttention& attention() noexcept;
    [[nodiscard]] const CausalSelfAttention& attention() const noexcept;
    [[nodiscard]] LayerNorm& norm1() noexcept;
    [[nodiscard]] const LayerNorm& norm1() const noexcept;
    [[nodiscard]] LayerNorm& norm2() noexcept;
    [[nodiscard]] const LayerNorm& norm2() const noexcept;
    [[nodiscard]] Tensor& fc1_weights() noexcept;
    [[nodiscard]] const Tensor& fc1_weights() const noexcept;
    [[nodiscard]] Tensor& fc2_weights() noexcept;
    [[nodiscard]] const Tensor& fc2_weights() const noexcept;

private:
    void validate_parameters() const;

    std::size_t mlp_hidden_dim_;
    CausalSelfAttention attention_;
    LayerNorm norm1_;
    LayerNorm norm2_;
    Tensor fc1_weights_;
    Tensor fc2_weights_;
};

}  // namespace mini_gpt

#endif

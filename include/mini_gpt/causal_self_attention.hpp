#ifndef MINI_GPT_CAUSAL_SELF_ATTENTION_HPP
#define MINI_GPT_CAUSAL_SELF_ATTENTION_HPP

#include "mini_gpt/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace mini_gpt {

struct AttentionForwardResult {
    Tensor output;
    Tensor context;
    Tensor queries;
    Tensor keys;
    Tensor values;
    Tensor logits;
    Tensor probabilities;
};

class CausalSelfAttention {
public:
    CausalSelfAttention(
        std::size_t embedding_dim = 256,
        std::size_t num_heads = 4,
        std::uint32_t seed = 42);

    [[nodiscard]] Tensor forward(const Tensor& input) const;
    [[nodiscard]] AttentionForwardResult forward_with_details(const Tensor& input) const;
    [[nodiscard]] AttentionForwardResult forward_from_qkv(
        const Tensor& queries,
        const Tensor& keys,
        const Tensor& values) const;

    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static CausalSelfAttention load(const std::filesystem::path& path);

    [[nodiscard]] std::size_t embedding_dim() const noexcept;
    [[nodiscard]] std::size_t num_heads() const noexcept;
    [[nodiscard]] std::size_t head_dim() const noexcept;

    [[nodiscard]] Tensor& q_weights() noexcept;
    [[nodiscard]] const Tensor& q_weights() const noexcept;
    [[nodiscard]] Tensor& k_weights() noexcept;
    [[nodiscard]] const Tensor& k_weights() const noexcept;
    [[nodiscard]] Tensor& v_weights() noexcept;
    [[nodiscard]] const Tensor& v_weights() const noexcept;
    [[nodiscard]] Tensor& output_weights() noexcept;
    [[nodiscard]] const Tensor& output_weights() const noexcept;

private:
    CausalSelfAttention(
        std::size_t num_heads,
        Tensor q_weights,
        Tensor k_weights,
        Tensor v_weights,
        Tensor output_weights);

    [[nodiscard]] AttentionForwardResult attend(
        Tensor queries,
        Tensor keys,
        Tensor values) const;
    void validate_parameter_shapes() const;

    std::size_t embedding_dim_;
    std::size_t num_heads_;
    std::size_t head_dim_;
    Tensor q_weights_;
    Tensor k_weights_;
    Tensor v_weights_;
    Tensor output_weights_;
};

}  // namespace mini_gpt

#endif

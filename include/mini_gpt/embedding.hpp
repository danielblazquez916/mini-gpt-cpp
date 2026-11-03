#ifndef MINI_GPT_EMBEDDING_HPP
#define MINI_GPT_EMBEDDING_HPP

#include "mini_gpt/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace mini_gpt {

class Embedding {
public:
    using TokenId = std::uint32_t;

    Embedding(
        std::size_t vocab_size,
        std::size_t embedding_dim = 256,
        std::size_t max_seq_len = 1024,
        std::uint32_t seed = 42);

    [[nodiscard]] Tensor token_forward(
        std::span<const TokenId> token_ids,
        std::size_t batch_size,
        std::size_t seq_len) const;

    [[nodiscard]] Tensor forward(
        std::span<const TokenId> token_ids,
        std::size_t batch_size,
        std::size_t seq_len) const;

    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static Embedding load(const std::filesystem::path& path);

    [[nodiscard]] std::size_t vocab_size() const noexcept;
    [[nodiscard]] std::size_t embedding_dim() const noexcept;
    [[nodiscard]] std::size_t max_seq_len() const noexcept;

    [[nodiscard]] Tensor& token_embeddings() noexcept;
    [[nodiscard]] const Tensor& token_embeddings() const noexcept;
    [[nodiscard]] Tensor& positional_embeddings() noexcept;
    [[nodiscard]] const Tensor& positional_embeddings() const noexcept;

private:
    Embedding(Tensor token_embeddings, Tensor positional_embeddings);

    void validate_shapes() const;

    std::size_t vocab_size_;
    std::size_t embedding_dim_;
    std::size_t max_seq_len_;
    Tensor token_embeddings_;
    Tensor positional_embeddings_;
};

}  // namespace mini_gpt

#endif

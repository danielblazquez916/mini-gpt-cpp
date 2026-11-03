#ifndef MINI_GPT_CROSS_ENTROPY_HPP
#define MINI_GPT_CROSS_ENTROPY_HPP

#include "mini_gpt/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mini_gpt {

struct ShiftedTokens {
    std::vector<std::uint32_t> inputs;
    std::vector<std::uint32_t> targets;
};

[[nodiscard]] ShiftedTokens shift_tokens(std::span<const std::uint32_t> tokens);

[[nodiscard]] float cross_entropy(
    const Tensor& logits,
    std::span<const std::uint32_t> targets,
    std::span<const std::uint8_t> mask = {});

[[nodiscard]] float perplexity(float loss);

}  // namespace mini_gpt

#endif

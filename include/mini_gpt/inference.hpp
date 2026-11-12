#ifndef MINI_GPT_INFERENCE_HPP
#define MINI_GPT_INFERENCE_HPP

#include "mini_gpt/gpt.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mini_gpt {

struct TokenProbability {
    std::uint32_t id = 0;
    float probability = 0.0F;
    std::size_t rank = 0;
};

struct ContinuationEvaluation {
    double loss = 0.0;
    std::vector<TokenProbability> tokens;
};

[[nodiscard]] std::vector<TokenProbability> top_token_probabilities(
    const GPT& model,
    std::span<const std::uint32_t> context,
    std::size_t count);

[[nodiscard]] ContinuationEvaluation evaluate_continuation(
    const GPT& model,
    std::span<const std::uint32_t> prompt,
    std::span<const std::uint32_t> expected);

}  // namespace mini_gpt

#endif

#include "mini_gpt/cross_entropy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mini_gpt {

ShiftedTokens shift_tokens(std::span<const std::uint32_t> tokens) {
    if (tokens.size() < 2) {
        throw std::invalid_argument("At least two tokens are required for a shifted pair");
    }

    ShiftedTokens shifted;
    shifted.inputs.reserve(tokens.size() - 1);
    shifted.targets.reserve(tokens.size() - 1);
    shifted.inputs.insert(shifted.inputs.end(), tokens.begin(), tokens.end() - 1);
    shifted.targets.insert(shifted.targets.end(), tokens.begin() + 1, tokens.end());
    return shifted;
}

float cross_entropy(
    const Tensor& logits,
    std::span<const std::uint32_t> targets,
    std::span<const std::uint8_t> mask) {
    if (logits.rank() != 3) {
        throw std::invalid_argument("Logits must have shape [batch_size, seq_len, vocab_size]");
    }

    const std::size_t batch_size = logits.shape()[0];
    const std::size_t seq_len = logits.shape()[1];
    const std::size_t vocab_size = logits.shape()[2];
    const std::size_t positions = batch_size * seq_len;
    if (vocab_size == 0) {
        throw std::invalid_argument("Vocabulary size must be greater than zero");
    }
    if (targets.size() != positions) {
        throw std::invalid_argument("Target count must equal batch_size * seq_len");
    }
    if (!mask.empty() && mask.size() != positions) {
        throw std::invalid_argument("Mask size must equal batch_size * seq_len");
    }
    for (const std::uint8_t value : mask) {
        if (value > 1) {
            throw std::invalid_argument("Mask values must be either zero or one");
        }
    }

    const float* logit_data = logits.data().data();
    double total_loss = 0.0;
    std::size_t valid_positions = 0;
    for (std::size_t position = 0; position < positions; ++position) {
        if (!mask.empty() && mask[position] == 0) {
            continue;
        }
        if (targets[position] >= vocab_size) {
            throw std::out_of_range("Target token ID is outside the vocabulary");
        }

        const std::size_t offset = position * vocab_size;
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t token = 0; token < vocab_size; ++token) {
            const float value = logit_data[offset + token];
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Logits must be finite at valid positions");
            }
            maximum = std::max(maximum, static_cast<double>(value));
        }

        double exponential_sum = 0.0;
        for (std::size_t token = 0; token < vocab_size; ++token) {
            exponential_sum += std::exp(static_cast<double>(logit_data[offset + token]) - maximum);
        }
        const double log_sum_exp = maximum + std::log(exponential_sum);
        total_loss += log_sum_exp - static_cast<double>(logit_data[offset + targets[position]]);
        ++valid_positions;
    }

    if (valid_positions == 0) {
        throw std::invalid_argument("Cross-Entropy requires at least one valid position");
    }
    return static_cast<float>(total_loss / static_cast<double>(valid_positions));
}

float perplexity(float loss) {
    if (!std::isfinite(loss) || loss < 0.0F) {
        throw std::invalid_argument("Perplexity requires a finite, non-negative loss");
    }
    return std::exp(loss);
}

}  // namespace mini_gpt

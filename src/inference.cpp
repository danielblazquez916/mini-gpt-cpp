#include "mini_gpt/inference.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace mini_gpt {
namespace {

std::vector<TokenProbability> ranked_probabilities(
    const GPT& model,
    std::span<const std::uint32_t> context) {
    if (context.empty()) {
        throw std::invalid_argument("Probability analysis requires a non-empty context");
    }

    const Tensor logits = model.forward(context, 1, context.size());
    const auto last_logits = logits.data().subspan(
        (context.size() - 1) * model.vocab_size(), model.vocab_size());
    const auto probabilities = GPT::softmax(last_logits, 1.0F);
    std::vector<TokenProbability> ranked;
    ranked.reserve(probabilities.size());
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        ranked.push_back(TokenProbability{
            static_cast<std::uint32_t>(i), probabilities[i], 0});
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.probability != rhs.probability) {
            return lhs.probability > rhs.probability;
        }
        return lhs.id < rhs.id;
    });
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        ranked[i].rank = i + 1;
    }
    return ranked;
}

}  // namespace

std::vector<TokenProbability> top_token_probabilities(
    const GPT& model,
    std::span<const std::uint32_t> context,
    std::size_t count) {
    auto ranked = ranked_probabilities(model, context);
    if (count == 0 || count > ranked.size()) {
        throw std::invalid_argument("Top count must be between 1 and vocabulary size");
    }
    ranked.resize(count);
    return ranked;
}

ContinuationEvaluation evaluate_continuation(
    const GPT& model,
    std::span<const std::uint32_t> prompt,
    std::span<const std::uint32_t> expected) {
    if (prompt.empty() || expected.empty()) {
        throw std::invalid_argument("Continuation evaluation requires prompt and expected text");
    }

    std::vector<std::uint32_t> sequence;
    sequence.reserve(prompt.size() + expected.size());
    sequence.insert(sequence.end(), prompt.begin(), prompt.end());
    sequence.insert(sequence.end(), expected.begin(), expected.end());

    ContinuationEvaluation evaluation;
    evaluation.tokens.reserve(expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const std::size_t context_size = prompt.size() + index;
        const std::size_t start = context_size > model.max_seq_len()
            ? context_size - model.max_seq_len()
            : 0;
        const auto context = std::span<const std::uint32_t>(
            sequence.data() + start, context_size - start);
        const auto ranked = ranked_probabilities(model, context);
        const std::uint32_t target = expected[index];
        if (target >= model.vocab_size()) {
            throw std::out_of_range("Expected token ID is outside the vocabulary");
        }
        const auto target_it = std::find_if(
            ranked.begin(), ranked.end(), [&](const auto& item) { return item.id == target; });
        const float probability = target_it->probability;
        evaluation.tokens.push_back(TokenProbability{target, probability, target_it->rank});
        evaluation.loss -= std::log(std::max<double>(probability, 1.0e-300));
    }
    evaluation.loss /= expected.size();
    return evaluation;
}

}  // namespace mini_gpt

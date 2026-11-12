#include "mini_gpt/backpropagation.hpp"

#include "mini_gpt/cross_entropy.hpp"
#include "mini_gpt/gpt.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mini_gpt {

LinearBackwardResult linear_backward(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& output_gradient) {
    if (input.rank() != 3 || weights.rank() != 2 ||
        input.shape()[2] != weights.shape()[0] ||
        output_gradient.shape() !=
            Tensor::Shape{input.shape()[0], input.shape()[1], weights.shape()[1]}) {
        throw std::invalid_argument("Incompatible shapes in linear backward");
    }

    const std::size_t batch_size = input.shape()[0];
    const std::size_t seq_len = input.shape()[1];
    const std::size_t input_dim = weights.shape()[0];
    const std::size_t output_dim = weights.shape()[1];
    const std::size_t rows = batch_size * seq_len;

    LinearBackwardResult result{
        Tensor::zeros(input.shape()), Tensor::zeros(weights.shape())};
    const float* input_data = input.data().data();
    const float* weight_data = weights.data().data();
    const float* gradient_data = output_gradient.data().data();
    float* input_gradient = result.input_gradient.data().data();
    float* weight_gradient = result.weight_gradient.data().data();

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t inner = 0; inner < input_dim; ++inner) {
            for (std::size_t output = 0; output < output_dim; ++output) {
                const float gradient = gradient_data[row * output_dim + output];
                weight_gradient[inner * output_dim + output] +=
                    input_data[row * input_dim + inner] * gradient;
                input_gradient[row * input_dim + inner] +=
                    gradient * weight_data[inner * output_dim + output];
            }
        }
    }
    return result;
}

Tensor gelu_backward(const Tensor& input, const Tensor& output_gradient) {
    if (input.shape() != output_gradient.shape()) {
        throw std::invalid_argument("GELU backward requires matching shapes");
    }

    Tensor input_gradient = Tensor::zeros(input.shape());
    for (std::size_t i = 0; i < input.size(); ++i) {
        const double value = input.data()[i];
        const double cdf = 0.5 * (1.0 + std::erf(value / std::sqrt(2.0)));
        const double pdf = 0.3989422804014327 * std::exp(-0.5 * value * value);
        input_gradient.data()[i] = static_cast<float>(
            static_cast<double>(output_gradient.data()[i]) * (cdf + value * pdf));
    }
    return input_gradient;
}

LayerNormBackwardResult layer_norm_backward(
    const LayerNorm& layer_norm,
    const Tensor& input,
    const Tensor& output_gradient) {
    const std::size_t dimension = layer_norm.normalized_dim();
    if (input.rank() == 0 || input.shape().back() != dimension ||
        output_gradient.shape() != input.shape() ||
        layer_norm.gamma().shape() != Tensor::Shape{dimension} ||
        layer_norm.beta().shape() != Tensor::Shape{dimension}) {
        throw std::invalid_argument("Incompatible shapes in LayerNorm backward");
    }

    LayerNormBackwardResult result{
        Tensor::zeros(input.shape()),
        LayerNormGradients{Tensor::zeros({dimension}), Tensor::zeros({dimension})}};
    const std::size_t rows = input.size() / dimension;
    std::vector<double> normalized(dimension);
    std::vector<double> normalized_gradient(dimension);

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * dimension;
        double sum = 0.0;
        for (std::size_t i = 0; i < dimension; ++i) {
            sum += input.data()[offset + i];
        }
        const double mean = sum / static_cast<double>(dimension);
        double squared_differences = 0.0;
        for (std::size_t i = 0; i < dimension; ++i) {
            const double difference = input.data()[offset + i] - mean;
            squared_differences += difference * difference;
        }
        const double variance = squared_differences / static_cast<double>(dimension);
        const double inverse_standard_deviation =
            1.0 / std::sqrt(variance + layer_norm.epsilon());

        double gradient_sum = 0.0;
        double gradient_normalized_sum = 0.0;
        for (std::size_t i = 0; i < dimension; ++i) {
            normalized[i] =
                (input.data()[offset + i] - mean) * inverse_standard_deviation;
            const double gradient = output_gradient.data()[offset + i];
            result.parameters.gamma.data()[i] += gradient * normalized[i];
            result.parameters.beta.data()[i] += gradient;
            normalized_gradient[i] = gradient * layer_norm.gamma().data()[i];
            gradient_sum += normalized_gradient[i];
            gradient_normalized_sum += normalized_gradient[i] * normalized[i];
        }

        for (std::size_t i = 0; i < dimension; ++i) {
            const double value = inverse_standard_deviation / dimension *
                (dimension * normalized_gradient[i] - gradient_sum -
                 normalized[i] * gradient_normalized_sum);
            result.input_gradient.data()[offset + i] = static_cast<float>(value);
        }
    }
    return result;
}

EmbeddingGradients embedding_backward(
    const Embedding& embedding,
    std::span<const std::uint32_t> token_ids,
    std::size_t batch_size,
    std::size_t seq_len,
    const Tensor& output_gradient) {
    const std::size_t dimension = embedding.embedding_dim();
    if (seq_len > embedding.max_seq_len() ||
        (batch_size != 0 && seq_len > std::numeric_limits<std::size_t>::max() / batch_size) ||
        token_ids.size() != batch_size * seq_len ||
        output_gradient.shape() != Tensor::Shape{batch_size, seq_len, dimension}) {
        throw std::invalid_argument("Incompatible shapes in embedding backward");
    }

    EmbeddingGradients gradients{
        Tensor::zeros(embedding.token_embeddings().shape()),
        Tensor::zeros(embedding.positional_embeddings().shape())};
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < seq_len; ++position) {
            const std::size_t index = batch * seq_len + position;
            const std::uint32_t id = token_ids[index];
            if (id >= embedding.vocab_size()) {
                throw std::out_of_range("Token ID is outside the embedding vocabulary");
            }
            for (std::size_t i = 0; i < dimension; ++i) {
                const float gradient = output_gradient.data()[index * dimension + i];
                gradients.token_embeddings.data()[id * dimension + i] += gradient;
                gradients.positional_embeddings.data()[position * dimension + i] += gradient;
            }
        }
    }
    return gradients;
}

AttentionBackwardResult attention_backward(
    const CausalSelfAttention& attention,
    const Tensor& input,
    const Tensor& output_gradient) {
    if (input.rank() != 3 || input.shape()[2] != attention.embedding_dim() ||
        output_gradient.shape() != input.shape()) {
        throw std::invalid_argument("Incompatible shapes in attention backward");
    }

    const auto forward = attention.forward_with_details(input);
    const auto output_backward =
        linear_backward(forward.context, attention.output_weights(), output_gradient);
    const std::size_t batch_size = input.shape()[0];
    const std::size_t seq_len = input.shape()[1];
    const std::size_t embedding_dim = attention.embedding_dim();
    const std::size_t num_heads = attention.num_heads();
    const std::size_t head_dim = attention.head_dim();

    Tensor queries_gradient = Tensor::zeros(forward.queries.shape());
    Tensor keys_gradient = Tensor::zeros(forward.keys.shape());
    Tensor values_gradient = Tensor::zeros(forward.values.shape());
    std::vector<double> probability_gradients(seq_len);
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t head = 0; head < num_heads; ++head) {
            for (std::size_t query = 0; query < seq_len; ++query) {
                std::fill(probability_gradients.begin(), probability_gradients.end(), 0.0);
                const std::size_t query_offset =
                    ((batch * num_heads + head) * seq_len + query) * head_dim;
                const std::size_t probability_offset =
                    ((batch * num_heads + head) * seq_len + query) * seq_len;
                const std::size_t context_offset =
                    (batch * seq_len + query) * embedding_dim + head * head_dim;

                for (std::size_t key = 0; key <= query; ++key) {
                    const std::size_t value_offset =
                        ((batch * num_heads + head) * seq_len + key) * head_dim;
                    const float probability = forward.probabilities.data()[probability_offset + key];
                    for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                        const float context_gradient =
                            output_backward.input_gradient.data()[context_offset + dimension];
                        const float value = forward.values.data()[value_offset + dimension];
                        values_gradient.data()[value_offset + dimension] +=
                            probability * context_gradient;
                        probability_gradients[key] += context_gradient * value;
                    }
                }

                double weighted_probability_gradient = 0.0;
                for (std::size_t key = 0; key <= query; ++key) {
                    weighted_probability_gradient += probability_gradients[key] *
                        forward.probabilities.data()[probability_offset + key];
                }
                for (std::size_t key = 0; key <= query; ++key) {
                    const float probability = forward.probabilities.data()[probability_offset + key];
                    const double score_gradient = probability *
                        (probability_gradients[key] - weighted_probability_gradient) * scale;
                    const std::size_t key_offset =
                        ((batch * num_heads + head) * seq_len + key) * head_dim;
                    for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                        queries_gradient.data()[query_offset + dimension] +=
                            score_gradient * forward.keys.data()[key_offset + dimension];
                        keys_gradient.data()[key_offset + dimension] +=
                            score_gradient * forward.queries.data()[query_offset + dimension];
                    }
                }
            }
        }
    }

    Tensor flat_queries_gradient = Tensor::zeros(input.shape());
    Tensor flat_keys_gradient = Tensor::zeros(input.shape());
    Tensor flat_values_gradient = Tensor::zeros(input.shape());
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t head = 0; head < num_heads; ++head) {
            for (std::size_t position = 0; position < seq_len; ++position) {
                for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                    const std::size_t head_offset =
                        ((batch * num_heads + head) * seq_len + position) * head_dim + dimension;
                    const std::size_t flat_offset =
                        (batch * seq_len + position) * embedding_dim + head * head_dim + dimension;
                    flat_queries_gradient.data()[flat_offset] =
                        queries_gradient.data()[head_offset];
                    flat_keys_gradient.data()[flat_offset] = keys_gradient.data()[head_offset];
                    flat_values_gradient.data()[flat_offset] = values_gradient.data()[head_offset];
                }
            }
        }
    }

    auto query_backward = linear_backward(input, attention.q_weights(), flat_queries_gradient);
    auto key_backward = linear_backward(input, attention.k_weights(), flat_keys_gradient);
    auto value_backward = linear_backward(input, attention.v_weights(), flat_values_gradient);
    Tensor input_gradient = query_backward.input_gradient + key_backward.input_gradient;
    input_gradient = input_gradient + value_backward.input_gradient;

    return AttentionBackwardResult{
        std::move(input_gradient),
        AttentionGradients{
            std::move(query_backward.weight_gradient),
            std::move(key_backward.weight_gradient),
            std::move(value_backward.weight_gradient),
            std::move(output_backward.weight_gradient)}};
}

namespace {

Tensor linear_forward(const Tensor& input, const Tensor& weights) {
    const std::size_t batch_size = input.shape()[0];
    const std::size_t seq_len = input.shape()[1];
    const std::size_t input_dim = input.shape()[2];
    const std::size_t output_dim = weights.shape()[1];
    const std::size_t rows = batch_size * seq_len;
    Tensor output = Tensor::zeros({batch_size, seq_len, output_dim});

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t inner = 0; inner < input_dim; ++inner) {
            const float input_value = input.data()[row * input_dim + inner];
            for (std::size_t output_index = 0; output_index < output_dim; ++output_index) {
                output.data()[row * output_dim + output_index] += input_value *
                    weights.data()[inner * output_dim + output_index];
            }
        }
    }
    return output;
}

void accumulate(Tensor& target, const Tensor& source) {
    if (target.shape() != source.shape()) {
        throw std::logic_error("Gradient shapes do not match during accumulation");
    }
    for (std::size_t i = 0; i < target.size(); ++i) {
        target.data()[i] += source.data()[i];
    }
}

void accumulate(LayerNormGradients& target, const LayerNormGradients& source) {
    accumulate(target.gamma, source.gamma);
    accumulate(target.beta, source.beta);
}

void accumulate(AttentionGradients& target, const AttentionGradients& source) {
    accumulate(target.q_weights, source.q_weights);
    accumulate(target.k_weights, source.k_weights);
    accumulate(target.v_weights, source.v_weights);
    accumulate(target.output_weights, source.output_weights);
}

void accumulate(TransformerBlockGradients& target, const TransformerBlockGradients& source) {
    accumulate(target.norm1, source.norm1);
    accumulate(target.norm2, source.norm2);
    accumulate(target.attention, source.attention);
    accumulate(target.fc1_weights, source.fc1_weights);
    accumulate(target.fc2_weights, source.fc2_weights);
}

}  // namespace

TransformerBlockBackwardResult transformer_block_backward(
    const TransformerBlock& block,
    const Tensor& input,
    const Tensor& output_gradient) {
    if (input.rank() != 3 || input.shape()[2] != block.embedding_dim() ||
        output_gradient.shape() != input.shape()) {
        throw std::invalid_argument("Incompatible shapes in Transformer Block backward");
    }

    const Tensor normalized_attention = block.norm1().forward(input);
    const Tensor residual = input + block.attention().forward(normalized_attention);
    const Tensor normalized_mlp = block.norm2().forward(residual);
    const Tensor hidden = linear_forward(normalized_mlp, block.fc1_weights());
    const Tensor activated = gelu(hidden);

    auto fc2_backward = linear_backward(activated, block.fc2_weights(), output_gradient);
    const Tensor hidden_gradient = gelu_backward(hidden, fc2_backward.input_gradient);
    auto fc1_backward = linear_backward(normalized_mlp, block.fc1_weights(), hidden_gradient);
    auto norm2_backward =
        layer_norm_backward(block.norm2(), residual, fc1_backward.input_gradient);
    const Tensor residual_gradient = output_gradient + norm2_backward.input_gradient;

    auto attention_backward_result =
        attention_backward(block.attention(), normalized_attention, residual_gradient);
    auto norm1_backward =
        layer_norm_backward(block.norm1(), input, attention_backward_result.input_gradient);
    const Tensor input_gradient = residual_gradient + norm1_backward.input_gradient;

    return TransformerBlockBackwardResult{
        input_gradient,
        TransformerBlockGradients{
            std::move(norm1_backward.parameters),
            std::move(norm2_backward.parameters),
            std::move(attention_backward_result.parameters),
            std::move(fc1_backward.weight_gradient),
            std::move(fc2_backward.weight_gradient)}};
}

float GPT::backward(
    std::span<const TokenId> token_ids,
    std::size_t batch_size,
    std::size_t seq_len,
    std::span<const std::uint32_t> targets,
    std::span<const std::uint8_t> mask) {
    validate_parameters();
    std::vector<Tensor> block_inputs;
    block_inputs.reserve(blocks_.size());
    Tensor hidden = embedding_.forward(token_ids, batch_size, seq_len);
    for (const TransformerBlock& block : blocks_) {
        block_inputs.push_back(hidden);
        hidden = block.forward(hidden);
    }

    const Tensor final_input = hidden;
    const Tensor features = final_norm_.forward(final_input);
    const Tensor logits = linear_forward(features, output_weights_);
    const float loss = cross_entropy(logits, targets, mask);
    const Tensor logits_gradient = cross_entropy_backward(logits, targets, mask);

    auto output_backward = linear_backward(features, output_weights_, logits_gradient);
    accumulate(gradients_.output_weights, output_backward.weight_gradient);
    auto final_norm_backward =
        layer_norm_backward(final_norm_, final_input, output_backward.input_gradient);
    accumulate(gradients_.final_norm, final_norm_backward.parameters);
    hidden = final_norm_backward.input_gradient;

    for (std::size_t layer = blocks_.size(); layer-- > 0;) {
        auto block_backward = transformer_block_backward(
            blocks_[layer], block_inputs[layer], hidden);
        accumulate(gradients_.blocks[layer], block_backward.parameters);
        hidden = std::move(block_backward.input_gradient);
    }

    const auto embedding_gradients =
        embedding_backward(embedding_, token_ids, batch_size, seq_len, hidden);
    accumulate(gradients_.embedding.token_embeddings, embedding_gradients.token_embeddings);
    accumulate(
        gradients_.embedding.positional_embeddings,
        embedding_gradients.positional_embeddings);
    return loss;
}

}  // namespace mini_gpt

#ifndef MINI_GPT_CUDA_BACKWARD_CUH
#define MINI_GPT_CUDA_BACKWARD_CUH

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace mini_gpt::cuda {

float cross_entropy(
    const float* logits,
    const std::uint32_t* targets,
    const std::uint8_t* mask,
    float* output_gradient,
    std::size_t batch_size,
    std::size_t seq_len,
    std::size_t vocab_size,
    std::size_t valid_positions);

void linear_backward(
    const float* input,
    const float* weights,
    const float* output_gradient,
    float* input_gradient,
    float* weight_gradient,
    std::size_t rows,
    std::size_t input_dim,
    std::size_t output_dim);

void embedding_backward(
    const std::uint32_t* ids,
    const float* output_gradient,
    float* token_gradient,
    float* positional_gradient,
    std::size_t batch_size,
    std::size_t seq_len,
    std::size_t embedding_dim,
    std::size_t vocab_size,
    std::size_t max_seq_len);

void layer_norm_backward(
    const float* input,
    const float* gamma,
    const float* output_gradient,
    float* input_gradient,
    float* gamma_gradient,
    float* beta_gradient,
    std::size_t rows,
    std::size_t dimension,
    float epsilon);

void gelu_backward(const float* input, const float* output_gradient, float* input_gradient, std::size_t size);

void causal_attention_backward(
    const float* queries,
    const float* keys,
    const float* values,
    const float* probabilities,
    const float* context_gradient,
    float* queries_gradient,
    float* keys_gradient,
    float* values_gradient,
    std::size_t batch_size,
    std::size_t num_heads,
    std::size_t seq_len,
    std::size_t head_dim);

void adamw_update(
    float* parameter,
    const float* gradient,
    float* first_moment,
    float* second_moment,
    std::size_t size,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    bool apply_weight_decay,
    std::size_t step);

[[nodiscard]] bool all_finite(const float* values, std::size_t size);
void scale(float* values, std::size_t size, float factor);
void add_to(float* target, const float* source, std::size_t size);
double gradient_global_norm(const std::vector<std::pair<const float*, std::size_t>>& gradients);

}  // namespace mini_gpt::cuda

#endif

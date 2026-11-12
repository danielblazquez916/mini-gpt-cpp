#ifndef MINI_GPT_CUDA_OPS_CUH
#define MINI_GPT_CUDA_OPS_CUH

#include <cstddef>
#include <cstdint>

namespace mini_gpt::cuda {

enum class ElementwiseOperation {
    add,
    subtract,
    multiply,
};

void elementwise(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t size,
    ElementwiseOperation operation);

void transpose(const float* input, float* output, std::size_t rows, std::size_t columns);

void linear(
    const float* input,
    const float* weights,
    float* output,
    std::size_t rows,
    std::size_t input_dim,
    std::size_t output_dim);

void embedding(
    const std::uint32_t* ids,
    const float* token_weights,
    const float* positional_weights,
    float* output,
    std::size_t batch_size,
    std::size_t seq_len,
    std::size_t embedding_dim,
    std::size_t vocab_size,
    std::size_t max_seq_len,
    bool include_positions);

void layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t rows,
    std::size_t dimension,
    float epsilon);

void gelu(const float* input, float* output, std::size_t size);
void zero(float* output, std::size_t size);

void split_heads(
    const float* input,
    float* output,
    std::size_t batch_size,
    std::size_t num_heads,
    std::size_t seq_len,
    std::size_t head_dim);

void flatten_heads(
    const float* input,
    float* output,
    std::size_t batch_size,
    std::size_t num_heads,
    std::size_t seq_len,
    std::size_t head_dim);

void causal_attention(
    const float* queries,
    const float* keys,
    const float* values,
    float* logits,
    float* probabilities,
    float* combined,
    std::size_t batch_size,
    std::size_t num_heads,
    std::size_t seq_len,
    std::size_t head_dim,
    bool return_logits);

}  // namespace mini_gpt::cuda

#endif

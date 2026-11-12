#include "mini_gpt/cuda/ops.cuh"

#include "mini_gpt/cuda/matmul.cuh"
#include "mini_gpt/cuda/memory.cuh"
#include "mini_gpt/cuda/registry.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace mini_gpt::cuda {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ != 0) {
            data_ = static_cast<T*>(allocate_device(count_ * sizeof(T)));
            clear_alias(data_);
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            clear_alias(data_);
        deallocate_device(data_, count_ * sizeof(T));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                clear_alias(data_);
        deallocate_device(data_, count_ * sizeof(T));
            }
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    [[nodiscard]] T* get() const noexcept {
        return static_cast<T*>(alias_for_buffer(data_));
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
void copy_to_device(T* destination, const T* source, std::size_t count) {
    if (count == 0) {
        return;
    }
    if (is_registered(source)) {
        set_alias(destination, device_for_host(source));
    } else {
        set_alias(destination, destination);
        check_cuda(
            cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyHostToDevice),
            "copy to device");
    }
}

template <typename T>
T* output_pointer(T* output, DeviceBuffer<T>& buffer) {
    if (is_registered(output)) {
        return static_cast<T*>(device_for_host(output));
    }
    return buffer.get();
}

template <typename T>
void copy_to_host(T* destination, const T* source, std::size_t count) {
    if (count == 0) {
        return;
    }
    if (is_registered(destination)) {
        mark_device_modified(destination);
    } else {
        check_cuda(
            cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyDeviceToHost),
            "copy to host");
    }
}

__global__ void elementwise_kernel(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t size,
    ElementwiseOperation operation) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    if (operation == ElementwiseOperation::add) {
        output[index] = lhs[index] + rhs[index];
    } else if (operation == ElementwiseOperation::subtract) {
        output[index] = lhs[index] - rhs[index];
    } else {
        output[index] = lhs[index] * rhs[index];
    }
}

__global__ void transpose_kernel(const float* input, float* output, std::size_t rows, std::size_t columns) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * columns) return;
    const std::size_t row = index % rows;
    const std::size_t column = index / rows;
    output[column * rows + row] = input[row * columns + column];
}

__global__ void head_layout_kernel(
    const float* input,
    float* output,
    std::size_t size,
    std::size_t num_heads,
    std::size_t seq_len,
    std::size_t head_dim,
    bool split) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    const std::size_t dimension = index % head_dim;
    const std::size_t position = (index / head_dim) % seq_len;
    const std::size_t head = (index / (head_dim * seq_len)) % num_heads;
    const std::size_t batch = index / (head_dim * seq_len * num_heads);
    const std::size_t flat_index =
        (batch * seq_len + position) * num_heads * head_dim + head * head_dim + dimension;
    if (split) {
        output[index] = input[flat_index];
    } else {
        output[flat_index] = input[index];
    }
}

__global__ void embedding_kernel(
    const std::uint32_t* ids,
    const float* token_weights,
    const float* positional_weights,
    float* output,
    std::size_t token_count,
    std::size_t seq_len,
    std::size_t embedding_dim,
    bool include_positions) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= token_count * embedding_dim) return;
    const std::size_t token_index = index / embedding_dim;
    const std::size_t position = token_index % seq_len;
    const std::size_t dimension = index % embedding_dim;
    const std::uint32_t id = ids[token_index];
    float value = token_weights[id * embedding_dim + dimension];
    if (include_positions) {
        value += positional_weights[position * embedding_dim + dimension];
    }
    output[index] = value;
}

__global__ void layer_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t rows,
    std::size_t dimension,
    float epsilon) {
    const std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const std::size_t offset = row * dimension;

    float mean = 0.0F;
    for (std::size_t i = 0; i < dimension; ++i) {
        mean += input[offset + i];
    }
    mean /= dimension;
    float variance = 0.0F;
    for (std::size_t i = 0; i < dimension; ++i) {
        const float difference = input[offset + i] - mean;
        variance += difference * difference;
    }
    variance /= dimension;
    const float inverse = rsqrtf(variance + epsilon);
    for (std::size_t i = 0; i < dimension; ++i) {
        output[offset + i] = (input[offset + i] - mean) * inverse * gamma[i] + beta[i];
    }
}

__global__ void gelu_kernel(const float* input, float* output, std::size_t size) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    const float value = input[index];
    output[index] = 0.5F * value * (1.0F + erff(value * 0.7071067811865475244F));
}

__global__ void causal_attention_kernel(
    const float* queries,
    const float* keys,
    const float* values,
    float* logits,
    float* probabilities,
    float* combined,
    std::size_t num_heads,
    std::size_t seq_len,
    std::size_t head_dim) {
    const std::size_t batch = blockIdx.x;
    const std::size_t head = blockIdx.y;
    const std::size_t query = blockIdx.z;
    if (query >= seq_len) return;
    const std::size_t embedding_dim = num_heads * head_dim;
    const std::size_t query_offset =
        ((batch * num_heads + head) * seq_len + query) * head_dim;
    const std::size_t probability_offset =
        ((batch * num_heads + head) * seq_len + query) * seq_len;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    float maximum = -CUDART_INF_F;
    for (std::size_t key = 0; key < seq_len; ++key) {
        if (key > query) {
            logits[probability_offset + key] = -CUDART_INF_F;
            probabilities[probability_offset + key] = 0.0F;
            continue;
        }
        const std::size_t key_offset =
            ((batch * num_heads + head) * seq_len + key) * head_dim;
        float score = 0.0F;
        for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
            score += queries[query_offset + dimension] * keys[key_offset + dimension];
        }
        score *= scale;
        logits[probability_offset + key] = score;
        maximum = fmaxf(maximum, score);
    }

    float denominator = 0.0F;
    for (std::size_t key = 0; key <= query; ++key) {
        const float weight = expf(logits[probability_offset + key] - maximum);
        probabilities[probability_offset + key] = weight;
        denominator += weight;
    }
    for (std::size_t key = 0; key <= query; ++key) {
        const float probability = probabilities[probability_offset + key] / denominator;
        probabilities[probability_offset + key] = probability;
        const std::size_t value_offset =
            ((batch * num_heads + head) * seq_len + key) * head_dim;
        const std::size_t output_offset =
            (batch * seq_len + query) * embedding_dim + head * head_dim;
        for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
            combined[output_offset + dimension] += probability * values[value_offset + dimension];
        }
    }
}

}  // namespace

void elementwise(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t size,
    ElementwiseOperation operation) {
    if (size == 0) return;
    DeviceBuffer<float> device_lhs(size), device_rhs(size), device_output(size);
    copy_to_device(device_lhs.get(), lhs, size);
    copy_to_device(device_rhs.get(), rhs, size);
    elementwise_kernel<<<(size + 255) / 256, 256>>>(
        device_lhs.get(), device_rhs.get(), output_pointer(output, device_output), size, operation);
    check_cuda(cudaGetLastError(), "elementwise kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize elementwise");
    copy_to_host(output, device_output.get(), size);
}

void transpose(const float* input, float* output, std::size_t rows, std::size_t columns) {
    const std::size_t size = rows * columns;
    if (size == 0) return;
    DeviceBuffer<float> device_input(size), device_output(size);
    copy_to_device(device_input.get(), input, size);
    transpose_kernel<<<(size + 255) / 256, 256>>>(
        device_input.get(), output_pointer(output, device_output), rows, columns);
    check_cuda(cudaGetLastError(), "transpose kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize transpose");
    copy_to_host(output, device_output.get(), size);
}

void linear(
    const float* input,
    const float* weights,
    float* output,
    std::size_t rows,
    std::size_t input_dim,
    std::size_t output_dim) {
    matmul(input, weights, output, rows, input_dim, output_dim);
}

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
    bool include_positions) {
    const std::size_t token_count = batch_size * seq_len;
    const std::size_t output_size = token_count * embedding_dim;
    if (output_size == 0) return;

    DeviceBuffer<std::uint32_t> device_ids(token_count);
    DeviceBuffer<float> device_tokens(vocab_size * embedding_dim);
    DeviceBuffer<float> device_positions(max_seq_len * embedding_dim);
    DeviceBuffer<float> device_output(output_size);
    copy_to_device(device_ids.get(), ids, token_count);
    copy_to_device(device_tokens.get(), token_weights, vocab_size * embedding_dim);
    if (include_positions) {
        copy_to_device(
            device_positions.get(), positional_weights, max_seq_len * embedding_dim);
    }
    embedding_kernel<<<(output_size + 255) / 256, 256>>>(
        device_ids.get(),
        device_tokens.get(),
        device_positions.get(),
        output_pointer(output, device_output),
        token_count,
        seq_len,
        embedding_dim,
        include_positions);
    check_cuda(cudaGetLastError(), "embedding kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize embedding");
    copy_to_host(output, device_output.get(), output_size);
}

void layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t rows,
    std::size_t dimension,
    float epsilon) {
    if (rows == 0) return;
    DeviceBuffer<float> device_input(rows * dimension), device_gamma(dimension),
        device_beta(dimension), device_output(rows * dimension);
    copy_to_device(device_input.get(), input, rows * dimension);
    copy_to_device(device_gamma.get(), gamma, dimension);
    copy_to_device(device_beta.get(), beta, dimension);
    layer_norm_kernel<<<(rows + 255) / 256, 256>>>(
        device_input.get(),
        device_gamma.get(),
        device_beta.get(),
        output_pointer(output, device_output),
        rows,
        dimension,
        epsilon);
    check_cuda(cudaGetLastError(), "LayerNorm kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize LayerNorm");
    copy_to_host(output, device_output.get(), rows * dimension);
}

void gelu(const float* input, float* output, std::size_t size) {
    if (size == 0) return;
    DeviceBuffer<float> device_input(size), device_output(size);
    copy_to_device(device_input.get(), input, size);
    gelu_kernel<<<(size + 255) / 256, 256>>>(
        device_input.get(), output_pointer(output, device_output), size);
    check_cuda(cudaGetLastError(), "GELU kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize GELU");
    copy_to_host(output, device_output.get(), size);
}

void zero(float* output, std::size_t size) {
    if (size == 0) return;
    DeviceBuffer<float> device_output(size);
    float* output_kernel = output_pointer(output, device_output);
    check_cuda(cudaMemset(output_kernel, 0, size * sizeof(float)), "clear tensor");
    copy_to_host(output, device_output.get(), size);
}

void rearrange_heads(
    const float* input,
    float* output,
    std::size_t batch_size,
    std::size_t num_heads,
    std::size_t seq_len,
    std::size_t head_dim,
    bool split) {
    const std::size_t size = batch_size * num_heads * seq_len * head_dim;
    if (size == 0) return;
    DeviceBuffer<float> device_input(size), device_output(size);
    copy_to_device(device_input.get(), input, size);
    head_layout_kernel<<<(size + 255) / 256, 256>>>(
        device_input.get(), output_pointer(output, device_output), size, num_heads, seq_len, head_dim, split);
    check_cuda(cudaGetLastError(), "head layout kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize head layout");
    copy_to_host(output, device_output.get(), size);
}

void split_heads(
    const float* input,
    float* output,
    std::size_t batch_size,
    std::size_t num_heads,
    std::size_t seq_len,
    std::size_t head_dim) {
    rearrange_heads(input, output, batch_size, num_heads, seq_len, head_dim, true);
}

void flatten_heads(
    const float* input,
    float* output,
    std::size_t batch_size,
    std::size_t num_heads,
    std::size_t seq_len,
    std::size_t head_dim) {
    rearrange_heads(input, output, batch_size, num_heads, seq_len, head_dim, false);
}

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
    bool return_logits) {
    if (batch_size == 0 || seq_len == 0) return;
    const std::size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    const std::size_t score_size = batch_size * num_heads * seq_len * seq_len;
    const std::size_t combined_size = batch_size * seq_len * num_heads * head_dim;
    DeviceBuffer<float> device_queries(qkv_size), device_keys(qkv_size), device_values(qkv_size);
    DeviceBuffer<float> device_logits(score_size), device_probabilities(score_size);
    DeviceBuffer<float> device_combined(combined_size);
    copy_to_device(device_queries.get(), queries, qkv_size);
    copy_to_device(device_keys.get(), keys, qkv_size);
    copy_to_device(device_values.get(), values, qkv_size);
    float* combined_kernel = output_pointer(combined, device_combined);
    check_cuda(cudaMemset(combined_kernel, 0, combined_size * sizeof(float)), "clear combined");

    dim3 blocks(
        static_cast<unsigned int>(batch_size),
        static_cast<unsigned int>(num_heads),
        static_cast<unsigned int>(seq_len));
    causal_attention_kernel<<<blocks, 1>>>(
        device_queries.get(),
        device_keys.get(),
        device_values.get(),
        output_pointer(logits, device_logits),
        output_pointer(probabilities, device_probabilities),
        combined_kernel,
        num_heads,
        seq_len,
        head_dim);
    check_cuda(cudaGetLastError(), "causal attention kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize causal attention");
    if (return_logits) {
        copy_to_host(logits, device_logits.get(), score_size);
    }
    copy_to_host(probabilities, device_probabilities.get(), score_size);
    copy_to_host(combined, device_combined.get(), combined_size);
}

}  // namespace mini_gpt::cuda

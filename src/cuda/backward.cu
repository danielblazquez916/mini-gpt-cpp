#include "mini_gpt/cuda/backward.cuh"
#include "mini_gpt/cuda/memory.cuh"
#include "mini_gpt/cuda/registry.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <limits>
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

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuBLAS status " + std::to_string(status));
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
        if (data_ != nullptr) clear_alias(data_);
        deallocate_device(data_, count_ * sizeof(T));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) clear_alias(data_);
        deallocate_device(data_, count_ * sizeof(T));
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    [[nodiscard]] T* get() const noexcept { return static_cast<T*>(alias_for_buffer(data_)); }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
T* output_pointer(T* output, DeviceBuffer<T>& buffer) {
    if (is_registered(output)) {
        return static_cast<T*>(device_for_host(output));
    }
    return buffer.get();
}

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

__global__ void cross_entropy_kernel(
    const float* logits,
    const std::uint32_t* targets,
    const std::uint8_t* mask,
    float* output_gradient,
    float* losses,
    std::size_t positions,
    std::size_t vocab_size,
    std::size_t valid_positions) {
    const std::size_t position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= positions || (mask != nullptr && mask[position] == 0)) return;
    const std::size_t offset = position * vocab_size;
    double maximum = -CUDART_INF_F;
    for (std::size_t token = 0; token < vocab_size; ++token) {
        const float logit = logits[offset + token];
        if (!isfinite(logit)) {
            losses[position] = CUDART_NAN_F;
            return;
        }
        maximum = fmax(maximum, static_cast<double>(logit));
    }
    double denominator = 0.0;
    for (std::size_t token = 0; token < vocab_size; ++token) {
        denominator += exp(static_cast<double>(logits[offset + token]) - maximum);
    }
    const double log_sum_exp = maximum + log(denominator);
    losses[position] = static_cast<float>(log_sum_exp - logits[offset + targets[position]]);
    if (output_gradient != nullptr) {
        const double scale = 1.0 / valid_positions;
        for (std::size_t token = 0; token < vocab_size; ++token) {
            const double probability =
                exp(static_cast<double>(logits[offset + token]) - maximum) / denominator;
            output_gradient[offset + token] = static_cast<float>(probability * scale);
        }
        output_gradient[offset + targets[position]] -= scale;
    }
}

__global__ void embedding_backward_kernel(
    const std::uint32_t* ids,
    const float* output_gradient,
    float* token_gradient,
    float* positional_gradient,
    std::size_t token_count,
    std::size_t seq_len,
    std::size_t embedding_dim) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= token_count * embedding_dim) return;
    const std::size_t token_index = index / embedding_dim;
    const std::size_t position = token_index % seq_len;
    const std::size_t dimension = index % embedding_dim;
    atomicAdd(&token_gradient[ids[token_index] * embedding_dim + dimension], output_gradient[index]);
    atomicAdd(&positional_gradient[position * embedding_dim + dimension], output_gradient[index]);
}

__global__ void layer_norm_backward_kernel(
    const float* input,
    const float* gamma,
    const float* output_gradient,
    float* input_gradient,
    float* gamma_gradient,
    float* beta_gradient,
    std::size_t rows,
    std::size_t dimension,
    float epsilon) {
    const std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const std::size_t offset = row * dimension;
    float mean = 0.0F;
    for (std::size_t i = 0; i < dimension; ++i) mean += input[offset + i];
    mean /= dimension;
    float variance = 0.0F;
    for (std::size_t i = 0; i < dimension; ++i) {
        const float difference = input[offset + i] - mean;
        variance += difference * difference;
    }
    variance /= dimension;
    const float inverse = rsqrtf(variance + epsilon);

    float gradient_sum = 0.0F;
    float gradient_normalized_sum = 0.0F;
    for (std::size_t i = 0; i < dimension; ++i) {
        const float normalized = (input[offset + i] - mean) * inverse;
        const float gradient = output_gradient[offset + i];
        atomicAdd(&gamma_gradient[i], gradient * normalized);
        atomicAdd(&beta_gradient[i], gradient);
        const float normalized_gradient = gradient * gamma[i];
        gradient_sum += normalized_gradient;
        gradient_normalized_sum += normalized_gradient * normalized;
        input_gradient[offset + i] = normalized_gradient;
    }
    for (std::size_t i = 0; i < dimension; ++i) {
        const float normalized = (input[offset + i] - mean) * inverse;
        input_gradient[offset + i] = inverse / dimension *
            (dimension * input_gradient[offset + i] - gradient_sum -
             normalized * gradient_normalized_sum);
    }
}

__global__ void gelu_backward_kernel(
    const float* input,
    const float* output_gradient,
    float* input_gradient,
    std::size_t size) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    const float value = input[index];
    const float cdf = 0.5F * (1.0F + erff(value * 0.7071067811865475244F));
    const float pdf = 0.3989422804014327F * expf(-0.5F * value * value);
    input_gradient[index] = output_gradient[index] * (cdf + value * pdf);
}

__global__ void causal_attention_backward_kernel(
    const float* queries,
    const float* keys,
    const float* values,
    const float* probabilities,
    const float* context_gradient,
    float* queries_gradient,
    float* keys_gradient,
    float* values_gradient,
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
    const std::size_t context_offset =
        (batch * seq_len + query) * embedding_dim + head * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    extern __shared__ float probability_gradients[];
    float weighted_probability_gradient = 0.0F;
    for (std::size_t key = 0; key <= query; ++key) {
        const std::size_t value_offset =
            ((batch * num_heads + head) * seq_len + key) * head_dim;
        float probability_gradient = 0.0F;
        for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
            const float gradient = context_gradient[context_offset + dimension];
            const float value = values[value_offset + dimension];
            atomicAdd(
                &values_gradient[value_offset + dimension],
                probabilities[probability_offset + key] * gradient);
            probability_gradient += gradient * value;
        }
        probability_gradients[key] = probability_gradient;
        weighted_probability_gradient += probability_gradient * probabilities[probability_offset + key];
    }
    for (std::size_t key = 0; key <= query; ++key) {
        const std::size_t key_offset =
            ((batch * num_heads + head) * seq_len + key) * head_dim;
        const float score_gradient = probabilities[probability_offset + key] *
            (probability_gradients[key] - weighted_probability_gradient) * scale;
        for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
            queries_gradient[query_offset + dimension] +=
                score_gradient * keys[key_offset + dimension];
            atomicAdd(
                &keys_gradient[key_offset + dimension],
                score_gradient * queries[query_offset + dimension]);
        }
    }
}

__global__ void adamw_update_kernel(
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
    float first_correction,
    float second_correction) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    const float first = beta1 * first_moment[index] + (1.0F - beta1) * gradient[index];
    const float second = beta2 * second_moment[index] +
        (1.0F - beta2) * gradient[index] * gradient[index];
    first_moment[index] = first;
    second_moment[index] = second;
    const float corrected_first = first / first_correction;
    const float corrected_second = second / second_correction;
    const float value = parameter[index];
    const float adam_update = learning_rate * corrected_first /
        (sqrtf(corrected_second) + epsilon);
    const float decay_update = apply_weight_decay ? learning_rate * weight_decay * value : 0.0F;
    parameter[index] = value - adam_update - decay_update;
}

__global__ void scale_kernel(float* values, std::size_t size, float factor) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) values[index] *= factor;
}

__global__ void add_to_kernel(float* target, const float* source, std::size_t size) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) target[index] += source[index];
}

__global__ void squared_sum_kernel(const float* values, std::size_t size, double* output) {
    double sum = 0.0;
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < size;
         index += blockDim.x * gridDim.x) {
        const float value = values[index];
        sum += static_cast<double>(value) * value;
    }
    atomicAdd(output, sum);
}

__global__ void finite_kernel(const float* values, std::size_t size, int* valid) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && !isfinite(values[index])) {
        *valid = 0;
    }
}

}  // namespace

float cross_entropy(
    const float* logits,
    const std::uint32_t* targets,
    const std::uint8_t* mask,
    float* output_gradient,
    std::size_t batch_size,
    std::size_t seq_len,
    std::size_t vocab_size,
    std::size_t valid_positions) {
    const std::size_t positions = batch_size * seq_len;
    DeviceBuffer<float> device_logits(positions * vocab_size);
    DeviceBuffer<std::uint32_t> device_targets(positions);
    DeviceBuffer<std::uint8_t> device_mask(mask != nullptr ? positions : 0);
    DeviceBuffer<float> device_gradient(output_gradient != nullptr ? positions * vocab_size : 0);
    DeviceBuffer<float> device_losses(positions);
    copy_to_device(device_logits.get(), logits, positions * vocab_size);
    copy_to_device(device_targets.get(), targets, positions);
    if (mask != nullptr) copy_to_device(device_mask.get(), mask, positions);
    float* gradient_kernel = output_gradient != nullptr
        ? output_pointer(output_gradient, device_gradient)
        : nullptr;
    if (gradient_kernel != nullptr) {
        check_cuda(cudaMemset(
                       gradient_kernel,
                       0,
                       positions * vocab_size * sizeof(float)),
                   "clear cross entropy gradient");
    }
    cross_entropy_kernel<<<(positions + 255) / 256, 256>>>(
        device_logits.get(),
        device_targets.get(),
        mask != nullptr ? device_mask.get() : nullptr,
        gradient_kernel,
        device_losses.get(),
        positions,
        vocab_size,
        valid_positions);
    check_cuda(cudaGetLastError(), "Cross-Entropy kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize Cross-Entropy");

    std::vector<float> losses(positions);
    copy_to_host(losses.data(), device_losses.get(), positions);
    if (output_gradient != nullptr) {
        copy_to_host(output_gradient, device_gradient.get(), positions * vocab_size);
    }
    double total = 0.0;
    for (std::size_t position = 0; position < positions; ++position) {
        if (mask == nullptr || mask[position] != 0) {
            total += losses[position];
        }
    }
    return static_cast<float>(total / valid_positions);
}

void linear_backward(
    const float* input,
    const float* weights,
    const float* output_gradient,
    float* input_gradient,
    float* weight_gradient,
    std::size_t rows,
    std::size_t input_dim,
    std::size_t output_dim) {
    if (rows == 0 || input_dim == 0 || output_dim == 0) return;
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        input_dim > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        output_dim > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Linear backward dimensions exceed cuBLAS limits");
    }
    const std::size_t input_size = rows * input_dim;
    const std::size_t weight_size = input_dim * output_dim;
    DeviceBuffer<float> device_input(input_size), device_weights(weight_size),
        device_output_gradient(rows * output_dim), device_input_gradient(input_size),
        device_weight_gradient(weight_size);
    copy_to_device(device_input.get(), input, input_size);
    copy_to_device(device_weights.get(), weights, weight_size);
    copy_to_device(device_output_gradient.get(), output_gradient, rows * output_dim);
    float* input_gradient_kernel = output_pointer(input_gradient, device_input_gradient);
    float* weight_gradient_kernel = output_pointer(weight_gradient, device_weight_gradient);

    cublasHandle_t handle = nullptr;
    check_cublas(cublasCreate(&handle), "cublasCreate linear backward");
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    try {
        check_cublas(
            cublasSgemm(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(input_dim), static_cast<int>(rows), static_cast<int>(output_dim),
                &alpha, device_weights.get(), static_cast<int>(output_dim),
                device_output_gradient.get(), static_cast<int>(output_dim),
                &beta, input_gradient_kernel, static_cast<int>(input_dim)),
            "linear input gradient");
        check_cublas(
            cublasSgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(output_dim), static_cast<int>(input_dim), static_cast<int>(rows),
                &alpha, device_output_gradient.get(), static_cast<int>(output_dim),
                device_input.get(), static_cast<int>(input_dim),
                &beta, weight_gradient_kernel, static_cast<int>(output_dim)),
            "linear weight gradient");
        check_cuda(cudaDeviceSynchronize(), "synchronize linear backward");
        check_cublas(cublasDestroy(handle), "cublasDestroy linear backward");
    } catch (...) {
        cublasDestroy(handle);
        throw;
    }
    copy_to_host(input_gradient, device_input_gradient.get(), input_size);
    copy_to_host(weight_gradient, device_weight_gradient.get(), weight_size);
}

void embedding_backward(
    const std::uint32_t* ids,
    const float* output_gradient,
    float* token_gradient,
    float* positional_gradient,
    std::size_t batch_size,
    std::size_t seq_len,
    std::size_t embedding_dim,
    std::size_t vocab_size,
    std::size_t max_seq_len) {
    const std::size_t token_count = batch_size * seq_len;
    const std::size_t gradient_size = token_count * embedding_dim;
    DeviceBuffer<std::uint32_t> device_ids(token_count);
    DeviceBuffer<float> device_output_gradient(gradient_size);
    DeviceBuffer<float> device_token_gradient(vocab_size * embedding_dim);
    DeviceBuffer<float> device_positional_gradient(max_seq_len * embedding_dim);
    copy_to_device(device_ids.get(), ids, token_count);
    copy_to_device(device_output_gradient.get(), output_gradient, gradient_size);
    float* token_gradient_kernel = output_pointer(token_gradient, device_token_gradient);
    float* positional_gradient_kernel = output_pointer(positional_gradient, device_positional_gradient);
    check_cuda(cudaMemset(token_gradient_kernel, 0, vocab_size * embedding_dim * sizeof(float)), "clear token gradient");
    check_cuda(cudaMemset(positional_gradient_kernel, 0, max_seq_len * embedding_dim * sizeof(float)), "clear positional gradient");
    embedding_backward_kernel<<<(gradient_size + 255) / 256, 256>>>(
        device_ids.get(),
        device_output_gradient.get(),
        token_gradient_kernel,
        positional_gradient_kernel,
        token_count,
        seq_len,
        embedding_dim);
    check_cuda(cudaGetLastError(), "embedding backward kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize embedding backward");
    copy_to_host(token_gradient, device_token_gradient.get(), vocab_size * embedding_dim);
    copy_to_host(positional_gradient, device_positional_gradient.get(), max_seq_len * embedding_dim);
}

void layer_norm_backward(
    const float* input,
    const float* gamma,
    const float* output_gradient,
    float* input_gradient,
    float* gamma_gradient,
    float* beta_gradient,
    std::size_t rows,
    std::size_t dimension,
    float epsilon) {
    DeviceBuffer<float> device_input(rows * dimension), device_gamma(dimension),
        device_output_gradient(rows * dimension), device_input_gradient(rows * dimension),
        device_gamma_gradient(dimension), device_beta_gradient(dimension);
    copy_to_device(device_input.get(), input, rows * dimension);
    copy_to_device(device_gamma.get(), gamma, dimension);
    copy_to_device(device_output_gradient.get(), output_gradient, rows * dimension);
    float* input_gradient_kernel = output_pointer(input_gradient, device_input_gradient);
    float* gamma_gradient_kernel = output_pointer(gamma_gradient, device_gamma_gradient);
    float* beta_gradient_kernel = output_pointer(beta_gradient, device_beta_gradient);
    check_cuda(cudaMemset(input_gradient_kernel, 0, rows * dimension * sizeof(float)), "clear input gradient");
    check_cuda(cudaMemset(gamma_gradient_kernel, 0, dimension * sizeof(float)), "clear gamma gradient");
    check_cuda(cudaMemset(beta_gradient_kernel, 0, dimension * sizeof(float)), "clear beta gradient");
    layer_norm_backward_kernel<<<(rows + 255) / 256, 256>>>(
        device_input.get(),
        device_gamma.get(),
        device_output_gradient.get(),
        input_gradient_kernel,
        gamma_gradient_kernel,
        beta_gradient_kernel,
        rows,
        dimension,
        epsilon);
    check_cuda(cudaGetLastError(), "LayerNorm backward kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize LayerNorm backward");
    copy_to_host(input_gradient, device_input_gradient.get(), rows * dimension);
    copy_to_host(gamma_gradient, device_gamma_gradient.get(), dimension);
    copy_to_host(beta_gradient, device_beta_gradient.get(), dimension);
}

void gelu_backward(const float* input, const float* output_gradient, float* input_gradient, std::size_t size) {
    if (size == 0) return;
    DeviceBuffer<float> device_input(size), device_output_gradient(size), device_input_gradient(size);
    copy_to_device(device_input.get(), input, size);
    copy_to_device(device_output_gradient.get(), output_gradient, size);
    gelu_backward_kernel<<<(size + 255) / 256, 256>>>(
        device_input.get(), device_output_gradient.get(), output_pointer(input_gradient, device_input_gradient), size);
    check_cuda(cudaGetLastError(), "GELU backward kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize GELU backward");
    copy_to_host(input_gradient, device_input_gradient.get(), size);
}

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
    std::size_t head_dim) {
    if (batch_size == 0 || seq_len == 0) return;
    const std::size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    const std::size_t score_size = batch_size * num_heads * seq_len * seq_len;
    const std::size_t context_size = batch_size * seq_len * num_heads * head_dim;
    DeviceBuffer<float> device_queries(qkv_size), device_keys(qkv_size), device_values(qkv_size);
    DeviceBuffer<float> device_probabilities(score_size), device_context_gradient(context_size);
    DeviceBuffer<float> device_queries_gradient(qkv_size), device_keys_gradient(qkv_size),
        device_values_gradient(qkv_size);
    copy_to_device(device_queries.get(), queries, qkv_size);
    copy_to_device(device_keys.get(), keys, qkv_size);
    copy_to_device(device_values.get(), values, qkv_size);
    copy_to_device(device_probabilities.get(), probabilities, score_size);
    copy_to_device(device_context_gradient.get(), context_gradient, context_size);
    float* queries_gradient_kernel = output_pointer(queries_gradient, device_queries_gradient);
    float* keys_gradient_kernel = output_pointer(keys_gradient, device_keys_gradient);
    float* values_gradient_kernel = output_pointer(values_gradient, device_values_gradient);
    check_cuda(cudaMemset(queries_gradient_kernel, 0, qkv_size * sizeof(float)), "clear queries gradient");
    check_cuda(cudaMemset(keys_gradient_kernel, 0, qkv_size * sizeof(float)), "clear keys gradient");
    check_cuda(cudaMemset(values_gradient_kernel, 0, qkv_size * sizeof(float)), "clear values gradient");
    dim3 blocks(
        static_cast<unsigned int>(batch_size),
        static_cast<unsigned int>(num_heads),
        static_cast<unsigned int>(seq_len));
    causal_attention_backward_kernel<<<blocks, 1, seq_len * sizeof(float)>>>(
        device_queries.get(),
        device_keys.get(),
        device_values.get(),
        device_probabilities.get(),
        device_context_gradient.get(),
        queries_gradient_kernel,
        keys_gradient_kernel,
        values_gradient_kernel,
        num_heads,
        seq_len,
        head_dim);
    check_cuda(cudaGetLastError(), "causal attention backward kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize causal attention backward");
    copy_to_host(queries_gradient, device_queries_gradient.get(), qkv_size);
    copy_to_host(keys_gradient, device_keys_gradient.get(), qkv_size);
    copy_to_host(values_gradient, device_values_gradient.get(), qkv_size);
}

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
    std::size_t step) {
    if (size == 0) return;
    DeviceBuffer<float> device_parameter(size), device_gradient(size), device_first(size), device_second(size);
    copy_to_device(device_parameter.get(), parameter, size);
    copy_to_device(device_gradient.get(), gradient, size);
    copy_to_device(device_first.get(), first_moment, size);
    copy_to_device(device_second.get(), second_moment, size);
    const float first_correction = 1.0F - std::pow(beta1, static_cast<float>(step));
    const float second_correction = 1.0F - std::pow(beta2, static_cast<float>(step));
    adamw_update_kernel<<<(size + 255) / 256, 256>>>(
        output_pointer(parameter, device_parameter),
        device_gradient.get(),
        output_pointer(first_moment, device_first),
        output_pointer(second_moment, device_second),
        size,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        apply_weight_decay,
        first_correction,
        second_correction);
    check_cuda(cudaGetLastError(), "AdamW kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize AdamW");
    copy_to_host(parameter, device_parameter.get(), size);
    copy_to_host(first_moment, device_first.get(), size);
    copy_to_host(second_moment, device_second.get(), size);
}

void scale(float* values, std::size_t size, float factor) {
    if (size == 0) return;
    DeviceBuffer<float> device_values(size);
    copy_to_device(device_values.get(), values, size);
    scale_kernel<<<(size + 255) / 256, 256>>>(output_pointer(values, device_values), size, factor);
    check_cuda(cudaGetLastError(), "scale kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize scale");
    copy_to_host(values, device_values.get(), size);
}

void add_to(float* target, const float* source, std::size_t size) {
    if (size == 0) return;
    DeviceBuffer<float> device_target(size), device_source(size);
    copy_to_device(device_target.get(), target, size);
    copy_to_device(device_source.get(), source, size);
    add_to_kernel<<<(size + 255) / 256, 256>>>(output_pointer(target, device_target), device_source.get(), size);
    check_cuda(cudaGetLastError(), "add_to kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize add_to");
    copy_to_host(target, device_target.get(), size);
}

bool all_finite(const float* values, std::size_t size) {
    if (size == 0) return true;
    DeviceBuffer<float> device_values(size);
    DeviceBuffer<int> device_valid(1);
    copy_to_device(device_values.get(), values, size);
    const int initial = 1;
    check_cuda(
        cudaMemcpy(device_valid.get(), &initial, sizeof(int), cudaMemcpyHostToDevice),
        "initialize finite check");
    finite_kernel<<<(size + 255) / 256, 256>>>(device_values.get(), size, device_valid.get());
    check_cuda(cudaGetLastError(), "finite check kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize finite check");
    int valid = 0;
    copy_to_host(&valid, device_valid.get(), 1);
    return valid != 0;
}

double gradient_global_norm(const std::vector<std::pair<const float*, std::size_t>>& gradients) {
    double squared_sum = 0.0;
    for (const auto& [values, size] : gradients) {
        if (size == 0) continue;
        DeviceBuffer<float> device_values(size);
        DeviceBuffer<double> device_sum(1);
        copy_to_device(device_values.get(), values, size);
        check_cuda(cudaMemset(device_sum.get(), 0, sizeof(double)), "clear squared sum");
        const std::size_t blocks = std::min<std::size_t>((size + 255) / 256, 256);
        squared_sum_kernel<<<blocks, 256>>>(device_values.get(), size, device_sum.get());
        check_cuda(cudaGetLastError(), "squared sum kernel");
        check_cuda(cudaDeviceSynchronize(), "synchronize squared sum");
        double sum = 0.0;
        copy_to_host(&sum, device_sum.get(), 1);
        squared_sum += sum;
    }
    return std::sqrt(squared_sum);
}

}  // namespace mini_gpt::cuda

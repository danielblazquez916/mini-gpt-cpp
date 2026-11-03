#include "mini_gpt/layer_norm.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace mini_gpt {
namespace {

std::size_t require_positive(std::size_t value) {
    if (value == 0) {
        throw std::invalid_argument("normalized_dim must be greater than zero");
    }
    return value;
}

float require_positive_epsilon(float epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument("LayerNorm epsilon must be finite and greater than zero");
    }
    return epsilon;
}

}  // namespace

LayerNorm::LayerNorm(std::size_t normalized_dim, float epsilon)
    : normalized_dim_(require_positive(normalized_dim)),
      epsilon_(require_positive_epsilon(epsilon)),
      gamma_(Tensor::ones({normalized_dim_})),
      beta_(Tensor::zeros({normalized_dim_})) {}

Tensor LayerNorm::forward(const Tensor& input) const {
    validate_parameters();
    if (input.rank() == 0 || input.shape().back() != normalized_dim_) {
        throw std::invalid_argument("LayerNorm input must end with normalized_dim");
    }

    Tensor output = Tensor::zeros(input.shape());
    const float* input_data = input.data().data();
    const float* gamma_data = gamma_.data().data();
    const float* beta_data = beta_.data().data();
    float* output_data = output.data().data();
    const std::size_t rows = input.size() / normalized_dim_;

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * normalized_dim_;
        double sum = 0.0;
        for (std::size_t i = 0; i < normalized_dim_; ++i) {
            sum += input_data[offset + i];
        }
        const double mean = sum / static_cast<double>(normalized_dim_);

        double squared_differences = 0.0;
        for (std::size_t i = 0; i < normalized_dim_; ++i) {
            const double difference = input_data[offset + i] - mean;
            squared_differences += difference * difference;
        }
        const double variance =
            squared_differences / static_cast<double>(normalized_dim_);
        const double inverse_standard_deviation =
            1.0 / std::sqrt(variance + static_cast<double>(epsilon_));

        for (std::size_t i = 0; i < normalized_dim_; ++i) {
            const double normalized =
                (input_data[offset + i] - mean) * inverse_standard_deviation;
            output_data[offset + i] = static_cast<float>(
                normalized * gamma_data[i] + beta_data[i]);
        }
    }
    return output;
}

std::size_t LayerNorm::normalized_dim() const noexcept {
    return normalized_dim_;
}

float LayerNorm::epsilon() const noexcept {
    return epsilon_;
}

Tensor& LayerNorm::gamma() noexcept {
    return gamma_;
}

const Tensor& LayerNorm::gamma() const noexcept {
    return gamma_;
}

Tensor& LayerNorm::beta() noexcept {
    return beta_;
}

const Tensor& LayerNorm::beta() const noexcept {
    return beta_;
}

void LayerNorm::validate_parameters() const {
    if (gamma_.shape() != Tensor::Shape{normalized_dim_} ||
        beta_.shape() != Tensor::Shape{normalized_dim_}) {
        throw std::logic_error("LayerNorm parameter shapes do not match normalized_dim");
    }
}

}  // namespace mini_gpt

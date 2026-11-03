#include "mini_gpt/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace mini_gpt {
namespace {

std::string shape_string(const Tensor::Shape& shape) {
    std::ostringstream output;
    output << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            output << ", ";
        }
        output << shape[i];
    }
    output << ']';
    return output.str();
}

template <typename Operation>
Tensor elementwise(const Tensor& lhs, const Tensor& rhs, Operation operation) {
    if (lhs.shape() != rhs.shape()) {
        throw std::invalid_argument(
            "Elementwise operations require identical shapes: " +
            shape_string(lhs.shape()) + " and " + shape_string(rhs.shape()));
    }

    Tensor result = Tensor::zeros(lhs.shape());
    std::transform(
        lhs.data().begin(),
        lhs.data().end(),
        rhs.data().begin(),
        result.data().begin(),
        operation);
    return result;
}

}  // namespace

Tensor::Tensor(Shape shape)
    : shape_(std::move(shape)),
      strides_(calculate_strides(shape_)),
      size_(element_count(shape_)),
      data_(size_, 0.0F) {}

Tensor::Tensor(Shape shape, std::vector<float> data)
    : shape_(std::move(shape)),
      strides_(calculate_strides(shape_)),
      size_(element_count(shape_)),
      data_(std::move(data)) {
    if (data_.size() != size_) {
        throw std::invalid_argument(
            "Data size does not match shape " + shape_string(shape_) +
            ": expected " + std::to_string(size_) + ", got " +
            std::to_string(data_.size()));
    }
}

Tensor Tensor::zeros(Shape shape) {
    return Tensor(std::move(shape));
}

Tensor Tensor::ones(Shape shape) {
    Tensor result(std::move(shape));
    std::fill(result.data_.begin(), result.data_.end(), 1.0F);
    return result;
}

Tensor Tensor::random(Shape shape, std::uint32_t seed, float min, float max) {
    if (!std::isfinite(min) || !std::isfinite(max) || min > max) {
        throw std::invalid_argument("Random bounds must be finite and satisfy min <= max");
    }

    Tensor result(std::move(shape));
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(min, max);
    for (float& value : result.data_) {
        value = distribution(generator);
    }
    return result;
}

const Tensor::Shape& Tensor::shape() const noexcept {
    return shape_;
}

const Tensor::Shape& Tensor::strides() const noexcept {
    return strides_;
}

std::size_t Tensor::size() const noexcept {
    return size_;
}

std::size_t Tensor::rank() const noexcept {
    return shape_.size();
}

std::span<float> Tensor::data() noexcept {
    return std::span<float>(data_);
}

std::span<const float> Tensor::data() const noexcept {
    return std::span<const float>(data_);
}

float& Tensor::at(std::initializer_list<std::size_t> indices) {
    return data_[offset(indices)];
}

const float& Tensor::at(std::initializer_list<std::size_t> indices) const {
    return data_[offset(indices)];
}

Tensor Tensor::operator+(const Tensor& other) const {
    return elementwise(*this, other, std::plus<float>{});
}

Tensor Tensor::operator-(const Tensor& other) const {
    return elementwise(*this, other, std::minus<float>{});
}

Tensor Tensor::operator*(const Tensor& other) const {
    return elementwise(*this, other, std::multiplies<float>{});
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (rank() != 2 || other.rank() != 2) {
        throw std::invalid_argument("matmul requires two 2D tensors");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument(
            "matmul inner dimensions must match: " + shape_string(shape_) +
            " and " + shape_string(other.shape_));
    }

    const std::size_t rows = shape_[0];
    const std::size_t inner = shape_[1];
    const std::size_t columns = other.shape_[1];
    Tensor result = zeros({rows, columns});

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t k = 0; k < inner; ++k) {
            const float lhs_value = data_[row * inner + k];
            for (std::size_t column = 0; column < columns; ++column) {
                result.data_[row * columns + column] +=
                    lhs_value * other.data_[k * columns + column];
            }
        }
    }
    return result;
}

Tensor Tensor::transpose() const {
    if (rank() != 2) {
        throw std::invalid_argument("transpose requires a 2D tensor");
    }

    const std::size_t rows = shape_[0];
    const std::size_t columns = shape_[1];
    Tensor result = zeros({columns, rows});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            result.data_[column * rows + row] = data_[row * columns + column];
        }
    }
    return result;
}

void Tensor::reshape(Shape new_shape) {
    const std::size_t new_size = element_count(new_shape);
    Shape new_strides = calculate_strides(new_shape);
    if (new_size != size_) {
        throw std::invalid_argument(
            "reshape cannot change the element count from " + std::to_string(size_) +
            " to " + std::to_string(new_size));
    }

    shape_ = std::move(new_shape);
    strides_ = std::move(new_strides);
}

std::size_t Tensor::element_count(const Shape& shape) {
    if (std::find(shape.begin(), shape.end(), 0U) != shape.end()) {
        return 0;
    }

    std::size_t count = 1;
    for (const std::size_t dimension : shape) {
        if (dimension > std::numeric_limits<std::size_t>::max() / count) {
            throw std::length_error("Tensor element count exceeds size_t range");
        }
        count *= dimension;
    }
    return count;
}

Tensor::Shape Tensor::calculate_strides(const Shape& shape) {
    Shape strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        if (stride != 0 && shape[i] > std::numeric_limits<std::size_t>::max() / stride) {
            throw std::length_error("Tensor strides exceed size_t range");
        }
        stride *= shape[i];
    }
    return strides;
}

std::size_t Tensor::offset(std::initializer_list<std::size_t> indices) const {
    if (indices.size() != rank()) {
        throw std::invalid_argument(
            "Expected " + std::to_string(rank()) + " indices, got " +
            std::to_string(indices.size()));
    }

    std::size_t result = 0;
    std::size_t axis = 0;
    for (const std::size_t index : indices) {
        if (index >= shape_[axis]) {
            throw std::out_of_range(
                "Index " + std::to_string(index) + " is out of range for axis " +
                std::to_string(axis) + " with size " + std::to_string(shape_[axis]));
        }
        result += index * strides_[axis];
        ++axis;
    }
    return result;
}

}  // namespace mini_gpt

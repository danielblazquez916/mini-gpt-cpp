#ifndef MINI_GPT_LAYER_NORM_HPP
#define MINI_GPT_LAYER_NORM_HPP

#include "mini_gpt/tensor.hpp"

#include <cstddef>

namespace mini_gpt {

class LayerNorm {
public:
    LayerNorm(std::size_t normalized_dim, float epsilon = 1.0e-5F);

    [[nodiscard]] Tensor forward(const Tensor& input) const;

    [[nodiscard]] std::size_t normalized_dim() const noexcept;
    [[nodiscard]] float epsilon() const noexcept;
    [[nodiscard]] Tensor& gamma() noexcept;
    [[nodiscard]] const Tensor& gamma() const noexcept;
    [[nodiscard]] Tensor& beta() noexcept;
    [[nodiscard]] const Tensor& beta() const noexcept;

private:
    void validate_parameters() const;

    std::size_t normalized_dim_;
    float epsilon_;
    Tensor gamma_;
    Tensor beta_;
};

}  // namespace mini_gpt

#endif

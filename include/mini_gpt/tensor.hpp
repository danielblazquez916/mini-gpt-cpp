#ifndef MINI_GPT_TENSOR_HPP
#define MINI_GPT_TENSOR_HPP

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <vector>

namespace mini_gpt {

class Tensor {
public:
    using Shape = std::vector<std::size_t>;

    explicit Tensor(Shape shape);
    Tensor(Shape shape, std::vector<float> data);

    [[nodiscard]] static Tensor zeros(Shape shape);
    [[nodiscard]] static Tensor ones(Shape shape);

    // Genera valores uniformes en [min, max) usando la semilla indicada.
    [[nodiscard]] static Tensor random(
        Shape shape,
        std::uint32_t seed,
        float min = -1.0F,
        float max = 1.0F);

    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] const Shape& strides() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;

    [[nodiscard]] std::span<float> data() noexcept;
    [[nodiscard]] std::span<const float> data() const noexcept;

    float& at(std::initializer_list<std::size_t> indices);
    [[nodiscard]] const float& at(std::initializer_list<std::size_t> indices) const;

    [[nodiscard]] Tensor operator+(const Tensor& other) const;
    [[nodiscard]] Tensor operator-(const Tensor& other) const;
    [[nodiscard]] Tensor operator*(const Tensor& other) const;

    [[nodiscard]] Tensor matmul(const Tensor& other) const;
    [[nodiscard]] Tensor transpose() const;

    // Cambia la forma in situ, sin copiar ni reordenar los datos.
    void reshape(Shape new_shape);

private:
    [[nodiscard]] static std::size_t element_count(const Shape& shape);
    [[nodiscard]] static Shape calculate_strides(const Shape& shape);
    [[nodiscard]] std::size_t offset(std::initializer_list<std::size_t> indices) const;

    Shape shape_;
    Shape strides_;
    std::size_t size_;
    std::vector<float> data_;
};

}  // namespace mini_gpt

#endif

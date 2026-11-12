#ifndef MINI_GPT_ADAMW_HPP
#define MINI_GPT_ADAMW_HPP

#include "mini_gpt/gpt.hpp"
#include "mini_gpt/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace mini_gpt {

struct AdamWParameter {
    Tensor* parameter = nullptr;
    Tensor* gradient = nullptr;
    bool weight_decay = true;
    std::string name;
};

using AdamWParameters = std::vector<AdamWParameter>;

struct AdamWConfig {
    float learning_rate = 0.0003F;
    float beta1 = 0.9F;
    float beta2 = 0.999F;
    float epsilon = 1.0e-8F;
    float weight_decay = 0.0F;
};

[[nodiscard]] AdamWParameters gpt_parameters(GPT& model);
[[nodiscard]] double gradient_global_norm(const AdamWParameters& parameters);
double clip_gradients(const AdamWParameters& parameters, double max_norm);

class AdamW {
public:
    struct State {
        Tensor first_moment = Tensor::zeros({0});
        Tensor second_moment = Tensor::zeros({0});
    };

    explicit AdamW(GPT& model, AdamWConfig config = {});
    explicit AdamW(AdamWParameters parameters, AdamWConfig config = {});

    void step();

    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static AdamW load(const std::filesystem::path& path, GPT& model);
    [[nodiscard]] static AdamW load(
        const std::filesystem::path& path,
        AdamWParameters parameters);

    [[nodiscard]] const AdamWConfig& config() const noexcept;
    [[nodiscard]] std::size_t step_count() const noexcept;
    [[nodiscard]] std::size_t parameter_count() const noexcept;
    [[nodiscard]] const AdamWParameters& parameters() const noexcept;
    [[nodiscard]] const State& state(std::size_t index) const;

private:
    void initialize_states();
    void validate_before_step() const;

    AdamWParameters parameters_;
    std::vector<State> states_;
    AdamWConfig config_;
    std::size_t step_count_ = 0;
};

}  // namespace mini_gpt

#endif

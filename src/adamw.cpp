#include "mini_gpt/adamw.hpp"

#ifdef MINI_GPT_ENABLE_CUDA
#include "mini_gpt/cuda/backward.cuh"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace mini_gpt {
namespace {

constexpr std::string_view file_magic = "MINIADAM";
constexpr std::uint32_t file_version = 1;
static_assert(sizeof(float) == sizeof(std::uint32_t));

AdamWConfig validate_config(AdamWConfig config) {
    if (!std::isfinite(config.learning_rate) || config.learning_rate < 0.0F ||
        !std::isfinite(config.beta1) || config.beta1 < 0.0F || config.beta1 >= 1.0F ||
        !std::isfinite(config.beta2) || config.beta2 < 0.0F || config.beta2 >= 1.0F ||
        !std::isfinite(config.epsilon) || config.epsilon <= 0.0F ||
        !std::isfinite(config.weight_decay) || config.weight_decay < 0.0F) {
        throw std::invalid_argument("Invalid AdamW configuration");
    }
    return config;
}

void write_u32(std::ofstream& file, std::uint32_t value) {
    std::array<char, 4> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((value >> (i * 8)) & 0xFFU);
    }
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to write AdamW file");
    }
}

void write_u64(std::ofstream& file, std::uint64_t value) {
    write_u32(file, static_cast<std::uint32_t>(value & 0xFFFFFFFFU));
    write_u32(file, static_cast<std::uint32_t>(value >> 32));
}

void write_float(std::ofstream& file, float value) {
    write_u32(file, std::bit_cast<std::uint32_t>(value));
}

std::uint32_t read_u32(std::ifstream& file, std::uint64_t& remaining) {
    if (remaining < 4) {
        throw std::runtime_error("Unexpected end of AdamW file");
    }
    std::array<unsigned char, 4> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to read AdamW file");
    }
    remaining -= 4;

    std::uint32_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<std::uint32_t>(bytes[i]) << (i * 8);
    }
    return value;
}

std::uint64_t read_u64(std::ifstream& file, std::uint64_t& remaining) {
    const std::uint64_t low = read_u32(file, remaining);
    const std::uint64_t high = read_u32(file, remaining);
    return low | (high << 32);
}

float read_float(std::ifstream& file, std::uint64_t& remaining) {
    return std::bit_cast<float>(read_u32(file, remaining));
}

std::uint8_t read_u8(std::ifstream& file, std::uint64_t& remaining) {
    if (remaining < 1) {
        throw std::runtime_error("Unexpected end of AdamW file");
    }
    char value = 0;
    file.read(&value, 1);
    if (!file) {
        throw std::runtime_error("Failed to read AdamW file");
    }
    --remaining;
    return static_cast<std::uint8_t>(value);
}

void write_u8(std::ofstream& file, std::uint8_t value) {
    const char byte = static_cast<char>(value);
    file.write(&byte, 1);
    if (!file) {
        throw std::runtime_error("Failed to write AdamW file");
    }
}

std::vector<float> read_values(
    std::ifstream& file,
    std::uint64_t& remaining,
    std::size_t count) {
    std::vector<float> values(count);
    for (float& value : values) {
        value = read_float(file, remaining);
    }
    return values;
}

void validate_binding(const AdamWParameter& binding) {
    if (binding.parameter == nullptr || binding.gradient == nullptr ||
        binding.parameter->shape() != binding.gradient->shape()) {
        throw std::invalid_argument("Invalid parameter/gradient binding in AdamW");
    }
}

}  // namespace

AdamWParameters gpt_parameters(GPT& model) {
    AdamWParameters parameters;
    auto& gradients = model.gradients();
    const auto add = [&](Tensor& parameter, Tensor& gradient, bool decay, std::string name) {
        parameters.push_back(AdamWParameter{&parameter, &gradient, decay, std::move(name)});
    };

    add(model.embedding().token_embeddings(), gradients.embedding.token_embeddings, true,
        "embedding.token");
    add(model.embedding().positional_embeddings(), gradients.embedding.positional_embeddings,
        true, "embedding.position");
    for (std::size_t i = 0; i < model.blocks().size(); ++i) {
        auto& block = model.blocks()[i];
        auto& gradient = gradients.blocks[i];
        const std::string prefix = "block" + std::to_string(i) + ".";
        add(block.norm1().gamma(), gradient.norm1.gamma, false, prefix + "norm1.gamma");
        add(block.norm1().beta(), gradient.norm1.beta, false, prefix + "norm1.beta");
        add(block.norm2().gamma(), gradient.norm2.gamma, false, prefix + "norm2.gamma");
        add(block.norm2().beta(), gradient.norm2.beta, false, prefix + "norm2.beta");
        add(block.attention().q_weights(), gradient.attention.q_weights, true, prefix + "q");
        add(block.attention().k_weights(), gradient.attention.k_weights, true, prefix + "k");
        add(block.attention().v_weights(), gradient.attention.v_weights, true, prefix + "v");
        add(block.attention().output_weights(), gradient.attention.output_weights, true,
            prefix + "attention.output");
        add(block.fc1_weights(), gradient.fc1_weights, true, prefix + "fc1");
        add(block.fc2_weights(), gradient.fc2_weights, true, prefix + "fc2");
    }
    add(model.final_norm().gamma(), gradients.final_norm.gamma, false, "final_norm.gamma");
    add(model.final_norm().beta(), gradients.final_norm.beta, false, "final_norm.beta");
    add(model.output_weights(), gradients.output_weights, true, "output_weights");
    return parameters;
}

double gradient_global_norm(const AdamWParameters& parameters) {
#ifdef MINI_GPT_ENABLE_CUDA
    std::vector<std::pair<const float*, std::size_t>> gradients;
    gradients.reserve(parameters.size());
#endif
#ifndef MINI_GPT_ENABLE_CUDA
    double squared_sum = 0.0;
#endif
    for (const AdamWParameter& binding : parameters) {
        validate_binding(binding);
#ifdef MINI_GPT_ENABLE_CUDA
        gradients.emplace_back(binding.gradient->raw_data(), binding.gradient->size());
#else
        for (const float value : binding.gradient->data()) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Non-finite gradient in " + binding.name);
            }
            squared_sum += static_cast<double>(value) * value;
        }
#endif
    }
#ifdef MINI_GPT_ENABLE_CUDA
    const double norm = cuda::gradient_global_norm(gradients);
    if (!std::isfinite(norm)) {
        for (const AdamWParameter& binding : parameters) {
            if (!cuda::all_finite(binding.gradient->raw_data(), binding.gradient->size())) {
                throw std::runtime_error("Non-finite gradient in " + binding.name);
            }
        }
        throw std::runtime_error("Non-finite global gradient norm");
    }
    return norm;
#else
    return std::sqrt(squared_sum);
#endif
}

double clip_gradients(const AdamWParameters& parameters, double max_norm) {
    if (!std::isfinite(max_norm) || max_norm <= 0.0) {
        throw std::invalid_argument("Gradient clipping norm must be finite and positive");
    }
    const double norm = gradient_global_norm(parameters);
    if (norm > max_norm) {
        const double scale = max_norm / norm;
        for (const AdamWParameter& binding : parameters) {
#ifdef MINI_GPT_ENABLE_CUDA
            cuda::scale(binding.gradient->raw_data(), binding.gradient->size(), static_cast<float>(scale));
#else
            for (float& value : binding.gradient->data()) {
                value = static_cast<float>(value * scale);
            }
#endif
        }
    }
    return norm;
}

AdamW::AdamW(GPT& model, AdamWConfig config)
    : AdamW(gpt_parameters(model), config) {}

AdamW::AdamW(AdamWParameters parameters, AdamWConfig config)
    : parameters_(std::move(parameters)), config_(validate_config(config)) {
    if (parameters_.empty()) {
        throw std::invalid_argument("AdamW requires at least one parameter tensor");
    }
    initialize_states();
}

void AdamW::step() {
    validate_before_step();
    if (step_count_ == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("AdamW step counter overflow");
    }
    ++step_count_;

    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        const AdamWParameter& binding = parameters_[index];
        State& state = states_[index];
#ifdef MINI_GPT_ENABLE_CUDA
        cuda::adamw_update(
            binding.parameter->raw_data(),
            binding.gradient->raw_data(),
            state.first_moment.raw_data(),
            state.second_moment.raw_data(),
            binding.parameter->size(),
            config_.learning_rate,
            config_.beta1,
            config_.beta2,
            config_.epsilon,
            config_.weight_decay,
            binding.weight_decay,
            step_count_);
        continue;
#else
        const double beta1 = config_.beta1;
        const double beta2 = config_.beta2;
        const double first_correction = 1.0 - std::pow(beta1, static_cast<double>(step_count_));
        const double second_correction = 1.0 - std::pow(beta2, static_cast<double>(step_count_));
        for (std::size_t i = 0; i < binding.parameter->size(); ++i) {
            const double gradient = binding.gradient->data()[i];
            const double first = beta1 * state.first_moment.data()[i] +
                                 (1.0 - beta1) * gradient;
            const double second = beta2 * state.second_moment.data()[i] +
                                  (1.0 - beta2) * gradient * gradient;
            state.first_moment.data()[i] = static_cast<float>(first);
            state.second_moment.data()[i] = static_cast<float>(second);

            const double corrected_first = first / first_correction;
            const double corrected_second = second / second_correction;
            const double parameter = binding.parameter->data()[i];
            const double adam_update = config_.learning_rate * corrected_first /
                (std::sqrt(corrected_second) + config_.epsilon);
            const double decay_update = binding.weight_decay
                ? config_.learning_rate * config_.weight_decay * parameter
                : 0.0;
            binding.parameter->data()[i] =
                static_cast<float>(parameter - adam_update - decay_update);
        }
#endif
    }
}

void AdamW::save(const std::filesystem::path& path) const {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot open AdamW file for writing");
    }
    file.write(file_magic.data(), static_cast<std::streamsize>(file_magic.size()));
    if (!file) {
        throw std::runtime_error("Failed to write AdamW file");
    }
    write_u32(file, file_version);
    write_float(file, config_.learning_rate);
    write_float(file, config_.beta1);
    write_float(file, config_.beta2);
    write_float(file, config_.epsilon);
    write_float(file, config_.weight_decay);
    write_u64(file, step_count_);
    write_u64(file, parameters_.size());
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        write_u8(file, parameters_[index].weight_decay ? 1 : 0);
        write_u64(file, parameters_[index].parameter->size());
        for (const Tensor* tensor : {&states_[index].first_moment, &states_[index].second_moment}) {
            for (const float value : tensor->data()) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("Cannot save non-finite AdamW state");
                }
                write_float(file, value);
            }
        }
    }
    file.close();
    if (!file) {
        throw std::runtime_error("Failed to close AdamW file");
    }
}

AdamW AdamW::load(const std::filesystem::path& path, GPT& model) {
    return load(path, gpt_parameters(model));
}

AdamW AdamW::load(const std::filesystem::path& path, AdamWParameters parameters) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open AdamW file for reading");
    }
    const std::streamoff file_size = file.tellg();
    if (file_size < 0) {
        throw std::runtime_error("Cannot determine AdamW file size");
    }
    std::uint64_t remaining = static_cast<std::uint64_t>(file_size);
    file.seekg(0, std::ios::beg);

    if (remaining < file_magic.size()) {
        throw std::runtime_error("Invalid AdamW file header");
    }
    std::array<char, 8> magic{};
    file.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!file || std::string_view(magic.data(), magic.size()) != file_magic) {
        throw std::runtime_error("Invalid AdamW file magic");
    }
    remaining -= magic.size();
    if (read_u32(file, remaining) != file_version) {
        throw std::runtime_error("Unsupported AdamW file version");
    }

    AdamWConfig config;
    config.learning_rate = read_float(file, remaining);
    config.beta1 = read_float(file, remaining);
    config.beta2 = read_float(file, remaining);
    config.epsilon = read_float(file, remaining);
    config.weight_decay = read_float(file, remaining);
    const std::size_t step_count = static_cast<std::size_t>(read_u64(file, remaining));
    const std::size_t parameter_count = static_cast<std::size_t>(read_u64(file, remaining));
    if (parameter_count != parameters.size()) {
        throw std::runtime_error("AdamW parameter count does not match the model");
    }

    AdamW optimizer(std::move(parameters), config);
    optimizer.step_count_ = step_count;
    for (std::size_t index = 0; index < parameter_count; ++index) {
        const std::uint8_t decay = read_u8(file, remaining);
        if (decay > 1) {
            throw std::runtime_error("Invalid AdamW weight decay flag");
        }
        optimizer.parameters_[index].weight_decay = decay != 0;
        const std::uint64_t value_count = read_u64(file, remaining);
        if (value_count != optimizer.parameters_[index].parameter->size()) {
            throw std::runtime_error("AdamW state dimensions do not match the model");
        }
        const std::size_t count = static_cast<std::size_t>(value_count);
        optimizer.states_[index].first_moment = Tensor(
            optimizer.parameters_[index].parameter->shape(), read_values(file, remaining, count));
        optimizer.states_[index].second_moment = Tensor(
            optimizer.parameters_[index].parameter->shape(), read_values(file, remaining, count));
    }
    if (remaining != 0) {
        throw std::runtime_error("Unexpected trailing data in AdamW file");
    }
    return optimizer;
}

const AdamWConfig& AdamW::config() const noexcept {
    return config_;
}

std::size_t AdamW::step_count() const noexcept {
    return step_count_;
}

std::size_t AdamW::parameter_count() const noexcept {
    return parameters_.size();
}

const AdamWParameters& AdamW::parameters() const noexcept {
    return parameters_;
}

const AdamW::State& AdamW::state(std::size_t index) const {
    return states_.at(index);
}

void AdamW::initialize_states() {
    states_.clear();
    states_.reserve(parameters_.size());
    for (const AdamWParameter& binding : parameters_) {
        validate_binding(binding);
        states_.push_back(State{
            Tensor::zeros(binding.parameter->shape()),
            Tensor::zeros(binding.parameter->shape())});
    }
}

void AdamW::validate_before_step() const {
    if (states_.size() != parameters_.size()) {
        throw std::logic_error("AdamW state does not match parameter count");
    }
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        const AdamWParameter& binding = parameters_[index];
        validate_binding(binding);
        if (states_[index].first_moment.shape() != binding.parameter->shape() ||
            states_[index].second_moment.shape() != binding.parameter->shape()) {
            throw std::logic_error("AdamW state dimensions do not match parameters");
        }
#ifdef MINI_GPT_ENABLE_CUDA
        if (!cuda::all_finite(binding.parameter->raw_data(), binding.parameter->size())) {
            throw std::runtime_error("Non-finite parameter in " + binding.name);
        }
        if (!cuda::all_finite(binding.gradient->raw_data(), binding.gradient->size())) {
            throw std::runtime_error("Non-finite gradient in " + binding.name);
        }
#else
        for (const float value : binding.parameter->data()) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Non-finite parameter in " + binding.name);
            }
        }
        for (const float value : binding.gradient->data()) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Non-finite gradient in " + binding.name);
            }
        }
#endif
    }
}

}  // namespace mini_gpt

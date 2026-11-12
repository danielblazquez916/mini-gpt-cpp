#ifndef MINI_GPT_TRAINING_STATE_HPP
#define MINI_GPT_TRAINING_STATE_HPP

#include "mini_gpt/data_loader.hpp"

#include <cstddef>
#include <filesystem>
#include <string>

namespace mini_gpt {

struct TrainingState {
    std::string corpus_path;
    std::string tokenizer_path;
    std::size_t epoch = 0;
    std::size_t processed_tokens = 0;
    DataLoaderState data;
};

void save_training_state(const std::filesystem::path& path, const TrainingState& state);
[[nodiscard]] TrainingState load_training_state(const std::filesystem::path& path);

}  // namespace mini_gpt

#endif

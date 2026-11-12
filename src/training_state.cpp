#include "mini_gpt/training_state.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace mini_gpt {

void save_training_state(const std::filesystem::path& path, const TrainingState& state) {
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot open training state for writing: " + path.string());
    }
    file << "MINI_GPT_TRAINING_STATE 1\n"
         << "corpus " << std::quoted(state.corpus_path) << '\n'
         << "tokenizer " << std::quoted(state.tokenizer_path) << '\n'
         << "epoch " << state.epoch << '\n'
         << "processed_tokens " << state.processed_tokens << '\n'
         << "generator " << std::quoted(state.data.generator_state) << '\n'
         << "training_cursor " << state.data.training_cursor << '\n'
         << "validation_cursor " << state.data.validation_cursor << '\n'
         << "training_order";
    for (const std::size_t index : state.data.training_order) {
        file << ' ' << index;
    }
    file << "\nvalidation_order";
    for (const std::size_t index : state.data.validation_order) {
        file << ' ' << index;
    }
    file << '\n';
    file.close();
    if (!file) {
        throw std::runtime_error("Failed to write training state: " + path.string());
    }
}

TrainingState load_training_state(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open training state: " + path.string());
    }

    std::string magic;
    int version = 0;
    file >> magic >> version;
    if (magic != "MINI_GPT_TRAINING_STATE" || version != 1) {
        throw std::runtime_error("Invalid training state header: " + path.string());
    }

    TrainingState state;
    std::string key;
    file >> key;
    if (key != "corpus") throw std::runtime_error("Missing corpus in training state");
    file >> std::quoted(state.corpus_path);
    file >> key;
    if (key != "tokenizer") throw std::runtime_error("Missing tokenizer in training state");
    file >> std::quoted(state.tokenizer_path);
    file >> key;
    if (key != "epoch") throw std::runtime_error("Missing epoch in training state");
    file >> state.epoch;
    file >> key;
    if (key != "processed_tokens") throw std::runtime_error("Missing processed token count");
    file >> state.processed_tokens;
    file >> key;
    if (key != "generator") throw std::runtime_error("Missing generator state");
    file >> std::quoted(state.data.generator_state);
    file >> key;
    if (key != "training_cursor") throw std::runtime_error("Missing training cursor");
    file >> state.data.training_cursor;
    file >> key;
    if (key != "validation_cursor") throw std::runtime_error("Missing validation cursor");
    file >> state.data.validation_cursor;
    file >> key;
    if (key != "training_order") throw std::runtime_error("Missing training order");
    std::size_t index = 0;
    while (file >> index) {
        state.data.training_order.push_back(index);
        if (file.peek() == '\n' || file.peek() == '\r' || file.peek() == EOF) break;
    }
    file.clear();
    file >> key;
    if (key != "validation_order") throw std::runtime_error("Missing validation order");
    while (file >> index) {
        state.data.validation_order.push_back(index);
    }
    if (!file.eof()) {
        throw std::runtime_error("Invalid training state data");
    }
    return state;
}

}  // namespace mini_gpt

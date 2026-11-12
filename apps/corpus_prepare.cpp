#include "mini_gpt/data_loader.hpp"
#include "mini_gpt/gpt.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using mini_gpt::BPETokenizer;
using mini_gpt::GPT;
using mini_gpt::TextDataLoader;

struct Options {
    std::filesystem::path corpus;
    std::filesystem::path vocabulary_output = "gongora.bpe";
    std::filesystem::path model_output = "gongora_model.bin";
    std::size_t vocabulary_size = 2048;
    std::size_t embedding_dim = 256;
    std::size_t num_heads = 4;
    std::size_t num_layers = 4;
    std::size_t hidden_dim = 1024;
    std::size_t max_seq_len = 128;
    std::size_t batch_size = 2;
    std::uint32_t seed = 42;
};

std::size_t parse_size(const char* text) {
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != std::string(text).size() || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("Expected a positive integer");
    }
    return static_cast<std::size_t>(value);
}

Options parse_options(int argc, char** argv) {
    if (argc < 2 || argc > 12) {
        throw std::invalid_argument(
            "Uso: corpus_prepare <corpus> [bpe_salida] [modelo_salida] [vocab_size] [embedding] [heads] [layers] [hidden] [max_seq] [batch] [seed]");
    }
    Options options;
    options.corpus = argv[1];
    if (argc >= 3) options.vocabulary_output = argv[2];
    if (argc >= 4) options.model_output = argv[3];
    if (argc >= 5) options.vocabulary_size = parse_size(argv[4]);
    if (argc >= 6) options.embedding_dim = parse_size(argv[5]);
    if (argc >= 7) options.num_heads = parse_size(argv[6]);
    if (argc >= 8) options.num_layers = parse_size(argv[7]);
    if (argc >= 9) options.hidden_dim = parse_size(argv[8]);
    if (argc >= 10) options.max_seq_len = parse_size(argv[9]);
    if (argc >= 11) options.batch_size = parse_size(argv[10]);
    if (argc >= 12) options.seed = static_cast<std::uint32_t>(parse_size(argv[11]));
    if (options.max_seq_len > 256) {
        throw std::invalid_argument("max_seq_len no puede superar 256");
    }
    return options;
}

void require_new_file(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("No se sobrescribira el archivo existente: " + path.string());
    }
}

std::size_t nonempty_lines(const std::string& text) {
    std::size_t count = 0;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t end = text.find('\n', start);
        const std::size_t length = (end == std::string::npos ? text.size() : end) - start;
        if (length != 0) {
            ++count;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        require_new_file(options.vocabulary_output);
        require_new_file(options.model_output);
        const std::string corpus = mini_gpt::read_utf8_file(options.corpus);
        const auto split = mini_gpt::split_corpus_utf8(corpus);

        BPETokenizer tokenizer(options.vocabulary_size);
        tokenizer.train(split.training);
        if (tokenizer.decode(tokenizer.encode(split.training)) != split.training ||
            tokenizer.decode(tokenizer.encode(split.validation)) != split.validation) {
            throw std::runtime_error("Fallo de reconstruccion encode/decode");
        }
        tokenizer.save(options.vocabulary_output);

        TextDataLoader loader(
            tokenizer,
            split.training,
            split.validation,
            options.max_seq_len,
            options.batch_size,
            options.seed);
        GPT model(
            tokenizer.vocab_size(),
            options.embedding_dim,
            options.num_heads,
            options.num_layers,
            options.hidden_dim,
            options.max_seq_len,
            options.seed);
        model.save(options.model_output);

        const std::size_t corpus_tokens = tokenizer.encode(corpus).size();
        const std::size_t lines = std::max<std::size_t>(1, nonempty_lines(corpus));
        const auto parameters = model.parameter_count();
        std::cout << "Vocabulario guardado: " << options.vocabulary_output.string()
                  << "\nModelo nuevo guardado: " << options.model_output.string()
                  << "\nVocabulario: " << tokenizer.vocab_size()
                  << "\nTokens del corpus: " << corpus_tokens
                  << "\nMedia aproximada por linea: "
                  << static_cast<double>(corpus_tokens) / lines
                  << "\nTokens entrenamiento: " << loader.training_token_count()
                  << "\nTokens validacion: " << loader.validation_token_count()
                  << "\nSecuencias entrenamiento: " << loader.training_sample_count()
                  << "\nSecuencias validacion: " << loader.validation_sample_count()
                  << "\nParametros totales: " << parameters.total()
                  << "\n  Embeddings: " << parameters.embeddings
                  << "\n  Bloques: " << parameters.transformer_blocks
                  << "\n  LayerNorm final: " << parameters.final_layer_norm
                  << "\n  Salida: " << parameters.output_projection << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/cli.hpp"
#include "mini_gpt/data_loader.hpp"

#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_help() {
    std::cout
        << "tokenizer_train - Entrena Byte-Level BPE\n\n"
        << "Uso: tokenizer_train --corpus <archivo> [opciones]\n\n"
        << "  --corpus <ruta>          Corpus UTF-8 (obligatorio)\n"
        << "  --vocab-size <N>         Vocabulario máximo (defecto: 2048)\n"
        << "  --output <ruta>          Vocabulario de salida (defecto: tokenizer.bpe)\n"
        << "  --validation-ratio <x>   Fracción reservada, no usada para BPE (defecto: 0.1; 0 desactiva)\n"
        << "  --progress-every <N>     Mostrar progreso cada N merges (defecto: 25)\n"
        << "  --overwrite             Autoriza sobrescribir la salida\n"
        << "  --help                  Muestra esta ayuda\n";
}

std::size_t nonempty_lines(const std::string& text) {
    std::size_t count = 0;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t end = text.find('\n', start);
        const std::size_t length = (end == std::string::npos ? text.size() : end) - start;
        if (length != 0) ++count;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = mini_gpt::cli::collect_arguments(argc, argv);
        if (mini_gpt::cli::has(args, "--help")) {
            print_help();
            return 0;
        }
        mini_gpt::cli::validate(
            args,
            {"--help", "--overwrite"},
            {"--corpus", "--vocab-size", "--output", "--validation-ratio", "--progress-every"});

        const std::filesystem::path corpus_path = mini_gpt::cli::required(args, "--corpus");
        const std::filesystem::path output_path =
            mini_gpt::cli::value(args, "--output", "tokenizer.bpe");
        const std::size_t vocab_size = mini_gpt::cli::size_value(args, "--vocab-size", 2048, 260);
        const float validation_ratio = mini_gpt::cli::float_value(args, "--validation-ratio", 0.1F);
        const std::size_t progress_every = mini_gpt::cli::size_value(args, "--progress-every", 25);
        const bool overwrite = mini_gpt::cli::has(args, "--overwrite");
        if (validation_ratio < 0.0F || validation_ratio >= 1.0F) {
            throw std::invalid_argument("--validation-ratio debe estar entre 0 y 1, sin incluir 1");
        }
        mini_gpt::cli::require_input_file(corpus_path);
        mini_gpt::cli::require_output_file(output_path, overwrite);

        const std::string corpus = mini_gpt::read_utf8_file(corpus_path);
        const auto split = validation_ratio == 0.0F
            ? mini_gpt::CorpusSplit{corpus, ""}
            : mini_gpt::split_corpus_utf8(corpus, validation_ratio);
        std::cout << "Corpus: " << corpus_path.string()
                  << "\nBytes: " << corpus.size()
                  << "\nPuntos de código UTF-8: " << mini_gpt::utf8_codepoint_count(corpus)
                  << "\nCaracteres entrenamiento: " << split.training.size()
                  << "\nCaracteres validación: " << split.validation.size()
                  << "\nIniciando BPE con máximo " << vocab_size << " tokens...\n";

        mini_gpt::BPETokenizer tokenizer(vocab_size);
        tokenizer.train(split.training, [&](std::size_t learned, std::size_t target) {
            if (learned % progress_every == 0 || learned == target) {
                std::cout << "  merges=" << learned << '/' << target
                          << " (" << (target == 0 ? 100.0 : 100.0 * learned / target) << "%)\n";
            }
        });

        if (tokenizer.decode(tokenizer.encode(split.training)) != split.training ||
            tokenizer.decode(tokenizer.encode(split.validation)) != split.validation) {
            throw std::runtime_error("La comprobación encode/decode ha fallado");
        }
        tokenizer.save(output_path);

        const std::size_t train_tokens = tokenizer.encode(split.training).size();
        const std::size_t validation_tokens = tokenizer.encode(split.validation).size();
        const std::size_t total_tokens = train_tokens + validation_tokens;
        const std::size_t lines = std::max<std::size_t>(1, nonempty_lines(corpus));
        std::cout << std::fixed << std::setprecision(4)
                  << "Entrenamiento BPE completado\n"
                  << "Vocabulario aprendido: " << tokenizer.vocab_size()
                  << "\nMerges aprendidos: " << tokenizer.merges().size()
                  << "\nTokens entrenamiento: " << train_tokens
                  << "\nTokens validación: " << validation_tokens
                  << "\nTokens totales: " << total_tokens
                  << "\nMedia aproximada por línea: " << static_cast<double>(total_tokens) / lines
                  << "\nBytes por token: " << static_cast<double>(corpus.size()) / total_tokens
                  << "\nencode/decode: correcto"
                  << "\nVocabulario guardado en: " << output_path.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

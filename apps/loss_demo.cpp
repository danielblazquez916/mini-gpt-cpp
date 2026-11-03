#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/cross_entropy.hpp"
#include "mini_gpt/gpt.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using mini_gpt::BPETokenizer;
using mini_gpt::GPT;
using mini_gpt::SamplingStrategy;
using mini_gpt::Tensor;
using TokenId = GPT::TokenId;

std::string escape_text(std::string_view text) {
    constexpr char hexadecimal[] = "0123456789ABCDEF";
    std::string escaped;
    for (const unsigned char byte : text) {
        switch (byte) {
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            case '\0': escaped += "\\0"; break;
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            default:
                if (byte >= 32 && byte <= 126) {
                    escaped += static_cast<char>(byte);
                } else {
                    escaped += "\\x";
                    escaped += hexadecimal[byte >> 4];
                    escaped += hexadecimal[byte & 0x0F];
                }
        }
    }
    return escaped;
}

std::string read_corpus(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se puede abrir el corpus: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void print_shape(const Tensor& tensor) {
    std::cout << '[';
    for (std::size_t i = 0; i < tensor.shape().size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << tensor.shape()[i];
    }
    std::cout << ']';
}

void print_shift_example(const std::vector<TokenId>& inputs, const std::vector<TokenId>& targets) {
    const std::size_t count = std::min<std::size_t>(5, inputs.size());
    std::cout << "Ejemplo de desplazamiento:\n";
    for (std::size_t i = 0; i < count; ++i) {
        std::cout << "  input=" << inputs[i] << " -> target=" << targets[i] << '\n';
    }
}

void print_predictions(
    const BPETokenizer& tokenizer,
    const Tensor& logits,
    const std::vector<TokenId>& targets) {
    const std::size_t count = std::min<std::size_t>(5, targets.size());
    const std::size_t vocab_size = tokenizer.vocab_size();
    std::mt19937 generator(0);
    std::cout << "Tokens esperados y predichos:\n";
    for (std::size_t position = 0; position < count; ++position) {
        const auto position_logits = logits.data().subspan(position * vocab_size, vocab_size);
        const TokenId predicted = GPT::select_token(
            position_logits, 1.0F, SamplingStrategy::greedy, generator);
        const TokenId expected = targets[position];
        std::cout << "  pos=" << position << " esperado=" << expected << " (\""
                  << escape_text(tokenizer.token(expected)) << "\") predicho=" << predicted
                  << " (\"" << escape_text(tokenizer.token(predicted)) << "\")\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 4) {
        std::cerr << "Uso: " << argv[0] << " [vocabulario.bpe] [modelo.bin] [corpus.txt]\n";
        return 2;
    }

    const std::filesystem::path vocabulary_path = argc >= 2
        ? std::filesystem::path(argv[1])
        : std::filesystem::current_path() / "tokenizer_demo.bpe";
    const std::filesystem::path model_path = argc >= 3
        ? std::filesystem::path(argv[2])
        : std::filesystem::current_path() / "gpt_model.bin";
    const std::filesystem::path corpus_path = argc == 4
        ? std::filesystem::path(argv[3])
        : std::filesystem::path(MINI_GPT_LOSS_CORPUS_PATH);

    try {
        const BPETokenizer tokenizer = BPETokenizer::load(vocabulary_path);
        const GPT model = GPT::load(model_path);
        if (tokenizer.vocab_size() != model.vocab_size()) {
            throw std::invalid_argument("El vocabulario BPE no coincide con el modelo");
        }

        const std::string corpus = read_corpus(corpus_path);
        const auto tokens = tokenizer.encode(corpus);
        auto shifted = mini_gpt::shift_tokens(tokens);
        if (shifted.inputs.size() > model.max_seq_len()) {
            const std::size_t start = shifted.inputs.size() - model.max_seq_len();
            shifted.inputs = std::vector<TokenId>(
                shifted.inputs.begin() + start, shifted.inputs.end());
            shifted.targets = std::vector<TokenId>(
                shifted.targets.begin() + start, shifted.targets.end());
            std::cout << "Evaluando las últimas " << model.max_seq_len()
                      << " posiciones del corpus.\n";
        }

        const Tensor logits = model.forward(shifted.inputs, 1, shifted.inputs.size());
        const float loss = mini_gpt::cross_entropy(logits, shifted.targets);
        std::cout << std::fixed << std::setprecision(6)
                  << "Tokens del corpus: " << tokens.size()
                  << "\nPosiciones evaluadas: " << shifted.inputs.size()
                  << "\nDimensiones de logits: ";
        print_shape(logits);
        std::cout << "\nCross-Entropy media: " << loss
                  << "\nPerplexity: " << mini_gpt::perplexity(loss) << '\n';
        print_shift_example(shifted.inputs, shifted.targets);
        print_predictions(tokenizer, logits, shifted.targets);
        std::cout << "Evaluación completada sin modificar los pesos.\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/embedding.hpp"
#include "mini_gpt/transformer_block.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using mini_gpt::BPETokenizer;
using mini_gpt::Embedding;
using mini_gpt::Tensor;
using mini_gpt::TransformerBlock;
using TokenId = Embedding::TokenId;

constexpr std::size_t demo_num_heads = 4;
constexpr std::uint32_t demo_seed = 42;

class ConsoleUtf8 {
public:
    ConsoleUtf8() {
#ifdef _WIN32
        previous_input_ = GetConsoleCP();
        previous_output_ = GetConsoleOutputCP();
        SetConsoleCP(CP_UTF8);
        SetConsoleOutputCP(CP_UTF8);
#endif
    }

    ~ConsoleUtf8() {
#ifdef _WIN32
        std::cout.flush();
        std::cerr.flush();
        if (previous_input_ != 0) {
            SetConsoleCP(previous_input_);
        }
        if (previous_output_ != 0) {
            SetConsoleOutputCP(previous_output_);
        }
#endif
    }

private:
#ifdef _WIN32
    UINT previous_input_ = 0;
    UINT previous_output_ = 0;
#endif
};

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

void print_vector_prefixes(const Tensor& tensor, const char* label) {
    constexpr std::size_t max_vectors = 3;
    constexpr std::size_t max_values = 8;
    const std::size_t seq_len = tensor.shape()[1];
    const std::size_t embedding_dim = tensor.shape()[2];
    const std::size_t count = std::min(max_vectors, seq_len);

    std::cout << label << ":\n" << std::fixed << std::setprecision(6);
    for (std::size_t position = 0; position < count; ++position) {
        std::cout << "  Posición " << position << ": [";
        const std::size_t value_count = std::min(max_values, embedding_dim);
        const std::size_t offset = position * embedding_dim;
        for (std::size_t i = 0; i < value_count; ++i) {
            if (i != 0) {
                std::cout << ", ";
            }
            std::cout << tensor.data()[offset + i];
        }
        if (value_count < embedding_dim) {
            std::cout << ", ...";
        }
        std::cout << "]\n";
    }
}

void process_text(
    const BPETokenizer& tokenizer,
    const Embedding& embedding,
    const TransformerBlock& block,
    const std::string& text) {
    const std::vector<TokenId> ids = tokenizer.encode(text);
    const Tensor input = embedding.forward(ids, 1, ids.size());
    const Tensor output = block.forward(input);

    std::cout << "\nTexto original: " << text
              << "\nNúmero de tokens: " << ids.size()
              << "\nDimensiones de entrada: ";
    print_shape(input);
    std::cout << "\nDimensiones de salida: ";
    print_shape(output);
    std::cout << '\n';
    print_vector_prefixes(input, "Valores de entrada");
    print_vector_prefixes(output, "Valores de salida");

    const bool modified = !std::equal(
        input.data().begin(), input.data().end(), output.data().begin());
    std::cout << "Representaciones modificadas: " << (modified ? "sí" : "no")
              << "\nDimensiones conservadas: "
              << (input.shape() == output.shape() ? "sí" : "no") << '\n';
}

TransformerBlock prepare_block(
    std::size_t embedding_dim,
    const std::filesystem::path& weights_path) {
    if (std::filesystem::exists(weights_path)) {
        TransformerBlock block = TransformerBlock::load(weights_path);
        if (block.embedding_dim() != embedding_dim) {
            throw std::invalid_argument(
                "La dimensión del bloque no coincide con los embeddings");
        }
        std::cout << "Pesos del Transformer Block cargados: " << weights_path.string() << '\n';
        return block;
    }

    TransformerBlock block(embedding_dim, demo_num_heads, 4 * embedding_dim, demo_seed);
    block.save(weights_path);
    std::cout << "Transformer Block inicializado y guardado: " << weights_path.string() << '\n';
    return block;
}

void run_interactive_demo(
    const BPETokenizer& tokenizer,
    const Embedding& embedding,
    const TransformerBlock& block) {
    std::cout << "Cabezas: " << block.num_heads()
              << "\nDimensión oculta del MLP: " << block.mlp_hidden_dim()
              << "\n\nEscribe una frase. Usa /exit para salir.";
    while (true) {
        std::cout << "\n\n> " << std::flush;
        std::string text;
        if (!std::getline(std::cin, text)) {
            std::cout << "\nEntrada finalizada.\n";
            return;
        }
        if (text == "/exit") {
            return;
        }
        try {
            process_text(tokenizer, embedding, block, text);
        } catch (const std::exception& error) {
            std::cout << "Error: " << error.what() << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    ConsoleUtf8 console_utf8;
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc > 4) {
        std::cerr << "Uso: " << argv[0]
                  << " [vocabulario.bpe] [embeddings.bin] [bloque.bin]\n";
        return 2;
    }

    const std::filesystem::path vocabulary_path = argc >= 2
        ? std::filesystem::path(argv[1])
        : std::filesystem::current_path() / "tokenizer_demo.bpe";
    const std::filesystem::path embedding_path = argc >= 3
        ? std::filesystem::path(argv[2])
        : std::filesystem::current_path() / "embedding_weights.bin";
    const std::filesystem::path block_path = argc == 4
        ? std::filesystem::path(argv[3])
        : std::filesystem::current_path() / "transformer_block.bin";

    try {
        if (!std::filesystem::exists(vocabulary_path) ||
            !std::filesystem::exists(embedding_path)) {
            throw std::runtime_error(
                "Faltan el vocabulario o los embeddings. Ejecuta primero tokenizer_demo y embedding_demo.");
        }

        const BPETokenizer tokenizer = BPETokenizer::load(vocabulary_path);
        const Embedding embedding = Embedding::load(embedding_path);
        if (embedding.vocab_size() != tokenizer.vocab_size()) {
            throw std::invalid_argument(
                "El vocabulario BPE no coincide con la tabla de embeddings");
        }
        std::cout << "Vocabulario y embeddings cargados correctamente.\n";
        const TransformerBlock block = prepare_block(embedding.embedding_dim(), block_path);
        run_interactive_demo(tokenizer, embedding, block);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

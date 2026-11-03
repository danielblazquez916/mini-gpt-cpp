#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/embedding.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using mini_gpt::BPETokenizer;
using mini_gpt::Embedding;
using mini_gpt::Tensor;
using TokenId = Embedding::TokenId;

constexpr std::size_t demo_embedding_dim = 256;
constexpr std::size_t demo_max_seq_len = 1024;
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

void print_ids(const std::vector<TokenId>& ids) {
    std::cout << "IDs: [";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << ids[i];
    }
    std::cout << "]\n";
}

void print_vector_prefixes(const Tensor& embeddings) {
    constexpr std::size_t max_vectors = 3;
    constexpr std::size_t max_values = 8;
    const std::size_t sequence_length = embeddings.shape()[1];
    const std::size_t embedding_dim = embeddings.shape()[2];
    const std::size_t vector_count = std::min(max_vectors, sequence_length);

    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t position = 0; position < vector_count; ++position) {
        std::cout << "  Posición " << position << ": [";
        const std::size_t value_count = std::min(max_values, embedding_dim);
        const std::size_t offset = position * embedding_dim;
        for (std::size_t i = 0; i < value_count; ++i) {
            if (i != 0) {
                std::cout << ", ";
            }
            std::cout << embeddings.data()[offset + i];
        }
        if (value_count < embedding_dim) {
            std::cout << ", ...";
        }
        std::cout << "]\n";
    }
}

bool same_vector(
    const Tensor& embeddings,
    std::size_t first_position,
    std::size_t second_position) {
    const std::size_t embedding_dim = embeddings.shape()[2];
    const auto data = embeddings.data();
    return std::equal(
        data.begin() + first_position * embedding_dim,
        data.begin() + (first_position + 1) * embedding_dim,
        data.begin() + second_position * embedding_dim);
}

void check_repeated_token(
    const std::vector<TokenId>& ids,
    const Tensor& token_embeddings,
    const Tensor& combined_embeddings) {
    std::unordered_map<TokenId, std::size_t> first_occurrence;
    for (std::size_t position = 0; position < ids.size(); ++position) {
        const auto [it, inserted] = first_occurrence.emplace(ids[position], position);
        if (inserted) {
            continue;
        }

        const std::size_t previous = it->second;
        std::cout << "Token repetido ID " << ids[position] << " en posiciones " << previous
                  << " y " << position << ":\n"
                  << "  Mismo embedding antes de posiciones: "
                  << (same_vector(token_embeddings, previous, position) ? "sí" : "no")
                  << "\n  Vectores tras sumar posiciones: "
                  << (same_vector(combined_embeddings, previous, position)
                          ? "iguales"
                          : "diferentes")
                  << '\n';
        return;
    }
    std::cout << "No hay IDs repetidos. Prueba, por ejemplo: <|user|> <|user|>\n";
}

void process_text(
    const BPETokenizer& tokenizer,
    const Embedding& embedding,
    const std::string& text) {
    const std::vector<TokenId> ids = tokenizer.encode(text);
    const Tensor token_embeddings = embedding.token_forward(ids, 1, ids.size());
    const Tensor combined_embeddings = embedding.forward(ids, 1, ids.size());

    std::cout << "\nTexto original: " << text << '\n';
    print_ids(ids);
    std::cout << "Número de tokens: " << ids.size()
              << "\nDimensiones sin posiciones: ";
    print_shape(token_embeddings);
    std::cout << "\nDimensiones con posiciones: ";
    print_shape(combined_embeddings);
    std::cout << "\nPrimeros valores del tensor combinado:\n";
    print_vector_prefixes(combined_embeddings);
    check_repeated_token(ids, token_embeddings, combined_embeddings);
}

Embedding prepare_embeddings(
    std::size_t vocab_size,
    const std::filesystem::path& weights_path) {
    if (std::filesystem::exists(weights_path)) {
        Embedding embedding = Embedding::load(weights_path);
        if (embedding.vocab_size() != vocab_size) {
            throw std::invalid_argument(
                "El vocabulario de los pesos no coincide con el vocabulario BPE");
        }
        std::cout << "Pesos de embeddings cargados: " << weights_path.string() << '\n';
        return embedding;
    }

    Embedding embedding(vocab_size, demo_embedding_dim, demo_max_seq_len, demo_seed);
    embedding.save(weights_path);
    std::cout << "Pesos de embeddings inicializados y guardados: " << weights_path.string()
              << '\n';
    return embedding;
}

void run_interactive_demo(const BPETokenizer& tokenizer, const Embedding& embedding) {
    std::cout << "Dimensión de embedding: " << embedding.embedding_dim()
              << "\nLongitud máxima de secuencia: " << embedding.max_seq_len()
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
            process_text(tokenizer, embedding, text);
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

    if (argc > 3) {
        std::cerr << "Uso: " << argv[0] << " [vocabulario.bpe] [pesos.bin]\n";
        return 2;
    }

    const std::filesystem::path vocabulary_path = argc >= 2
        ? std::filesystem::path(argv[1])
        : std::filesystem::current_path() / "tokenizer_demo.bpe";
    const std::filesystem::path weights_path = argc == 3
        ? std::filesystem::path(argv[2])
        : std::filesystem::current_path() / "embedding_weights.bin";

    try {
        if (!std::filesystem::exists(vocabulary_path)) {
            throw std::runtime_error(
                "No existe el vocabulario BPE. Ejecuta tokenizer_demo primero o indica su ruta.");
        }
        const BPETokenizer tokenizer = BPETokenizer::load(vocabulary_path);
        std::cout << "Vocabulario BPE cargado: " << vocabulary_path.string() << " ("
                  << tokenizer.vocab_size() << " tokens)\n";
        const Embedding embedding = prepare_embeddings(tokenizer.vocab_size(), weights_path);
        run_interactive_demo(tokenizer, embedding);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/causal_self_attention.hpp"
#include "mini_gpt/embedding.hpp"

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

using mini_gpt::AttentionForwardResult;
using mini_gpt::BPETokenizer;
using mini_gpt::CausalSelfAttention;
using mini_gpt::Embedding;
using mini_gpt::Tensor;
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

bool future_probabilities_are_zero(const Tensor& probabilities) {
    const std::size_t batch_size = probabilities.shape()[0];
    const std::size_t num_heads = probabilities.shape()[1];
    const std::size_t seq_len = probabilities.shape()[2];
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t head = 0; head < num_heads; ++head) {
            for (std::size_t query = 0; query < seq_len; ++query) {
                for (std::size_t key = query + 1; key < seq_len; ++key) {
                    if (probabilities.at({batch, head, query, key}) != 0.0F) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

void print_attention_matrix(const Tensor& probabilities, std::size_t head) {
    constexpr std::size_t max_display_positions = 10;
    const std::size_t seq_len = probabilities.shape()[2];
    if (seq_len == 0) {
        std::cout << "No hay matriz de atención para una secuencia vacía.\n";
        return;
    }

    const std::size_t display = std::min(seq_len, max_display_positions);
    std::cout << "Matriz de atención de la cabeza " << head
              << " (filas = queries, columnas = keys)";
    if (display < seq_len) {
        std::cout << ", primeras " << display << " posiciones";
    }
    std::cout << ":\n" << std::fixed << std::setprecision(4);

    std::cout << "  Q\\K";
    for (std::size_t key = 0; key < display; ++key) {
        std::cout << std::setw(9) << key;
    }
    std::cout << '\n';
    for (std::size_t query = 0; query < display; ++query) {
        std::cout << std::setw(5) << query;
        for (std::size_t key = 0; key < display; ++key) {
            std::cout << std::setw(9) << probabilities.at({0, head, query, key});
        }
        std::cout << '\n';
    }
}

void process_text(
    const BPETokenizer& tokenizer,
    const Embedding& embedding,
    const CausalSelfAttention& attention,
    const std::string& text) {
    const std::vector<TokenId> ids = tokenizer.encode(text);
    const Tensor embeddings = embedding.forward(ids, 1, ids.size());
    const AttentionForwardResult result = attention.forward_with_details(embeddings);

    std::cout << "\nTexto original: " << text << '\n';
    print_ids(ids);
    std::cout << "Embeddings combinados: ";
    print_shape(embeddings);
    std::cout << "\nQ: ";
    print_shape(result.queries);
    std::cout << "\nK: ";
    print_shape(result.keys);
    std::cout << "\nV: ";
    print_shape(result.values);
    std::cout << "\nSalida: ";
    print_shape(result.output);
    std::cout << '\n';
    print_attention_matrix(result.probabilities, 0);
    std::cout << "Posiciones futuras con probabilidad cero: "
              << (future_probabilities_are_zero(result.probabilities) ? "sí" : "no")
              << '\n';
}

CausalSelfAttention prepare_attention(
    std::size_t embedding_dim,
    const std::filesystem::path& weights_path) {
    if (std::filesystem::exists(weights_path)) {
        CausalSelfAttention attention = CausalSelfAttention::load(weights_path);
        if (attention.embedding_dim() != embedding_dim) {
            throw std::invalid_argument(
                "La dimensión de los pesos de atención no coincide con los embeddings");
        }
        std::cout << "Pesos de atención cargados: " << weights_path.string() << '\n';
        return attention;
    }

    CausalSelfAttention attention(embedding_dim, demo_num_heads, demo_seed);
    attention.save(weights_path);
    std::cout << "Pesos de atención inicializados y guardados: " << weights_path.string()
              << '\n';
    return attention;
}

void run_interactive_demo(
    const BPETokenizer& tokenizer,
    const Embedding& embedding,
    const CausalSelfAttention& attention) {
    std::cout << "Cabezas: " << attention.num_heads()
              << "\nDimensión por cabeza: " << attention.head_dim()
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
            process_text(tokenizer, embedding, attention, text);
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
                  << " [vocabulario.bpe] [embeddings.bin] [atencion.bin]\n";
        return 2;
    }

    const std::filesystem::path vocabulary_path = argc >= 2
        ? std::filesystem::path(argv[1])
        : std::filesystem::current_path() / "tokenizer_demo.bpe";
    const std::filesystem::path embedding_path = argc >= 3
        ? std::filesystem::path(argv[2])
        : std::filesystem::current_path() / "embedding_weights.bin";
    const std::filesystem::path attention_path = argc == 4
        ? std::filesystem::path(argv[3])
        : std::filesystem::current_path() / "attention_weights.bin";

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
        const CausalSelfAttention attention =
            prepare_attention(embedding.embedding_dim(), attention_path);
        run_interactive_demo(tokenizer, embedding, attention);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

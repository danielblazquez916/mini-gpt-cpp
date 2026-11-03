#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/gpt.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using mini_gpt::BPETokenizer;
using mini_gpt::GPT;
using mini_gpt::GenerationConfig;
using mini_gpt::SamplingStrategy;
using mini_gpt::Tensor;
using TokenId = GPT::TokenId;

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

void print_ids(const std::vector<TokenId>& ids, const char* label) {
    std::cout << label << ": [";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << ids[i];
    }
    std::cout << "]\n";
}

std::size_t parse_token_count(const std::string& text) {
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("El número de tokens debe ser un entero positivo");
    }
    return static_cast<std::size_t>(value);
}

float parse_temperature(const std::string& text) {
    std::size_t consumed = 0;
    const float value = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value) || value <= 0.0F) {
        throw std::invalid_argument("La temperatura debe ser un número positivo");
    }
    return value;
}

GenerationConfig read_generation_config() {
    GenerationConfig config;
    config.max_new_tokens = 8;
    config.stop_condition = mini_gpt::StopCondition::end_of_conversation;
    config.seed = 42;

    std::string line;
    std::cout << "Número máximo de tokens nuevos [8]: " << std::flush;
    if (!std::getline(std::cin, line)) {
        throw std::runtime_error("Entrada finalizada");
    }
    if (!line.empty()) {
        config.max_new_tokens = parse_token_count(line);
    }

    std::cout << "Temperatura [1.0]: " << std::flush;
    if (!std::getline(std::cin, line)) {
        throw std::runtime_error("Entrada finalizada");
    }
    if (!line.empty()) {
        config.temperature = parse_temperature(line);
    }

    std::cout << "Estrategia greedy/sampling [greedy]: " << std::flush;
    if (!std::getline(std::cin, line)) {
        throw std::runtime_error("Entrada finalizada");
    }
    if (line.empty() || line == "greedy") {
        config.strategy = SamplingStrategy::greedy;
    } else if (line == "sampling") {
        config.strategy = SamplingStrategy::sampling;
    } else {
        throw std::invalid_argument("Estrategia desconocida: " + line);
    }
    return config;
}

GPT prepare_model(std::size_t vocab_size, const std::filesystem::path& model_path) {
    if (std::filesystem::exists(model_path)) {
        GPT model = GPT::load(model_path);
        if (model.vocab_size() != vocab_size) {
            throw std::invalid_argument(
                "El vocabulario BPE no coincide con el vocabulario del modelo");
        }
        std::cout << "Modelo GPT cargado: " << model_path.string() << '\n';
        return model;
    }

    GPT model(vocab_size, 256, 4, 4, 1024, 256, 42);
    model.save(model_path);
    std::cout << "Modelo GPT inicializado y guardado: " << model_path.string() << '\n';
    return model;
}

void print_parameter_count(const GPT& model) {
    const auto count = model.parameter_count();
    std::cout << "Parámetros entrenables:\n"
              << "  Embeddings: " << count.embeddings << '\n'
              << "  Transformer Blocks: " << count.transformer_blocks << '\n'
              << "  LayerNorm final: " << count.final_layer_norm << '\n'
              << "  Proyección de salida: " << count.output_projection << '\n'
              << "  Total: " << count.total() << '\n';
}

void print_top_tokens(
    const BPETokenizer& tokenizer,
    const Tensor& logits,
    std::size_t context_size,
    float temperature) {
    const std::size_t vocab_size = tokenizer.vocab_size();
    const auto last_logits = logits.data().subspan(
        (context_size - 1) * vocab_size, vocab_size);
    const auto probabilities = GPT::softmax(last_logits, temperature);
    std::vector<TokenId> ids(vocab_size);
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](TokenId lhs, TokenId rhs) {
        if (probabilities[lhs] != probabilities[rhs]) {
            return probabilities[lhs] > probabilities[rhs];
        }
        return lhs < rhs;
    });

    std::cout << "Cinco tokens más probables:\n" << std::fixed << std::setprecision(6);
    const std::size_t count = std::min<std::size_t>(5, ids.size());
    for (std::size_t i = 0; i < count; ++i) {
        const TokenId id = ids[i];
        std::cout << "  id=" << id << " token=\"" << escape_text(tokenizer.token(id))
                  << "\" probabilidad=" << probabilities[id] << '\n';
    }
}

void process_text(
    const BPETokenizer& tokenizer,
    const GPT& model,
    const GenerationConfig& config,
    const std::string& text) {
    const std::vector<TokenId> prompt = tokenizer.encode(text);
    if (prompt.empty()) {
        throw std::invalid_argument("Introduce un texto con al menos un token");
    }

    std::vector<TokenId> context = prompt;
    if (context.size() > model.max_seq_len()) {
        context.erase(
            context.begin(),
            context.begin() + (context.size() - model.max_seq_len()));
        std::cout << "El contexto se ha limitado a los últimos " << model.max_seq_len()
                  << " tokens.\n";
    }

    const Tensor logits = model.forward(context, 1, context.size());
    std::cout << "\nTexto original: " << text << '\n';
    print_ids(prompt, "IDs del prompt");
    std::cout << "Dimensiones de logits: ";
    print_shape(logits);
    std::cout << '\n';
    print_top_tokens(tokenizer, logits, context.size(), config.temperature);

    const auto generated = model.generate(prompt, config);
    print_ids(generated, "IDs generados");
    const std::string generated_text = tokenizer.decode(generated);
    std::cout << "Texto generado: " << generated_text
              << "\nTexto generado visible: \"" << escape_text(generated_text) << "\"\n";
}

void run_interactive_demo(
    const BPETokenizer& tokenizer,
    const GPT& model,
    const GenerationConfig& config) {
    std::cout << "Los pesos son aleatorios: la salida no será lingüísticamente coherente."
              << "\nEscribe un prompt. Usa /exit para salir.";
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
            process_text(tokenizer, model, config, text);
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
        std::cerr << "Uso: " << argv[0] << " [vocabulario.bpe] [modelo.bin]\n";
        return 2;
    }

    const std::filesystem::path vocabulary_path = argc >= 2
        ? std::filesystem::path(argv[1])
        : std::filesystem::current_path() / "tokenizer_demo.bpe";
    const std::filesystem::path model_path = argc == 3
        ? std::filesystem::path(argv[2])
        : std::filesystem::current_path() / "gpt_model.bin";

    try {
        if (!std::filesystem::exists(vocabulary_path)) {
            throw std::runtime_error(
                "No existe el vocabulario BPE. Ejecuta tokenizer_demo primero o indica su ruta.");
        }
        const BPETokenizer tokenizer = BPETokenizer::load(vocabulary_path);
        const GPT model = prepare_model(tokenizer.vocab_size(), model_path);
        print_parameter_count(model);
        const GenerationConfig config = read_generation_config();
        run_interactive_demo(tokenizer, model, config);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

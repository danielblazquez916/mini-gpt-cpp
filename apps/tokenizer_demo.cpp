#include "mini_gpt/bpe_tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using mini_gpt::BPETokenizer;

constexpr std::size_t demo_vocabulary_size = 300;

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

bool is_valid_utf8(std::string_view text) {
    std::size_t i = 0;
    while (i < text.size()) {
        const auto lead = static_cast<unsigned char>(text[i]);
        if (lead <= 0x7F) {
            ++i;
            continue;
        }

        std::size_t continuation_count = 0;
        std::uint32_t code_point = 0;
        std::uint32_t minimum = 0;
        if (lead >= 0xC2 && lead <= 0xDF) {
            continuation_count = 1;
            code_point = lead & 0x1FU;
            minimum = 0x80;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            continuation_count = 2;
            code_point = lead & 0x0FU;
            minimum = 0x800;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            continuation_count = 3;
            code_point = lead & 0x07U;
            minimum = 0x10000;
        } else {
            return false;
        }

        if (i + continuation_count >= text.size()) {
            return false;
        }
        for (std::size_t j = 1; j <= continuation_count; ++j) {
            const auto continuation = static_cast<unsigned char>(text[i + j]);
            if ((continuation & 0xC0U) != 0x80U) {
                return false;
            }
            code_point = (code_point << 6) | (continuation & 0x3FU);
        }
        if (code_point < minimum || code_point > 0x10FFFF ||
            (code_point >= 0xD800 && code_point <= 0xDFFF)) {
            return false;
        }
        i += continuation_count + 1;
    }
    return true;
}

std::string escape_token(std::string_view token) {
    constexpr char hexadecimal[] = "0123456789ABCDEF";
    const bool valid_utf8 = is_valid_utf8(token);
    std::string escaped;
    for (const unsigned char byte : token) {
        switch (byte) {
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            default:
                if ((byte >= 32 && byte <= 126) || (byte >= 0x80 && valid_utf8)) {
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

std::optional<BPETokenizer> prepare_tokenizer(
    const std::filesystem::path& vocabulary_path,
    const std::filesystem::path& corpus_path) {
    if (std::filesystem::exists(vocabulary_path)) {
        BPETokenizer tokenizer = BPETokenizer::load(vocabulary_path);
        std::cout << "Vocabulario cargado: " << vocabulary_path.string() << " ("
                  << tokenizer.vocab_size() << " tokens)\n";
        return tokenizer;
    }

    std::cout << "No existe un vocabulario en: " << vocabulary_path.string() << '\n'
              << "¿Entrenar uno de demostración con " << corpus_path.string()
              << "? [s/N] " << std::flush;
    std::string answer;
    if (!std::getline(std::cin, answer) || (answer != "s" && answer != "S")) {
        return std::nullopt;
    }

    BPETokenizer tokenizer(demo_vocabulary_size);
    tokenizer.train(read_corpus(corpus_path));
    tokenizer.save(vocabulary_path);
    std::cout << "Vocabulario entrenado y guardado: " << vocabulary_path.string() << " ("
              << tokenizer.vocab_size() << " tokens)\n";
    return tokenizer;
}

void show_analysis(const BPETokenizer& tokenizer, const std::string& text) {
    const std::vector<BPETokenizer::TokenId> ids = tokenizer.encode(text);
    const std::string reconstructed = tokenizer.decode(ids);

    std::cout << "\nTexto original: " << text << "\nTokens:\n";
    if (ids.empty()) {
        std::cout << "  (sin tokens)\n";
    }
    for (std::size_t i = 0; i < ids.size(); ++i) {
        std::cout << "  [" << i << "] id=" << ids[i] << " token=\""
                  << escape_token(tokenizer.token(ids[i])) << "\"\n";
    }

    std::cout << "IDs: [";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << ids[i];
    }
    std::cout << "]\nNúmero total de tokens: " << ids.size()
              << "\nTexto reconstruido: " << reconstructed
              << "\nCoincidencia exacta: " << (reconstructed == text ? "sí" : "no")
              << '\n';
}

void run_interactive_demo(const BPETokenizer& tokenizer) {
    std::cout << "\nEscribe texto para tokenizar. Usa /exit para salir.";
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
        show_analysis(tokenizer, text);
    }
}

}  // namespace

int main(int argc, char** argv) {
    ConsoleUtf8 console_utf8;
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc > 3) {
        std::cerr << "Uso: " << argv[0] << " [vocabulario.bpe] [corpus.txt]\n";
        return 2;
    }

    const std::filesystem::path vocabulary_path = argc >= 2
        ? std::filesystem::path(argv[1])
        : std::filesystem::current_path() / "tokenizer_demo.bpe";
    const std::filesystem::path corpus_path = argc == 3
        ? std::filesystem::path(argv[2])
        : std::filesystem::path(MINI_GPT_DEMO_CORPUS_PATH);

    try {
        auto tokenizer = prepare_tokenizer(vocabulary_path, corpus_path);
        if (!tokenizer) {
            std::cout << "No se entrenó ningún vocabulario. Saliendo.\n";
            return 0;
        }
        run_interactive_demo(*tokenizer);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/chat_mode.hpp"
#include "mini_gpt/cli.hpp"
#include "mini_gpt/gpt.hpp"
#include "mini_gpt/inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
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
        if (previous_input_ != 0) SetConsoleCP(previous_input_);
        if (previous_output_ != 0) SetConsoleOutputCP(previous_output_);
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
        std::size_t continuations = 0;
        std::uint32_t code_point = 0;
        std::uint32_t minimum = 0;
        if (lead >= 0xC2 && lead <= 0xDF) {
            continuations = 1;
            code_point = lead & 0x1FU;
            minimum = 0x80;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            continuations = 2;
            code_point = lead & 0x0FU;
            minimum = 0x800;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            continuations = 3;
            code_point = lead & 0x07U;
            minimum = 0x10000;
        } else {
            return false;
        }
        if (i + continuations >= text.size()) return false;
        for (std::size_t j = 1; j <= continuations; ++j) {
            const auto byte = static_cast<unsigned char>(text[i + j]);
            if ((byte & 0xC0U) != 0x80U) return false;
            code_point = (code_point << 6) | (byte & 0x3FU);
        }
        if (code_point < minimum || code_point > 0x10FFFF ||
            (code_point >= 0xD800 && code_point <= 0xDFFF)) {
            return false;
        }
        i += continuations + 1;
    }
    return true;
}

std::string escape_text(std::string_view text) {
    constexpr char hexadecimal[] = "0123456789ABCDEF";
    const bool valid_utf8 = is_valid_utf8(text);
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

void print_ranked_tokens(
    const mini_gpt::BPETokenizer& tokenizer,
    const std::vector<mini_gpt::TokenProbability>& ranked) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(4);
    for (const auto& item : ranked) {
        std::cout << item.rank << ". \"" << escape_text(tokenizer.token(item.id))
                  << "\" (id=" << item.id << ") " << item.probability * 100.0F << " %\n";
    }
}

void show_top_predictions(
    const mini_gpt::BPETokenizer& tokenizer,
    const mini_gpt::GPT& model,
    const std::vector<mini_gpt::GPT::TokenId>& context,
    std::size_t count,
    bool adjusted,
    const mini_gpt::GenerationConfig& config) {
    if (!adjusted) {
        std::cout << "Predicciones del siguiente token (logits originales del modelo, "
                     "sin temperatura, Top-k ni repetition penalty):\n";
        print_ranked_tokens(
            tokenizer,
            mini_gpt::top_token_probabilities(model, context, count));
        return;
    }

    const mini_gpt::Tensor logits = model.forward(context, 1, context.size());
    const auto last_logits = logits.data().subspan(
        (context.size() - 1) * model.vocab_size(), model.vocab_size());
    std::vector<float> adjusted_logits(last_logits.begin(), last_logits.end());
    mini_gpt::GPT::apply_repetition_penalty(
        context,
        adjusted_logits,
        config.repetition_penalty,
        config.repetition_window);
    const auto probabilities = mini_gpt::GPT::softmax(adjusted_logits, config.temperature);
    std::vector<mini_gpt::TokenProbability> ranked;
    ranked.reserve(probabilities.size());
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        ranked.push_back(mini_gpt::TokenProbability{
            static_cast<std::uint32_t>(i), probabilities[i], 0});
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.probability != rhs.probability) return lhs.probability > rhs.probability;
        return lhs.id < rhs.id;
    });
    for (std::size_t i = 0; i < ranked.size(); ++i) ranked[i].rank = i + 1;
    ranked.resize(count);
    std::cout << "Predicciones del siguiente token (logits ajustados por repetition penalty "
                 "y temperatura, antes de Top-k):\n";
    print_ranked_tokens(tokenizer, ranked);
}

void print_continuation_evaluation(
    const mini_gpt::BPETokenizer& tokenizer,
    const mini_gpt::ContinuationEvaluation& evaluation) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);
    std::cout << "Evaluación determinista de la continuación esperada:\n";
    for (std::size_t i = 0; i < evaluation.tokens.size(); ++i) {
        const auto& item = evaluation.tokens[i];
        std::cout << "  posición " << i << ": \"" << escape_text(tokenizer.token(item.id))
                  << "\" (id=" << item.id << ") probabilidad=" << item.probability
                  << " ranking=" << item.rank << '\n';
    }
    std::cout << "Cross-Entropy media de la continuación: " << evaluation.loss << '\n';
}

void print_help() {
    std::cout
        << "gpt_chat - Generación interactiva con un GPT\n\n"
        << "Uso: gpt_chat --model <checkpoint> --tokenizer <bpe> [opciones]\n\n"
        << "  --model <ruta>          Modelo o checkpoint (obligatorio)\n"
        << "  --tokenizer <ruta>      Vocabulario BPE (obligatorio)\n"
        << "  --strategy <estrategia> greedy o sampling (defecto: greedy)\n"
        << "  --temperature <x>       Temperatura para sampling (defecto: 1.0)\n"
        << "  --max-tokens <N>        Máximo de tokens nuevos por turno (defecto: 16)\n"
        << "  --seed <N>              Semilla base de sampling (defecto: 42)\n"
        << "  --mode <modo>           chat o completion (defecto: chat; continue es alias)\n"
        << "  --top-k <N>             Candidatos para sampling; 0 desactiva (defecto: 0)\n"
        << "  --repetition-penalty <x> Penalización de repetición (defecto: 1.0)\n"
        << "  --repetition-window <N> Tokens recientes penalizados; 0 = todo el contexto\n"
        << "  --show-top <N>          Mostrar N predicciones (defecto: 0)\n"
        << "  --show-adjusted         Usar logits ajustados en vez de originales\n"
        << "  --prompt <texto>        Procesar un prompt una vez y salir\n"
        << "  --expected <texto>      Continuación esperada; evalúa sin generar\n"
        << "  --help                  Muestra esta ayuda\n\n"
        << "chat usa <|user|>, <|assistant|> y se detiene en <|eot|>.\n"
        << "completion genera continuaciones sin añadir tokens especiales.\n"
        << "Comandos interactivos: /reset limpia el contexto; /exit termina.\n";
}

}  // namespace

int main(int argc, char** argv) {
    ConsoleUtf8 console_utf8;
    try {
        const auto args = mini_gpt::cli::collect_arguments(argc, argv);
        if (mini_gpt::cli::has(args, "--help")) {
            print_help();
            return 0;
        }
        mini_gpt::cli::validate(
            args,
            {"--help", "--show-adjusted"},
            {"--model", "--tokenizer", "--strategy", "--temperature", "--max-tokens",
             "--seed", "--mode", "--top-k", "--repetition-penalty", "--repetition-window",
             "--show-top", "--prompt", "--expected"});
        const std::filesystem::path model_path = mini_gpt::cli::required(args, "--model");
        const std::filesystem::path tokenizer_path = mini_gpt::cli::required(args, "--tokenizer");
        const std::string strategy = mini_gpt::cli::value(args, "--strategy", "greedy");
        const mini_gpt::ChatMode mode =
            mini_gpt::parse_chat_mode(mini_gpt::cli::value(args, "--mode", "chat"));
        if (strategy != "greedy" && strategy != "sampling") {
            throw std::invalid_argument("--strategy debe ser greedy o sampling");
        }
        mini_gpt::cli::require_input_file(model_path);
        mini_gpt::cli::require_input_file(tokenizer_path);

        GenerationConfig config;
        config.max_new_tokens = mini_gpt::cli::size_value(args, "--max-tokens", 16);
        config.temperature = mini_gpt::cli::float_value(args, "--temperature", 1.0F);
        config.seed = static_cast<std::uint32_t>(mini_gpt::cli::size_value(args, "--seed", 42, 0));
        config.strategy = strategy == "greedy" ? SamplingStrategy::greedy : SamplingStrategy::sampling;
        config.top_k = mini_gpt::cli::size_value(args, "--top-k", 0, 0);
        config.repetition_penalty =
            mini_gpt::cli::float_value(args, "--repetition-penalty", 1.0F);
        config.repetition_window = mini_gpt::cli::size_value(args, "--repetition-window", 0, 0);
        config.stop_condition = mode == mini_gpt::ChatMode::chat
            ? mini_gpt::StopCondition::end_of_turn
            : mini_gpt::StopCondition::end_of_conversation;
        if (config.temperature <= 0.0F || !std::isfinite(config.temperature)) {
            throw std::invalid_argument("--temperature debe ser positiva");
        }
        if (!std::isfinite(config.repetition_penalty) || config.repetition_penalty < 1.0F) {
            throw std::invalid_argument("--repetition-penalty debe ser al menos 1.0");
        }

        const std::size_t show_top = mini_gpt::cli::size_value(args, "--show-top", 0, 0);
        const bool show_adjusted = mini_gpt::cli::has(args, "--show-adjusted");
        const std::string prompt_argument = mini_gpt::cli::value(args, "--prompt", "");
        const std::string expected = mini_gpt::cli::value(args, "--expected", "");
        if (!expected.empty() && prompt_argument.empty()) {
            throw std::invalid_argument("--expected requiere también --prompt");
        }

        const BPETokenizer tokenizer = BPETokenizer::load(tokenizer_path);
        const GPT model = GPT::load(model_path);
        if (tokenizer.vocab_size() != model.vocab_size()) {
            throw std::invalid_argument("El vocabulario no es compatible con el modelo");
        }
        if (config.top_k > model.vocab_size()) {
            throw std::invalid_argument("--top-k no puede superar el vocabulario");
        }
        if (show_top > model.vocab_size()) {
            throw std::invalid_argument("--show-top no puede superar el vocabulario");
        }

        const auto process_prompt = [&](const std::string& input, std::vector<TokenId>& context) {
            const std::string prompt = mini_gpt::build_chat_prompt(mode, input);
            const auto prompt_ids = tokenizer.encode(prompt);
            context.insert(context.end(), prompt_ids.begin(), prompt_ids.end());
            if (context.size() > model.max_seq_len()) {
                context.erase(context.begin(), context.begin() + (context.size() - model.max_seq_len()));
            }
            if (show_top != 0) {
                show_top_predictions(
                    tokenizer, model, context, show_top, show_adjusted, config);
            }
            if (!expected.empty()) {
                const auto expected_ids = tokenizer.encode(expected);
                print_continuation_evaluation(
                    tokenizer,
                    mini_gpt::evaluate_continuation(model, context, expected_ids));
                return std::vector<TokenId>{};
            }
            const auto generated = model.generate(context, config);
            const std::string text = tokenizer.decode(generated);
            std::cout << (mode == mini_gpt::ChatMode::chat ? "asistente> " : "continuación> ")
                      << text << '\n';
            return generated;
        };

        if (!prompt_argument.empty()) {
            std::vector<TokenId> context;
            static_cast<void>(process_prompt(prompt_argument, context));
            return 0;
        }

        std::vector<TokenId> context;
        std::size_t turn = 0;
        std::cout << "Modelo: " << model_path.string()
                  << "\nModo: " << (mode == mini_gpt::ChatMode::chat ? "chat" : "completion")
                  << "; estrategia: " << strategy
                  << "\nEscribe /reset para limpiar el contexto o /exit para salir.";
        while (true) {
            std::cout << (mode == mini_gpt::ChatMode::chat ? "\n\nusuario> " : "\n\ntexto> ")
                      << std::flush;
            std::string input;
            if (!std::getline(std::cin, input)) {
                std::cout << "\nEntrada finalizada.\n";
                break;
            }
            if (input == "/exit") break;
            if (input == "/reset") {
                context.clear();
                std::cout << "Contexto reiniciado.\n";
                continue;
            }
            if (input.empty() && context.empty()) {
                std::cout << "Introduce texto para iniciar el contexto.\n";
                continue;
            }

            config.seed = static_cast<std::uint32_t>(
                mini_gpt::cli::size_value(args, "--seed", 42, 0) + turn);
            const auto generated = process_prompt(input, context);
            context.insert(context.end(), generated.begin(), generated.end());
            if (context.size() > model.max_seq_len()) {
                context.erase(context.begin(), context.begin() + (context.size() - model.max_seq_len()));
            }
            ++turn;
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

#include "mini_gpt/adamw.hpp"
#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/cross_entropy.hpp"
#include "mini_gpt/gpt.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mini_gpt::AdamW;
using mini_gpt::AdamWConfig;
using mini_gpt::BPETokenizer;
using mini_gpt::GPT;
using TokenId = GPT::TokenId;

struct TrainingOptions {
    std::filesystem::path vocabulary_path = "tokenizer_demo.bpe";
    std::filesystem::path model_path = "gpt_model.bin";
    std::filesystem::path output_path = "gpt_model_trained.bin";
    std::filesystem::path corpus_path = MINI_GPT_TRAINING_CORPUS_PATH;
    std::size_t iterations = 100;
    AdamWConfig optimizer;
    double clipping_norm = 1.0;
};

std::string read_corpus(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se puede abrir el corpus: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::size_t parse_iterations(const std::string& text) {
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("Las iteraciones deben ser un entero positivo");
    }
    return static_cast<std::size_t>(value);
}

float parse_float(const std::string& text) {
    std::size_t consumed = 0;
    const float value = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("Valor numérico inválido: " + text);
    }
    return value;
}

TrainingOptions parse_options(int argc, char** argv) {
    if (argc > 11) {
        throw std::invalid_argument(
            "Uso: training_demo [vocabulario] [modelo] [salida] [iteraciones] [lr] [beta1] [beta2] [epsilon] [weight_decay] [clip]");
    }
    TrainingOptions options;
    if (argc >= 2) options.vocabulary_path = argv[1];
    if (argc >= 3) options.model_path = argv[2];
    if (argc >= 4) options.output_path = argv[3];
    if (argc >= 5) options.iterations = parse_iterations(argv[4]);
    if (argc >= 6) options.optimizer.learning_rate = parse_float(argv[5]);
    if (argc >= 7) options.optimizer.beta1 = parse_float(argv[6]);
    if (argc >= 8) options.optimizer.beta2 = parse_float(argv[7]);
    if (argc >= 9) options.optimizer.epsilon = parse_float(argv[8]);
    if (argc >= 10) options.optimizer.weight_decay = parse_float(argv[9]);
    if (argc >= 11) options.clipping_norm = parse_float(argv[10]);
    if (!std::isfinite(options.clipping_norm) || options.clipping_norm <= 0.0) {
        throw std::invalid_argument("La norma de clipping debe ser positiva");
    }
    return options;
}

std::array<float, 6> representative_parameters(const GPT& model) {
    return {
        model.embedding().token_embeddings().at({0, 0}),
        model.embedding().positional_embeddings().at({0, 0}),
        model.blocks()[0].attention().q_weights().at({0, 0}),
        model.blocks()[0].fc1_weights().at({0, 0}),
        model.final_norm().gamma().at({0}),
        model.output_weights().at({0, 0})};
}

float evaluate_loss(
    const GPT& model,
    const std::vector<TokenId>& inputs,
    const std::vector<TokenId>& targets) {
    return mini_gpt::cross_entropy(model.forward(inputs, 1, inputs.size()), targets);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        TrainingOptions options = parse_options(argc, argv);
        const BPETokenizer tokenizer = BPETokenizer::load(options.vocabulary_path);
        GPT model = GPT::load(options.model_path);
        if (tokenizer.vocab_size() != model.vocab_size()) {
            throw std::invalid_argument("El vocabulario BPE no coincide con el modelo");
        }

        auto shifted = mini_gpt::shift_tokens(tokenizer.encode(read_corpus(options.corpus_path)));
        if (shifted.inputs.size() > model.max_seq_len()) {
            const std::size_t start = shifted.inputs.size() - model.max_seq_len();
            shifted.inputs = std::vector<TokenId>(
                shifted.inputs.begin() + start, shifted.inputs.end());
            shifted.targets = std::vector<TokenId>(
                shifted.targets.begin() + start, shifted.targets.end());
        }

        const float initial_loss = evaluate_loss(model, shifted.inputs, shifted.targets);
        const auto initial_parameters = representative_parameters(model);
        AdamW optimizer(model, options.optimizer);
        const std::size_t print_interval = std::max<std::size_t>(1, options.iterations / 10);
        std::cout << std::fixed << std::setprecision(6)
                  << "Configuración: iteraciones=" << options.iterations
                  << " lr=" << options.optimizer.learning_rate
                  << " beta1=" << options.optimizer.beta1
                  << " beta2=" << options.optimizer.beta2
                  << " epsilon=" << options.optimizer.epsilon
                  << " weight_decay=" << options.optimizer.weight_decay
                  << " clip=" << options.clipping_norm << '\n'
                  << "Loss inicial: " << initial_loss
                  << "\nPerplexity inicial: " << mini_gpt::perplexity(initial_loss) << '\n';

        for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
            model.zero_grad();
            const float loss = model.backward(shifted.inputs, 1, shifted.inputs.size(), shifted.targets);
            const double gradient_norm = mini_gpt::clip_gradients(
                optimizer.parameters(), options.clipping_norm);
            if (!std::isfinite(loss) || !std::isfinite(gradient_norm)) {
                throw std::runtime_error("Entrenamiento detenido por loss o gradiente no finito");
            }
            optimizer.step();

            const std::size_t completed = iteration + 1;
            if (completed % print_interval == 0 || completed == 1 || completed == options.iterations) {
                std::cout << "Paso " << completed << ": loss=" << loss
                          << " perplexity=" << mini_gpt::perplexity(loss)
                          << " lr=" << options.optimizer.learning_rate
                          << " norma_gradiente=" << gradient_norm << '\n';
            }
        }

        const float final_loss = evaluate_loss(model, shifted.inputs, shifted.targets);
        const auto final_parameters = representative_parameters(model);
        if (!std::isfinite(final_loss)) {
            throw std::runtime_error("Loss final no finita");
        }
        if (final_parameters == initial_parameters) {
            throw std::runtime_error("Los parámetros no cambiaron durante el entrenamiento");
        }

        std::cout << "Loss final: " << final_loss
                  << "\nPerplexity final: " << mini_gpt::perplexity(final_loss)
                  << "\nReducción de loss: " << initial_loss - final_loss
                  << "\nCambio en parámetros representativos: confirmado\n";
        model.save(options.output_path);

        GPT loaded = GPT::load(options.output_path);
        const float loaded_loss = evaluate_loss(loaded, shifted.inputs, shifted.targets);
        if (loaded_loss != final_loss) {
            throw std::runtime_error("La recarga del modelo entrenado alteró la loss");
        }
        std::cout << "Modelo entrenado guardado y cargado: " << options.output_path.string()
                  << "\nLoss verificada tras recargar: " << loaded_loss << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

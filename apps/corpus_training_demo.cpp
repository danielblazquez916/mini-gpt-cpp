#include "mini_gpt/adamw.hpp"
#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/cross_entropy.hpp"
#include "mini_gpt/data_loader.hpp"
#include "mini_gpt/gpt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using mini_gpt::AdamW;
using mini_gpt::AdamWConfig;
using mini_gpt::BPETokenizer;
using mini_gpt::GPT;
using mini_gpt::SequenceBatch;
using mini_gpt::Tensor;
using mini_gpt::TextDataLoader;
using TokenId = GPT::TokenId;

struct Options {
    std::filesystem::path corpus;
    std::filesystem::path vocabulary = "gongora.bpe";
    std::filesystem::path model = "gongora_model.bin";
    std::filesystem::path optimizer = "gongora_adamw.bin";
    std::filesystem::path output_model = "gongora_checkpoint.bin";
    std::filesystem::path output_optimizer = "gongora_checkpoint_adamw.bin";
    std::size_t steps = 20;
    std::size_t epochs = 1;
    std::size_t seq_len = 128;
    std::size_t batch_size = 2;
    std::size_t checkpoint_every = 0;
    std::uint32_t seed = 42;
    bool resume = false;
    AdamWConfig optimizer_config;
    double clipping_norm = 1.0;
};

std::size_t parse_size(const std::string& text) {
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("Entero positivo inválido: " + text);
    }
    return static_cast<std::size_t>(value);
}

float parse_float(const std::string& text) {
    std::size_t consumed = 0;
    const float value = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("Número inválido: " + text);
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    Options options;
    options.optimizer_config.weight_decay = 0.01F;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--resume") {
            options.resume = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("Falta valor para " + key);
        }
        const std::string value = argv[++i];
        if (key == "--corpus") options.corpus = value;
        else if (key == "--vocab") options.vocabulary = value;
        else if (key == "--model") options.model = value;
        else if (key == "--optimizer") options.optimizer = value;
        else if (key == "--output-model") options.output_model = value;
        else if (key == "--output-optimizer") options.output_optimizer = value;
        else if (key == "--steps") options.steps = parse_size(value);
        else if (key == "--epochs") options.epochs = parse_size(value);
        else if (key == "--seq-len") options.seq_len = parse_size(value);
        else if (key == "--batch-size") options.batch_size = parse_size(value);
        else if (key == "--checkpoint-every") options.checkpoint_every = parse_size(value);
        else if (key == "--seed") options.seed = static_cast<std::uint32_t>(parse_size(value));
        else if (key == "--lr") options.optimizer_config.learning_rate = parse_float(value);
        else if (key == "--beta1") options.optimizer_config.beta1 = parse_float(value);
        else if (key == "--beta2") options.optimizer_config.beta2 = parse_float(value);
        else if (key == "--epsilon") options.optimizer_config.epsilon = parse_float(value);
        else if (key == "--weight-decay") options.optimizer_config.weight_decay = parse_float(value);
        else if (key == "--clip") options.clipping_norm = parse_float(value);
        else throw std::invalid_argument("Opción desconocida: " + key);
    }
    if (options.corpus.empty()) {
        throw std::invalid_argument("Debe indicarse --corpus <ruta>");
    }
    if (!std::isfinite(options.clipping_norm) || options.clipping_norm <= 0.0) {
        throw std::invalid_argument("Clipping debe ser positivo");
    }
    return options;
}

double evaluate(TextDataLoader& loader, GPT& model, bool validation) {
    if (validation) loader.reset_validation();
    else loader.reset_epoch();

    double loss_sum = 0.0;
    std::size_t tokens = 0;
    while (true) {
        auto batch = validation ? loader.next_validation_batch() : loader.next_batch();
        if (!batch) break;
        const Tensor logits = model.forward(batch->inputs, batch->batch_size, batch->seq_len);
        const float loss = mini_gpt::cross_entropy(logits, batch->targets, batch->mask);
        loss_sum += static_cast<double>(loss) * batch->real_tokens;
        tokens += batch->real_tokens;
    }
    if (tokens == 0) {
        throw std::runtime_error("No hay posiciones válidas para evaluar");
    }
    return loss_sum / tokens;
}

void check_finite(double loss, double norm) {
    if (!std::isfinite(loss) || !std::isfinite(norm)) {
        throw std::runtime_error("Entrenamiento detenido por valores no finitos");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (!options.resume && std::filesystem::exists(options.output_model)) {
            throw std::runtime_error("No se sobrescribirá el checkpoint existente");
        }

        const std::string corpus = mini_gpt::read_utf8_file(options.corpus);
        const auto split = mini_gpt::split_corpus_utf8(corpus);
        const BPETokenizer tokenizer = BPETokenizer::load(options.vocabulary);
        GPT model = GPT::load(options.model);
        if (tokenizer.vocab_size() != model.vocab_size()) {
            throw std::invalid_argument("El vocabulario BPE no coincide con el modelo");
        }
        AdamW optimizer = options.resume
            ? AdamW::load(options.optimizer, model)
            : AdamW(model, options.optimizer_config);
        TextDataLoader loader(
            tokenizer,
            split.training,
            split.validation,
            options.seq_len,
            options.batch_size,
            options.seed);

        const double initial_validation = evaluate(loader, model, true);
        const float initial_parameter = model.output_weights().at({0, 0});
        const auto parameters = model.parameter_count();
        std::cout << std::fixed << std::setprecision(6)
                  << "Parámetros: " << parameters.total()
                  << "\nValidation Loss inicial: " << initial_validation
                  << "\nPerplexity inicial: " << mini_gpt::perplexity(initial_validation)
                  << "\nPasos máximos: " << options.steps << "; épocas máximas: " << options.epochs
                  << "\nLearning rate: " << optimizer.config().learning_rate
                  << "; weight decay: " << optimizer.config().weight_decay
                  << "; batch: " << options.batch_size << "; contexto: " << options.seq_len
                  << '\n';

        const auto start_time = std::chrono::steady_clock::now();
        std::size_t completed = 0;
        std::size_t processed_tokens = 0;
        std::size_t epoch = 0;
        const std::size_t log_interval = std::max<std::size_t>(1, options.steps / 10);
        while (completed < options.steps && epoch < options.epochs) {
            ++epoch;
            loader.reset_epoch();
            while (completed < options.steps) {
                auto batch = loader.next_batch();
                if (!batch) break;

                const auto step_start = std::chrono::steady_clock::now();
                model.zero_grad();
                const float loss = model.backward(
                    batch->inputs,
                    batch->batch_size,
                    batch->seq_len,
                    batch->targets,
                    batch->mask);
                const double norm = mini_gpt::clip_gradients(
                    optimizer.parameters(), options.clipping_norm);
                check_finite(loss, norm);
                optimizer.step();
                ++completed;
                processed_tokens += batch->real_tokens;
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - step_start).count();
                const double tokens_per_second = batch->real_tokens / seconds;

                if (completed % log_interval == 0 || completed == 1 || completed == options.steps) {
                    std::cout << "Época " << epoch << " paso " << completed
                              << ": loss=" << loss
                              << " perplexity=" << mini_gpt::perplexity(loss)
                              << " lr=" << optimizer.config().learning_rate
                              << " norma=" << norm
                              << " tiempo=" << seconds << "s"
                              << " tokens/s=" << tokens_per_second << '\n';
                }
                if (options.checkpoint_every != 0 && completed % options.checkpoint_every == 0) {
                    model.save(options.output_model);
                    optimizer.save(options.output_optimizer);
                }
            }
        }

        const double training_loss = evaluate(loader, model, false);
        const double validation_loss = evaluate(loader, model, true);
        check_finite(training_loss, validation_loss);
        if (initial_parameter == model.output_weights().at({0, 0})) {
            throw std::runtime_error("El parámetro representativo no cambió");
        }
        model.save(options.output_model);
        optimizer.save(options.output_optimizer);
        const double total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();

        std::cout << "Training Loss final: " << training_loss
                  << "\nValidation Loss final: " << validation_loss
                  << "\nTraining Perplexity: " << mini_gpt::perplexity(training_loss)
                  << "\nValidation Perplexity: " << mini_gpt::perplexity(validation_loss)
                  << "\nPasos completados: " << completed
                  << "\nTokens procesados: " << processed_tokens
                  << "\nTiempo total: " << total_seconds << "s"
                  << "\nTokens/s medio: " << processed_tokens / total_seconds
                  << "\nCambio de pesos: confirmado\n";

        GPT loaded = GPT::load(options.output_model);
        AdamW loaded_optimizer = AdamW::load(options.output_optimizer, loaded);
        const double loaded_validation = evaluate(loader, loaded, true);
        if (loaded_validation != validation_loss || loaded_optimizer.step_count() != optimizer.step_count()) {
            throw std::runtime_error("La restauración del checkpoint no coincide");
        }
        std::cout << "Checkpoint restaurado correctamente. Validation Loss: " << loaded_validation
                  << "\nCheckpoint: " << options.output_model.string()
                  << "\nAdamW: " << options.output_optimizer.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

#include "mini_gpt/adamw.hpp"
#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/cli.hpp"
#include "mini_gpt/cross_entropy.hpp"
#include "mini_gpt/data_loader.hpp"
#include "mini_gpt/gpt.hpp"
#include "mini_gpt/training_state.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using mini_gpt::AdamW;
using mini_gpt::AdamWConfig;
using mini_gpt::BPETokenizer;
using mini_gpt::GPT;
using mini_gpt::Tensor;
using mini_gpt::TextDataLoader;
using mini_gpt::TrainingState;

volatile std::sig_atomic_t stop_requested = 0;

void handle_interrupt(int) {
    stop_requested = 1;
}

struct CheckpointPaths {
    std::filesystem::path model;
    std::filesystem::path optimizer;
    std::filesystem::path state;
};

CheckpointPaths derived_paths(const std::filesystem::path& model) {
    const auto parent = model.parent_path();
    const auto stem = model.stem().string();
    const auto extension = model.extension().string();
    return CheckpointPaths{
        model,
        parent / (stem + "_adamw" + extension),
        model.string() + ".data"};
}

CheckpointPaths next_paths(const std::filesystem::path& resume) {
    const auto parent = resume.parent_path();
    const auto stem = resume.stem().string();
    const auto extension = resume.extension().string();
    return derived_paths(parent / (stem + "_next" + extension));
}

std::string normalized_path(const std::filesystem::path& path) {
    return std::filesystem::absolute(path).lexically_normal().string();
}

void print_help() {
    std::cout
        << "gpt_train - Entrena o reanuda un GPT\n\n"
        << "Desde cero:\n"
        << "  gpt_train --corpus <ruta> --tokenizer <bpe> [opciones]\n"
        << "Reanudar:\n"
        << "  gpt_train --resume <checkpoint> --steps <N> [opciones]\n\n"
        << "Datos:\n"
        << "  --corpus <ruta>          Corpus UTF-8\n"
        << "  --tokenizer <ruta>       Vocabulario BPE\n"
        << "  --resume <ruta>          Checkpoint a continuar\n"
        << "  --steps <N>              Pasos adicionales (defecto: 20)\n"
        << "  --batch-size <N>         Tamaño de batch (defecto: 2)\n"
        << "  --context-length <N>     Tokens por secuencia (defecto: 128)\n"
        << "  --seed <N>               Semilla de datos (defecto: 42)\n\n"
        << "Arquitectura, solo desde cero:\n"
        << "  --embedding-dim <N>      Defecto: 256\n"
        << "  --num-heads <N>          Defecto: 4\n"
        << "  --num-layers <N>         Defecto: 4\n"
        << "  --mlp-hidden-dim <N>     Defecto: 1024\n"
        << "  --max-seq-len <N>        Defecto: 128; máximo: 256\n\n"
        << "Optimización:\n"
        << "  --lr <x>                 Defecto: 0.0003\n"
        << "  --beta1 <x>              Defecto: 0.9\n"
        << "  --beta2 <x>              Defecto: 0.999\n"
        << "  --epsilon <x>            Defecto: 1e-8\n"
        << "  --weight-decay <x>       Defecto: 0.01\n"
        << "  --clip <x>               Norma global máxima (defecto: 1.0)\n\n"
        << "Registro y checkpoints:\n"
        << "  --save-every <N>         Guardar cada N pasos; 0 desactiva (defecto: 50)\n"
        << "  --eval-every <N>         Evaluar cada N pasos; 0 desactiva (defecto: 10)\n"
        << "  --validation-samples <N> Secuencias por evaluación (defecto: 16)\n"
        << "  --log-every <N>          Registrar cada N pasos (defecto: 1)\n"
        << "  --output-model <ruta>    Modelo de salida\n"
        << "  --output-optimizer <ruta> Estado AdamW de salida\n"
        << "  --output-state <ruta>    Estado de datos de salida\n"
        << "  --optimizer <ruta>       AdamW de entrada al reanudar\n"
        << "  --overwrite             Autoriza sobrescribir salidas\n"
        << "  --help                  Muestra esta ayuda\n\n"
        << "Por defecto no se sobrescribe el checkpoint de entrada: se crea *_next.bin.\n"
        << "Ctrl+C guarda un checkpoint recuperable.\n";
}

double evaluate_subset(TextDataLoader& loader, GPT& model, std::size_t max_samples) {
    loader.reset_validation();
    double loss_sum = 0.0;
    std::size_t tokens = 0;
    std::size_t samples = 0;
    while (samples < max_samples) {
        auto batch = loader.next_validation_batch();
        if (!batch) break;
        const Tensor logits = model.forward(batch->inputs, batch->batch_size, batch->seq_len);
        const float loss = mini_gpt::cross_entropy(logits, batch->targets, batch->mask);
        loss_sum += static_cast<double>(loss) * batch->real_tokens;
        tokens += batch->real_tokens;
        samples += batch->batch_size;
    }
    if (tokens == 0) {
        throw std::runtime_error("No hay muestras de validación disponibles");
    }
    return loss_sum / tokens;
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
            {"--corpus", "--tokenizer", "--resume", "--steps", "--batch-size",
             "--context-length", "--seed", "--embedding-dim", "--num-heads",
             "--num-layers", "--mlp-hidden-dim", "--max-seq-len", "--lr", "--beta1",
             "--beta2", "--epsilon", "--weight-decay", "--clip", "--save-every",
             "--eval-every", "--validation-samples", "--log-every", "--output-model",
             "--output-optimizer", "--output-state", "--optimizer"});

        const std::string resume = mini_gpt::cli::value(args, "--resume", "");
        const bool is_resume = !resume.empty();
        const bool overwrite = mini_gpt::cli::has(args, "--overwrite");
        const std::size_t additional_steps = mini_gpt::cli::size_value(args, "--steps", 20);
        const std::size_t batch_size = mini_gpt::cli::size_value(args, "--batch-size", 2);
        const std::size_t context_length = mini_gpt::cli::size_value(args, "--context-length", 128);
        const std::size_t save_every = mini_gpt::cli::size_value(args, "--save-every", 50, 0);
        const std::size_t eval_every = mini_gpt::cli::size_value(args, "--eval-every", 10, 0);
        const std::size_t validation_samples = mini_gpt::cli::size_value(args, "--validation-samples", 16);
        const std::size_t log_every = mini_gpt::cli::size_value(args, "--log-every", 1);
        const std::uint32_t seed = static_cast<std::uint32_t>(mini_gpt::cli::size_value(args, "--seed", 42, 0));
        const double clipping = mini_gpt::cli::float_value(args, "--clip", 1.0F);
        if (!std::isfinite(clipping) || clipping <= 0.0) {
            throw std::invalid_argument("--clip debe ser positivo");
        }

        CheckpointPaths output = is_resume
            ? next_paths(resume)
            : derived_paths("gpt_checkpoint.bin");
        output.model = mini_gpt::cli::value(args, "--output-model", output.model.string());
        output.optimizer = mini_gpt::cli::value(args, "--output-optimizer", output.optimizer.string());
        output.state = mini_gpt::cli::value(args, "--output-state", output.state.string());
        mini_gpt::cli::require_output_file(output.model, overwrite);
        mini_gpt::cli::require_output_file(output.optimizer, overwrite);
        mini_gpt::cli::require_output_file(output.state, overwrite);

        std::optional<TrainingState> previous;
        if (is_resume && std::filesystem::exists(std::filesystem::path(resume + ".data"))) {
            previous = mini_gpt::load_training_state(resume + ".data");
        }
        const std::string corpus_arg = mini_gpt::cli::value(args, "--corpus", "");
        const std::string tokenizer_arg = mini_gpt::cli::value(args, "--tokenizer", "");
        const std::string corpus_path = !corpus_arg.empty() ? corpus_arg
            : previous ? previous->corpus_path
                       : "";
        const std::string tokenizer_path = !tokenizer_arg.empty() ? tokenizer_arg
            : previous ? previous->tokenizer_path
                       : "";
        if (corpus_path.empty() || tokenizer_path.empty()) {
            throw std::invalid_argument("Debes indicar --corpus y --tokenizer, o usar un checkpoint con metadatos");
        }
        mini_gpt::cli::require_input_file(corpus_path);
        mini_gpt::cli::require_input_file(tokenizer_path);

        AdamWConfig optimizer_config;
        optimizer_config.learning_rate = mini_gpt::cli::float_value(args, "--lr", 0.0003F);
        optimizer_config.beta1 = mini_gpt::cli::float_value(args, "--beta1", 0.9F);
        optimizer_config.beta2 = mini_gpt::cli::float_value(args, "--beta2", 0.999F);
        optimizer_config.epsilon = mini_gpt::cli::float_value(args, "--epsilon", 1.0e-8F);
        optimizer_config.weight_decay = mini_gpt::cli::float_value(args, "--weight-decay", 0.01F);

        const BPETokenizer tokenizer = BPETokenizer::load(tokenizer_path);
        GPT model = is_resume
            ? GPT::load(resume)
            : GPT(
                tokenizer.vocab_size(),
                mini_gpt::cli::size_value(args, "--embedding-dim", 256),
                mini_gpt::cli::size_value(args, "--num-heads", 4),
                mini_gpt::cli::size_value(args, "--num-layers", 4),
                mini_gpt::cli::size_value(args, "--mlp-hidden-dim", 1024),
                mini_gpt::cli::size_value(args, "--max-seq-len", 128),
                seed);
        if (tokenizer.vocab_size() != model.vocab_size()) {
            throw std::invalid_argument("El vocabulario no es compatible con el modelo");
        }
        if (context_length > model.max_seq_len() || model.max_seq_len() > 256) {
            throw std::invalid_argument("El contexto excede max_seq_len del modelo o el máximo 256");
        }

        AdamW optimizer = is_resume
            ? AdamW::load(
                mini_gpt::cli::value(
                    args, "--optimizer", derived_paths(resume).optimizer.string()),
                model)
            : AdamW(model, optimizer_config);
        if (is_resume) {
            mini_gpt::cli::require_input_file(
                mini_gpt::cli::value(args, "--optimizer", derived_paths(resume).optimizer.string()));
        }

        const std::string corpus = mini_gpt::read_utf8_file(corpus_path);
        const auto split = mini_gpt::split_corpus_utf8(corpus);
        TextDataLoader loader(tokenizer, split.training, split.validation, context_length, batch_size, seed);
        if (loader.training_sample_count() == 0) {
            throw std::runtime_error("El corpus no contiene suficientes tokens para entrenar");
        }

        const bool same_dataset = previous &&
            normalized_path(previous->corpus_path) == normalized_path(corpus_path) &&
            normalized_path(previous->tokenizer_path) == normalized_path(tokenizer_path);
        if (same_dataset) {
            try {
                loader.restore_state(previous->data);
            } catch (const std::invalid_argument&) {
                std::cout << "El estado de datos no es compatible; se inicia un nuevo muestreo.\n";
            }
        } else if (previous) {
            std::cout << "Corpus o vocabulario cambiado: se conservan pesos y AdamW, pero se reinicia el muestreo.\n";
        }

        std::size_t epoch = same_dataset ? previous->epoch : 1;
        std::size_t processed_tokens = same_dataset ? previous->processed_tokens : 0;
        std::size_t session_tokens = 0;
        std::size_t completed = 0;
        float last_loss = 0.0F;
        const float initial_parameter = model.output_weights().at({0, 0});
        const auto start_time = std::chrono::steady_clock::now();
        std::signal(SIGINT, handle_interrupt);
        stop_requested = 0;

        std::cout << std::fixed << std::setprecision(6)
                  << "Modo: " << (is_resume ? "reanudación" : "desde cero")
                  << "\nVocabulario: " << tokenizer.vocab_size()
                  << "\nParámetros: " << model.parameter_count().total()
                  << "\nPasos adicionales: " << additional_steps
                  << "\nBatch: " << batch_size << "; contexto: " << context_length
                  << "\nLearning rate: " << optimizer.config().learning_rate
                  << "\nPaso inicial de AdamW: " << optimizer.step_count() << '\n';
        if (eval_every != 0) {
            const double initial_validation = evaluate_subset(loader, model, validation_samples);
            std::cout << "Validation Loss inicial: " << initial_validation
                      << "\nPerplexity inicial: " << mini_gpt::perplexity(initial_validation) << '\n';
        }

        while (completed < additional_steps && stop_requested == 0) {
            auto batch = loader.next_batch();
            if (!batch) {
                ++epoch;
                loader.reset_epoch();
                batch = loader.next_batch();
                if (!batch) {
                    throw std::runtime_error("No hay más batches de entrenamiento");
                }
            }

            const auto step_start = std::chrono::steady_clock::now();
            model.zero_grad();
            const float loss = model.backward(
                batch->inputs, batch->batch_size, batch->seq_len, batch->targets, batch->mask);
            const double norm = mini_gpt::clip_gradients(optimizer.parameters(), clipping);
            if (!std::isfinite(loss) || !std::isfinite(norm)) {
                throw std::runtime_error("Entrenamiento detenido por valores no finitos");
            }
            optimizer.step();
            ++completed;
            processed_tokens += batch->real_tokens;
            session_tokens += batch->real_tokens;
            last_loss = loss;
            const double step_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - step_start).count();
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time).count();

            const bool should_log = completed % log_every == 0 || completed == additional_steps;
            if (should_log) {
                std::cout << "Paso " << optimizer.step_count()
                          << " (+" << completed << "): loss=" << loss
                          << " perplexity=" << mini_gpt::perplexity(loss)
                          << " tokens/s=" << batch->real_tokens / step_seconds
                          << " tiempo_paso=" << step_seconds << "s"
                          << " total=" << elapsed << "s\n";
            }
            if (eval_every != 0 && completed % eval_every == 0) {
                const double validation = evaluate_subset(loader, model, validation_samples);
                std::cout << "  Validation Loss: " << validation
                          << " perplexity=" << mini_gpt::perplexity(validation) << '\n';
            }
            if (save_every != 0 && completed % save_every == 0) {
                model.save(output.model);
                optimizer.save(output.optimizer);
                mini_gpt::save_training_state(
                    output.state,
                    TrainingState{
                        normalized_path(corpus_path), normalized_path(tokenizer_path),
                        epoch, processed_tokens, loader.state()});
                std::cout << "  Checkpoint guardado en " << output.model.string() << '\n';
            }
        }

        model.save(output.model);
        optimizer.save(output.optimizer);
        mini_gpt::save_training_state(
            output.state,
            TrainingState{
                normalized_path(corpus_path), normalized_path(tokenizer_path),
                epoch, processed_tokens, loader.state()});
        if (stop_requested != 0) {
            std::cout << "Interrupción recibida: checkpoint guardado.\n";
        }
        if (initial_parameter == model.output_weights().at({0, 0}) && completed != 0) {
            throw std::runtime_error("Los parámetros no cambiaron durante el entrenamiento");
        }
        if (eval_every != 0) {
            const double validation = evaluate_subset(loader, model, validation_samples);
            std::cout << "Validation Loss final: " << validation
                      << "\nPerplexity final: " << mini_gpt::perplexity(validation) << '\n';
        }
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        std::cout << "Última training loss: " << last_loss
                  << "\nPasos adicionales completados: " << completed
                  << "\nPaso total de AdamW: " << optimizer.step_count()
                  << "\nTiempo total: " << elapsed << "s"
                  << "\nTokens/s medio: " << session_tokens / elapsed
                  << "\nCheckpoint final: " << output.model.string()
                  << "\nAdamW: " << output.optimizer.string()
                  << "\nDatos: " << output.state.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

#include "mini_gpt/adamw.hpp"
#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/cross_entropy.hpp"
#include "mini_gpt/data_loader.hpp"
#include "mini_gpt/gpt.hpp"

#ifdef MINI_GPT_ENABLE_CUDA
#include "mini_gpt/cuda/memory.cuh"
#include "mini_gpt/cuda/registry.cuh"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using mini_gpt::AdamW;
using mini_gpt::BPETokenizer;
using mini_gpt::GPT;
using mini_gpt::Tensor;
using mini_gpt::TextDataLoader;

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void print_measurement(const char* name, double value, double total) {
    std::cout << name << ',' << value << ',';
    if (total > 0.0) {
        std::cout << 100.0 * value / total;
    } else {
        std::cout << "NA";
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 5) {
        std::cerr << "Uso: benchmark_demo [corpus] [vocabulario] [modelo] [adamw]\n";
        return 2;
    }

    const std::filesystem::path corpus_path = argc >= 2
        ? argv[1]
        : std::filesystem::path("tests/datasets/Gongora_poemas.txt");
    const std::filesystem::path vocabulary_path = argc >= 3 ? argv[2] : "gongora.bpe";
    const std::filesystem::path model_path = argc >= 4 ? argv[3] : "gongora_checkpoint.bin";
    const std::filesystem::path optimizer_path = argc >= 5
        ? argv[4]
        : std::filesystem::path("gongora_checkpoint_adamw.bin");

    try {
        const auto load_start = Clock::now();
        const BPETokenizer tokenizer = BPETokenizer::load(vocabulary_path);
        GPT model = GPT::load(model_path);
        AdamW optimizer = std::filesystem::exists(optimizer_path)
            ? AdamW::load(optimizer_path, model)
            : AdamW(model);
        const std::string corpus = mini_gpt::read_utf8_file(corpus_path);
        const auto split = mini_gpt::split_corpus_utf8(corpus);
        TextDataLoader loader(tokenizer, split.training, split.validation, 128, 2, 42, false);
        const double load_time = seconds(load_start);
        if (tokenizer.vocab_size() != model.vocab_size()) {
            throw std::invalid_argument("Vocabulario y modelo incompatibles");
        }

        auto batch = loader.next_batch();
        if (!batch || batch->batch_size != 2 || batch->seq_len != 128) {
            throw std::runtime_error("No se pudo obtener un batch 2 x 128");
        }

#ifdef MINI_GPT_ENABLE_CUDA
        mini_gpt::cuda::reset_transfer_stats();
        mini_gpt::cuda::reset_peak_live_device_bytes();
#endif
        const auto forward_start = Clock::now();
        const Tensor logits = model.forward(batch->inputs, batch->batch_size, batch->seq_len);
        const double forward_time = seconds(forward_start);

        const auto loss_start = Clock::now();
        const float loss = mini_gpt::cross_entropy(logits, batch->targets, batch->mask);
        const double loss_time = seconds(loss_start);

        model.zero_grad();
        const auto backward_start = Clock::now();
        const float backward_loss = model.backward(
            batch->inputs,
            batch->batch_size,
            batch->seq_len,
            batch->targets,
            batch->mask);
        const double backward_total_time = seconds(backward_start);

        const auto optimizer_start = Clock::now();
        const double gradient_norm = mini_gpt::clip_gradients(optimizer.parameters(), 1.0);
        optimizer.step();
        const double optimizer_time = seconds(optimizer_start);
#ifdef MINI_GPT_ENABLE_CUDA
        const auto step_transfers = mini_gpt::cuda::transfer_stats();
#endif
        if (!std::isfinite(loss) || !std::isfinite(backward_loss) || !std::isfinite(gradient_norm)) {
            throw std::runtime_error("Valores no finitos durante el benchmark");
        }

        const auto validation_start = Clock::now();
        double validation_sum = 0.0;
        std::size_t validation_tokens = 0;
        std::size_t validation_batches = 0;
        loader.reset_validation();
        while (auto validation_batch = loader.next_validation_batch()) {
            const Tensor validation_logits = model.forward(
                validation_batch->inputs,
                validation_batch->batch_size,
                validation_batch->seq_len);
            const float validation_loss = mini_gpt::cross_entropy(
                validation_logits, validation_batch->targets, validation_batch->mask);
            validation_sum += static_cast<double>(validation_loss) * validation_batch->real_tokens;
            validation_tokens += validation_batch->real_tokens;
            ++validation_batches;
        }
        const double validation_time = seconds(validation_start);
        const double validation_loss = validation_sum / validation_tokens;

        const double total_step = backward_total_time + optimizer_time;
        const double backward_only = std::max(0.0, backward_total_time - forward_time - loss_time);
        std::cout << std::fixed << std::setprecision(6)
                  << "OPTIMIZER_STEP," << optimizer.step_count()
                  << "\nBATCH_TOKENS," << batch->real_tokens
                  << "\nLOSS," << backward_loss
                  << "\nGRADIENT_NORM," << gradient_norm
                  << "\nMETRIC,SECONDS,PERCENT_OF_STEP\n";
        print_measurement("FORWARD", forward_time, total_step);
        print_measurement("LOSS", loss_time, total_step);
        print_measurement("BACKWARD_ONLY", backward_only, total_step);
        print_measurement("ADAMW_CLIP", optimizer_time, total_step);
        print_measurement("TOTAL_STEP", total_step, total_step);
        std::cout << "TOKENS_PER_SECOND," << batch->real_tokens / total_step
                  << "\nLOAD_TOKENIZE," << load_time
                  << "\nVALIDATION," << validation_time
                  << "\nVALIDATION_BATCHES," << validation_batches
                  << "\nVALIDATION_TOKENS," << validation_tokens
                  << "\nVALIDATION_LOSS," << validation_loss
                  << "\nVALIDATION_TOKENS_PER_SECOND," << validation_tokens / validation_time
#ifdef MINI_GPT_ENABLE_CUDA
                  << "\nCUDA_POOL_BYTES," << mini_gpt::cuda::cached_device_bytes()
                  << "\nCUDA_LIVE_BYTES," << mini_gpt::cuda::live_device_bytes()
                  << "\nCUDA_PEAK_LIVE_BYTES," << mini_gpt::cuda::peak_live_device_bytes()
                  << "\nCUDA_STEP_HOST_TO_DEVICE_BYTES," << step_transfers.host_to_device_bytes
                  << "\nCUDA_STEP_DEVICE_TO_HOST_BYTES," << step_transfers.device_to_host_bytes
#endif
                  << '\n';
#ifdef MINI_GPT_ENABLE_CUDA
        auto warm_batch = loader.next_batch();
        if (!warm_batch) {
            throw std::runtime_error("No se pudo obtener el batch de estado estacionario");
        }
        mini_gpt::cuda::reset_transfer_stats();
        mini_gpt::cuda::reset_peak_live_device_bytes();
        const auto warm_forward_start = Clock::now();
        const Tensor warm_logits = model.forward(
            warm_batch->inputs, warm_batch->batch_size, warm_batch->seq_len);
        const double warm_forward_time = seconds(warm_forward_start);
        const auto warm_loss_start = Clock::now();
        const float warm_loss = mini_gpt::cross_entropy(
            warm_logits, warm_batch->targets, warm_batch->mask);
        const double warm_loss_time = seconds(warm_loss_start);
        model.zero_grad();
        const auto warm_backward_start = Clock::now();
        const float warm_backward_loss = model.backward(
            warm_batch->inputs,
            warm_batch->batch_size,
            warm_batch->seq_len,
            warm_batch->targets,
            warm_batch->mask);
        const double warm_backward_total_time = seconds(warm_backward_start);
        const auto warm_optimizer_start = Clock::now();
        const double warm_gradient_norm = mini_gpt::clip_gradients(optimizer.parameters(), 1.0);
        optimizer.step();
        const double warm_optimizer_time = seconds(warm_optimizer_start);
        if (!std::isfinite(warm_loss) || !std::isfinite(warm_backward_loss) ||
            !std::isfinite(warm_gradient_norm)) {
            throw std::runtime_error("Valores no finitos durante el paso de estado estacionario");
        }
        const auto warm_transfers = mini_gpt::cuda::transfer_stats();
        const double warm_total_step = warm_backward_total_time + warm_optimizer_time;
        const double warm_backward_only = std::max(
            0.0, warm_backward_total_time - warm_forward_time - warm_loss_time);
        std::cout << "WARM_OPTIMIZER_STEP," << optimizer.step_count()
                  << "\nWARM_LOSS," << warm_backward_loss
                  << "\nWARM_METRIC,SECONDS,PERCENT_OF_STEP\n";
        print_measurement("WARM_FORWARD", warm_forward_time, warm_total_step);
        print_measurement("WARM_LOSS", warm_loss_time, warm_total_step);
        print_measurement("WARM_BACKWARD_ONLY", warm_backward_only, warm_total_step);
        print_measurement("WARM_ADAMW_CLIP", warm_optimizer_time, warm_total_step);
        print_measurement("WARM_TOTAL_STEP", warm_total_step, warm_total_step);
        std::cout << "WARM_TOKENS_PER_SECOND," << warm_batch->real_tokens / warm_total_step
                  << "\nWARM_CUDA_POOL_BYTES," << mini_gpt::cuda::cached_device_bytes()
                  << "\nWARM_CUDA_PEAK_LIVE_BYTES," << mini_gpt::cuda::peak_live_device_bytes()
                  << "\nWARM_CUDA_HOST_TO_DEVICE_BYTES," << warm_transfers.host_to_device_bytes
                  << "\nWARM_CUDA_DEVICE_TO_HOST_BYTES," << warm_transfers.device_to_host_bytes
                  << '\n';
#endif
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

#include "mini_gpt/backpropagation.hpp"
#include "mini_gpt/bpe_tokenizer.hpp"
#include "mini_gpt/cross_entropy.hpp"
#include "mini_gpt/gpt.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mini_gpt::BPETokenizer;
using mini_gpt::GPT;
using mini_gpt::Tensor;
using mini_gpt::TransformerBlockGradients;
using TokenId = GPT::TokenId;

std::string read_corpus(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se puede abrir el corpus: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

double gradient_norm(const Tensor& gradient) {
    double squared_sum = 0.0;
    for (const float value : gradient.data()) {
        squared_sum += static_cast<double>(value) * value;
    }
    return std::sqrt(squared_sum);
}

double block_gradient_norm(const TransformerBlockGradients& gradients) {
    double squared_sum = 0.0;
    for (const Tensor* tensor : {
             &gradients.norm1.gamma, &gradients.norm1.beta,
             &gradients.norm2.gamma, &gradients.norm2.beta,
             &gradients.attention.q_weights, &gradients.attention.k_weights,
             &gradients.attention.v_weights, &gradients.attention.output_weights,
             &gradients.fc1_weights, &gradients.fc2_weights}) {
        for (const float value : tensor->data()) {
            squared_sum += static_cast<double>(value) * value;
        }
    }
    return std::sqrt(squared_sum);
}

void check_finite_norm(double norm, const char* component) {
    if (!std::isfinite(norm)) {
        throw std::runtime_error(std::string("Gradiente no finito en ") + component);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 4) {
        std::cerr << "Uso: " << argv[0] << " [vocabulario.bpe] [modelo.bin] [corpus.txt]\n";
        return 2;
    }

    const std::filesystem::path vocabulary_path = argc >= 2
        ? std::filesystem::path(argv[1])
        : std::filesystem::current_path() / "tokenizer_demo.bpe";
    const std::filesystem::path model_path = argc >= 3
        ? std::filesystem::path(argv[2])
        : std::filesystem::current_path() / "gpt_model.bin";
    const std::filesystem::path corpus_path = argc == 4
        ? std::filesystem::path(argv[3])
        : std::filesystem::path(MINI_GPT_BACKPROP_CORPUS_PATH);

    try {
        const BPETokenizer tokenizer = BPETokenizer::load(vocabulary_path);
        GPT model = GPT::load(model_path);
        if (tokenizer.vocab_size() != model.vocab_size()) {
            throw std::invalid_argument("El vocabulario BPE no coincide con el modelo");
        }

        const auto tokens = tokenizer.encode(read_corpus(corpus_path));
        auto shifted = mini_gpt::shift_tokens(tokens);
        if (shifted.inputs.size() > model.max_seq_len()) {
            const std::size_t start = shifted.inputs.size() - model.max_seq_len();
            shifted.inputs = std::vector<TokenId>(
                shifted.inputs.begin() + start, shifted.inputs.end());
            shifted.targets = std::vector<TokenId>(
                shifted.targets.begin() + start, shifted.targets.end());
        }

        model.zero_grad();
        const float loss = model.backward(shifted.inputs, 1, shifted.inputs.size(), shifted.targets);
        const auto& gradients = model.gradients();
        const double token_norm = gradient_norm(gradients.embedding.token_embeddings);
        const double position_norm = gradient_norm(gradients.embedding.positional_embeddings);
        const double final_norm_gradient = gradient_norm(gradients.final_norm.gamma) +
                                           gradient_norm(gradients.final_norm.beta);
        const double output_norm = gradient_norm(gradients.output_weights);
        check_finite_norm(token_norm, "token embeddings");
        check_finite_norm(position_norm, "positional embeddings");
        check_finite_norm(final_norm_gradient, "final LayerNorm");
        check_finite_norm(output_norm, "output projection");

        std::cout << std::fixed << std::setprecision(6)
                  << "Cross-Entropy media: " << loss
                  << "\nNormas de gradientes:\n"
                  << std::scientific << std::setprecision(8)
                  << "  Token embeddings: " << token_norm << '\n'
                  << "  Positional embeddings: " << position_norm << '\n';
        for (std::size_t layer = 0; layer < gradients.blocks.size(); ++layer) {
            const double norm = block_gradient_norm(gradients.blocks[layer]);
            check_finite_norm(norm, "Transformer Block");
            std::cout << "  Bloque " << layer << ": " << norm << '\n';
        }
        std::cout << "  LayerNorm final: " << final_norm_gradient << '\n'
                  << "  Proyección de salida: " << output_norm << '\n';

        const TokenId first_id = shifted.inputs.front();
        std::cout << "Gradientes representativos:\n"
                  << "  token_embedding[" << first_id << ", 0]: "
                  << gradients.embedding.token_embeddings.at({first_id, 0}) << '\n'
                  << "  positional_embedding[0, 0]: "
                  << gradients.embedding.positional_embeddings.at({0, 0}) << '\n'
                  << "  bloque0.q_weight[0, 0]: "
                  << gradients.blocks[0].attention.q_weights.at({0, 0}) << '\n'
                  << "  bloque0.fc1_weight[0, 0]: "
                  << gradients.blocks[0].fc1_weights.at({0, 0}) << '\n'
                  << "  final_gamma[0]: " << gradients.final_norm.gamma.at({0}) << '\n'
                  << "  output_weight[0, 0]: " << gradients.output_weights.at({0, 0}) << '\n'
                  << "Backpropagation completada sin actualizar los pesos.\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

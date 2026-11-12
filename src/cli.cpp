#include "mini_gpt/cli.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mini_gpt::cli {

Arguments collect_arguments(int argc, char** argv) {
    Arguments args;
    args.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return args;
}

bool has(const Arguments& args, std::string_view flag) {
    return std::find(args.begin() + 1, args.end(), flag) != args.end();
}

std::string value(
    const Arguments& args,
    std::string_view option,
    const std::string& default_value) {
    std::string result = default_value;
    bool found = false;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] != option) {
            continue;
        }
        if (found) {
            throw std::invalid_argument("Opción duplicada: " + std::string(option));
        }
        if (i + 1 >= args.size() || args[i + 1].starts_with("--")) {
            throw std::invalid_argument("Falta valor para " + std::string(option));
        }
        result = args[++i];
        found = true;
    }
    return result;
}

std::string required(const Arguments& args, std::string_view option) {
    const std::string result = value(args, option, "");
    if (result.empty()) {
        throw std::invalid_argument("Falta la opción obligatoria " + std::string(option));
    }
    return result;
}

std::size_t size_value(
    const Arguments& args,
    std::string_view option,
    std::size_t default_value,
    std::size_t minimum) {
    const std::string text = value(args, option, std::to_string(default_value));
    std::size_t consumed = 0;
    unsigned long long result = 0;
    try {
        result = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("Entero inválido en " + std::string(option) + ": " + text);
    }
    if (consumed != text.size() || result < minimum ||
        result > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("Valor fuera de rango en " + std::string(option) + ": " + text);
    }
    return static_cast<std::size_t>(result);
}

float float_value(const Arguments& args, std::string_view option, float default_value) {
    std::ostringstream default_text;
    default_text << default_value;
    const std::string text = value(args, option, default_text.str());
    std::size_t consumed = 0;
    float result = 0.0F;
    try {
        result = std::stof(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("Número inválido en " + std::string(option) + ": " + text);
    }
    if (consumed != text.size() || !std::isfinite(result)) {
        throw std::invalid_argument("Número inválido en " + std::string(option) + ": " + text);
    }
    return result;
}

void validate(
    const Arguments& args,
    std::initializer_list<std::string_view> flags,
    std::initializer_list<std::string_view> options) {
    const auto is_flag = [&](std::string_view arg) {
        return std::find(flags.begin(), flags.end(), arg) != flags.end();
    };
    const auto is_option = [&](std::string_view arg) {
        return std::find(options.begin(), options.end(), arg) != options.end();
    };

    for (std::size_t i = 1; i < args.size(); ++i) {
        if (is_flag(args[i])) {
            continue;
        }
        if (is_option(args[i])) {
            if (i + 1 >= args.size() || args[i + 1].starts_with("--")) {
                throw std::invalid_argument("Falta valor para " + args[i]);
            }
            ++i;
            continue;
        }
        throw std::invalid_argument("Argumento desconocido: " + args[i]);
    }
}

void require_input_file(const std::filesystem::path& path) {
    if (path.empty() || !std::filesystem::exists(path) ||
        !std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("No existe el archivo de entrada: " + path.string());
    }
}

void require_output_file(const std::filesystem::path& path, bool overwrite) {
    if (path.empty()) {
        throw std::invalid_argument("La ruta de salida está vacía");
    }
    if (std::filesystem::exists(path) && (!overwrite || std::filesystem::is_directory(path))) {
        throw std::runtime_error("La salida ya existe; usa --overwrite para autorizarla: " + path.string());
    }
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent)) {
        throw std::runtime_error("No existe el directorio de salida: " + parent.string());
    }
}

}  // namespace mini_gpt::cli

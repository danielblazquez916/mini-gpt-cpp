#ifndef MINI_GPT_CLI_HPP
#define MINI_GPT_CLI_HPP

#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace mini_gpt::cli {

using Arguments = std::vector<std::string>;

[[nodiscard]] Arguments collect_arguments(int argc, char** argv);
[[nodiscard]] bool has(const Arguments& args, std::string_view flag);
[[nodiscard]] std::string value(
    const Arguments& args,
    std::string_view option,
    const std::string& default_value);
[[nodiscard]] std::string required(const Arguments& args, std::string_view option);
[[nodiscard]] std::size_t size_value(
    const Arguments& args,
    std::string_view option,
    std::size_t default_value,
    std::size_t minimum = 1);
[[nodiscard]] float float_value(
    const Arguments& args,
    std::string_view option,
    float default_value);
void validate(
    const Arguments& args,
    std::initializer_list<std::string_view> flags,
    std::initializer_list<std::string_view> options);
void require_input_file(const std::filesystem::path& path);
void require_output_file(const std::filesystem::path& path, bool overwrite);

}  // namespace mini_gpt::cli

#endif

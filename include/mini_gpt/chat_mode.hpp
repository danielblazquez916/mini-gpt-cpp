#ifndef MINI_GPT_CHAT_MODE_HPP
#define MINI_GPT_CHAT_MODE_HPP

#include <string>
#include <string_view>

namespace mini_gpt {

enum class ChatMode {
    chat,
    completion,
};

[[nodiscard]] ChatMode parse_chat_mode(std::string_view mode);
[[nodiscard]] std::string build_chat_prompt(ChatMode mode, const std::string& input);

}  // namespace mini_gpt

#endif

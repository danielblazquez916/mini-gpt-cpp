#include "mini_gpt/chat_mode.hpp"

#include <stdexcept>

namespace mini_gpt {

ChatMode parse_chat_mode(std::string_view mode) {
    if (mode == "chat") {
        return ChatMode::chat;
    }
    if (mode == "completion" || mode == "continue") {
        return ChatMode::completion;
    }
    throw std::invalid_argument("--mode debe ser chat o completion");
}

std::string build_chat_prompt(ChatMode mode, const std::string& input) {
    if (mode == ChatMode::chat) {
        return "<|user|> " + input + " <|eot|>\n<|assistant|>";
    }
    return input;
}

}  // namespace mini_gpt

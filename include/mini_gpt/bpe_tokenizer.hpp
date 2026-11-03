#ifndef MINI_GPT_BPE_TOKENIZER_HPP
#define MINI_GPT_BPE_TOKENIZER_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini_gpt {

class BPETokenizer {
public:
    using TokenId = std::uint32_t;

    struct Merge {
        TokenId left;
        TokenId right;

        bool operator==(const Merge&) const = default;
    };

    static constexpr TokenId user_id = 256;
    static constexpr TokenId assistant_id = 257;
    static constexpr TokenId eot_id = 258;
    static constexpr TokenId end_id = 259;
    static constexpr std::size_t base_vocabulary_size = 260;

    // El límite incluye los 256 bytes, los especiales y las fusiones.
    explicit BPETokenizer(std::size_t max_vocab_size = 4000);

    void train(std::string_view corpus);

    [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(std::span<const TokenId> ids) const;

    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static BPETokenizer load(const std::filesystem::path& path);

    [[nodiscard]] std::size_t vocab_size() const noexcept;
    [[nodiscard]] std::size_t max_vocab_size() const noexcept;
    [[nodiscard]] const std::vector<std::string>& vocabulary() const noexcept;
    [[nodiscard]] const std::vector<Merge>& merges() const noexcept;
    [[nodiscard]] std::string_view token(TokenId id) const;

private:
    using Pair = std::pair<TokenId, TokenId>;

    struct PairHash {
        [[nodiscard]] std::size_t operator()(const Pair& pair) const noexcept;
    };

    [[nodiscard]] static std::size_t validate_max_vocab_size(std::size_t size);
    void reset_to_base();
    void append_encoded(std::string_view text, std::vector<TokenId>& output) const;

    std::size_t max_vocab_size_;
    std::vector<std::string> vocabulary_;
    std::vector<Merge> merges_;
    std::unordered_map<Pair, std::size_t, PairHash> merge_ranks_;
};

}  // namespace mini_gpt

#endif

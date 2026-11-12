#ifndef MINI_GPT_CUDA_REGISTRY_CUH
#define MINI_GPT_CUDA_REGISTRY_CUH

#include <cstddef>
#include <cstdint>

namespace mini_gpt::cuda {

struct TransferStats {
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t device_to_host_bytes = 0;
};

void reset_transfer_stats() noexcept;
[[nodiscard]] TransferStats transfer_stats() noexcept;

void register_host(const void* host, std::size_t bytes, bool host_dirty);
void unregister_host(const void* host);
void move_host_registration(const void* old_host, const void* new_host);
[[nodiscard]] bool is_registered(const void* host);
void* device_for_host(const void* host);
void* alias_for_buffer(void* buffer);
void set_alias(void* buffer, void* device);
void clear_alias(void* buffer);
void sync_host(const void* host);
void mark_host_modified(const void* host);
void mark_device_modified(void* host);

}  // namespace mini_gpt::cuda

#endif

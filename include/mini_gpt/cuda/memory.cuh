#ifndef MINI_GPT_CUDA_MEMORY_CUH
#define MINI_GPT_CUDA_MEMORY_CUH

#include <cstddef>

namespace mini_gpt::cuda {

void* allocate_device(std::size_t bytes);
void deallocate_device(void* pointer, std::size_t bytes);
[[nodiscard]] std::size_t cached_device_bytes();
[[nodiscard]] std::size_t live_device_bytes();
[[nodiscard]] std::size_t peak_live_device_bytes();
void reset_peak_live_device_bytes();

}  // namespace mini_gpt::cuda

#endif

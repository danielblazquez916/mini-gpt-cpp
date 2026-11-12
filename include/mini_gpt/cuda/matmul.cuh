#ifndef MINI_GPT_CUDA_MATMUL_CUH
#define MINI_GPT_CUDA_MATMUL_CUH

#include <cstddef>

namespace mini_gpt::cuda {

void matmul(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t rows,
    std::size_t inner,
    std::size_t columns);

}  // namespace mini_gpt::cuda

#endif

#ifndef CUTE_TRANSPOSE_CUH
#define CUTE_TRANSPOSE_CUH

#include <cuda_runtime.h>

template <class TENSOR_SRC, class TENSOR_DST, class THREAD_LAYOUT>
__global__ void transpose_naive(TENSOR_SRC tensor_src,
                                TENSOR_DST tensor_dst_transposed,
                                THREAD_LAYOUT);

template <class TENSOR_SRC, class TENSOR_DST, class THREAD_LAYOUT>
__global__ void transpose_naive_coalesced_write(
    TENSOR_SRC tensor_src, TENSOR_DST tensor_dst_transposed, THREAD_LAYOUT);

template <class TENSOR_SRC, class TENSOR_DST, class THREAD_LAYOUT>
__global__ void transpose_shared_memory_bank_conflict_read(
    TENSOR_SRC tensor_src, TENSOR_DST tensor_dst_transposed, THREAD_LAYOUT);

template <class TENSOR_SRC, class TENSOR_DST, class THREAD_LAYOUT>
__global__ void transpose_shared_memory_bank_conflict_write(
    TENSOR_SRC tensor_src, TENSOR_DST tensor_dst_transposed, THREAD_LAYOUT);

template <class TENSOR_SRC, class TENSOR_DST, class THREAD_LAYOUT>
__global__ void transpose_naive_shared_memory_padded(
    TENSOR_SRC tensor_src, TENSOR_DST tensor_dst_transposed, THREAD_LAYOUT);

#endif // CUTE_TRANSPOSE_CUH

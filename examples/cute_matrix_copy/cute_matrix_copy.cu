#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "cute_matrix_copy.hpp"

template <class TENSOR_SRC, class TENSOR_DST, class THREAD_LAYOUT>
__global__ void copy(TENSOR_SRC tensor_src, TENSOR_DST tensor_dst_transposed, THREAD_LAYOUT)
{
    using Element = typename TENSOR_SRC::value_type;

    auto global_tile_src{tensor_src(cute::make_coord(cute::_, cute::_),
                                    blockIdx.y,
                                    blockIdx.x)}; // (TILE_SIZE_Y, TILE_SIZE_X)
    auto global_tile_dst_transposed{
        tensor_dst_transposed(cute::make_coord(cute::_, cute::_), blockIdx.y,
                              blockIdx.x)}; // (TILE_SIZE_Y, TILE_SIZE_X)

    auto thread_global_tile_src{cute::local_partition(
        global_tile_src, THREAD_LAYOUT{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_Y, THREAD_VALUE_SIZE_X)
    auto thread_global_tile_dst_transposed{cute::local_partition(
        global_tile_dst_transposed, THREAD_LAYOUT{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_Y, THREAD_VALUE_SIZE_X)
}
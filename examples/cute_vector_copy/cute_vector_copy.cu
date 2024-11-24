#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "cute_vector_copy.hpp"

template <class TENSOR_SRC, class TENSOR_DST, class THREAD_LAYOUT>
static __global__ void vector_copy(TENSOR_SRC tensor_src, TENSOR_DST tensor_dst,
                                   unsigned int size, THREAD_LAYOUT)
{
    using Element = typename TENSOR_SRC::value_type;

    auto global_tile_src{tensor_src(cute::make_coord(cute::_), blockIdx.x)};
    auto global_tile_dst{tensor_dst(cute::make_coord(cute::_), blockIdx.x)};

    auto thread_global_tile_src{
        cute::local_partition(global_tile_src, THREAD_LAYOUT{}, threadIdx.x)};
    auto thread_global_tile_dst{
        cute::local_partition(global_tile_dst, THREAD_LAYOUT{}, threadIdx.x)};

    auto const identity_tensor{cute::make_identity_tensor(
        cute::make_shape(cute::size(global_tile_src)))};
    auto const thread_identity_tensor{
        cute::local_partition(identity_tensor, THREAD_LAYOUT{}, threadIdx.x)};

    auto fragment{cute::make_fragment_like(thread_global_tile_src)};
    auto predicator{
        cute::make_tensor<bool>(cute::make_shape(cute::size(fragment)))};

    constexpr auto tile_size{cute::size<0>(global_tile_src)};

    CUTE_UNROLL
    for (unsigned int i{0}; i < cute::size(predicator); ++i)
    {
        auto const thread_identity{thread_identity_tensor(i)};
        bool const is_in_bound{
            cute::get<0>(thread_identity) + blockIdx.x * tile_size < size};
        predicator(i) = is_in_bound;
    }

    cute::copy_if(predicator, thread_global_tile_src, fragment);
    cute::copy_if(predicator, fragment, thread_global_tile_dst);
}

template <typename T>
static cudaError_t launch_vector_copy(T const* input_vector, T* output_vector,
                                      unsigned int size, cudaStream_t stream)
{
    auto const tensor_shape{cute::make_shape(size)};
    auto const global_memory_layout_src{cute::make_layout(tensor_shape)};
    auto const global_memory_layout_dst{cute::make_layout(tensor_shape)};

    auto const tensor_src{cute::make_tensor(cute::make_gmem_ptr(input_vector),
                                            global_memory_layout_src)};
    auto const tensor_dst{cute::make_tensor(cute::make_gmem_ptr(output_vector),
                                            global_memory_layout_dst)};

    using TILE_SIZE_X = cute::Int<2048>;

    constexpr auto block_shape{cute::make_shape(TILE_SIZE_X{})};

    auto const tiled_tensor_src{cute::tiled_divide(tensor_src, block_shape)};
    auto const tiled_tensor_dst{cute::tiled_divide(tensor_dst, block_shape)};

    using THREAD_BLOCK_SIZE_X = cute::Int<256>;

    constexpr auto thread_block_shape{cute::make_shape(THREAD_BLOCK_SIZE_X{})};
    constexpr auto thread_layout{cute::make_layout(thread_block_shape)};

    dim3 const grid_dim{cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{cute::size(thread_layout)};

    CUTE_STATIC_ASSERT(TILE_SIZE_X::value % THREAD_BLOCK_SIZE_X::value == 0,
                       "TILE_SIZE_X must be divisible by THREAD_BLOCK_SIZE_X");
    vector_copy<<<grid_dim, thread_dim, 0, stream>>>(
        tiled_tensor_src, tiled_tensor_dst, size, thread_layout);

    return cudaGetLastError();
}

// Explicit instantiation.
template cudaError_t launch_vector_copy<float>(float const* input_vector,
                                               float* output_vector,
                                               unsigned int size,
                                               cudaStream_t stream);
template cudaError_t launch_vector_copy<double>(double const* input_vector,
                                                double* output_vector,
                                                unsigned int size,
                                                cudaStream_t stream);
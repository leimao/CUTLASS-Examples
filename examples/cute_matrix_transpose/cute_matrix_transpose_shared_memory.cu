#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "cute_matrix_transpose.hpp"

template <class TENSOR_SRC, class TENSOR_DST, class SHARED_MEMORY_LAYOUT_SRC,
          class SHARED_MEMORY_LAYOUT_DST, class THREAD_LAYOUT_SRC,
          class THREAD_LAYOUT_DST>
static __global__ void
transpose_shared_memory(TENSOR_SRC tensor_src, TENSOR_DST tensor_dst,
                        SHARED_MEMORY_LAYOUT_SRC, SHARED_MEMORY_LAYOUT_DST,
                        THREAD_LAYOUT_SRC, THREAD_LAYOUT_DST)
{
    using Element = typename TENSOR_SRC::value_type;
    CUTE_STATIC_ASSERT(cute::size(SHARED_MEMORY_LAYOUT_SRC{}) ==
                           cute::size(SHARED_MEMORY_LAYOUT_DST{}),
                       "SHARED_MEMORY_LAYOUT_SRC and SHARED_MEMORY_LAYOUT_DST "
                       "must have the same size.");
    __shared__ Element shared_memory[cute::cosize(SHARED_MEMORY_LAYOUT_SRC{})];

    auto tensor_cache_src{cute::make_tensor(cute::make_smem_ptr(shared_memory),
                                            SHARED_MEMORY_LAYOUT_SRC{})};
    auto tensor_cache_dst{cute::make_tensor(cute::make_smem_ptr(shared_memory),
                                            SHARED_MEMORY_LAYOUT_DST{})};

    auto global_tile_src{tensor_src(cute::make_coord(cute::_, cute::_),
                                    blockIdx.y,
                                    blockIdx.x)}; // (TILE_SIZE_Y, TILE_SIZE_X)
    auto global_tile_dst{tensor_dst(cute::make_coord(cute::_, cute::_),
                                    blockIdx.y,
                                    blockIdx.x)}; // (TILE_SIZE_Y, TILE_SIZE_X)

    auto thread_global_tile_src{cute::local_partition(
        global_tile_src, THREAD_LAYOUT_SRC{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_Y, THREAD_VALUE_SIZE_X)
    auto thread_global_tile_dst{cute::local_partition(
        global_tile_dst, THREAD_LAYOUT_DST{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_X, THREAD_VALUE_SIZE_Y)

    auto thread_shared_tile_src{cute::local_partition(
        tensor_cache_src, THREAD_LAYOUT_SRC{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_Y, THREAD_VALUE_SIZE_X)
    auto thread_shared_tile_dst{cute::local_partition(
        tensor_cache_dst, THREAD_LAYOUT_DST{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_X, THREAD_VALUE_SIZE_Y)

    // A 2D array of tuples that maps (x, y) to (x, y).
    auto const identity_tensor_src{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(global_tile_src), cute::size<1>(global_tile_src)))};
    auto const thread_identity_tensor_src{cute::local_partition(
        identity_tensor_src, THREAD_LAYOUT_SRC{}, threadIdx.x)};
    auto predicator_src{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(thread_global_tile_src),
                         cute::size<1>(thread_global_tile_src)))};

    auto const identity_tensor_dst{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(global_tile_dst), cute::size<1>(global_tile_dst)))};
    auto const thread_identity_tensor_dst{cute::local_partition(
        identity_tensor_dst, THREAD_LAYOUT_DST{}, threadIdx.x)};
    auto predicator_dst{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(thread_global_tile_dst),
                         cute::size<1>(thread_global_tile_dst)))};

    auto const num_max_columns{cute::stride<0>(global_tile_src)};
    auto const num_max_rows{cute::stride<1>(global_tile_dst)};
    constexpr auto global_tile_columns{cute::size<1>(global_tile_src)};
    constexpr auto global_tile_rows{cute::size<0>(global_tile_src)};

    CUTE_UNROLL
    for (unsigned int i{0}; i < cute::size<0>(predicator_src); ++i)
    {
        CUTE_UNROLL
        for (unsigned int j{0}; j < cute::size<1>(predicator_src); ++j)
        {
            auto const thread_identity{thread_identity_tensor_src(i, j)};
            bool const is_row_in_bound{cute::get<0>(thread_identity) +
                                           blockIdx.y * global_tile_rows <
                                       num_max_rows};
            bool const is_column_in_bound{cute::get<1>(thread_identity) +
                                              blockIdx.x * global_tile_columns <
                                          num_max_columns};
            predicator_src(i, j) = is_row_in_bound && is_column_in_bound;
        }
    }

    CUTE_UNROLL
    for (unsigned int i{0}; i < cute::size<0>(predicator_dst); ++i)
    {
        CUTE_UNROLL
        for (unsigned int j{0}; j < cute::size<1>(predicator_dst); ++j)
        {
            auto const thread_identity{thread_identity_tensor_dst(i, j)};
            bool const is_row_in_bound{cute::get<0>(thread_identity) +
                                           blockIdx.y * global_tile_rows <
                                       num_max_rows};
            bool const is_column_in_bound{cute::get<1>(thread_identity) +
                                              blockIdx.x * global_tile_columns <
                                          num_max_columns};
            predicator_dst(i, j) = is_row_in_bound && is_column_in_bound;
        }
    }

    cute::copy_if(predicator_src, thread_global_tile_src,
                  thread_shared_tile_src);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
    cute::copy_if(predicator_dst, thread_shared_tile_dst,
                  thread_global_tile_dst);
}

enum class SharedMemoryBankConflictAccessMode
{
    Read,
    Write
};

template <typename T>
static cudaError_t launch_transpose_shared_memory_bank_conflict_base(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    SharedMemoryBankConflictAccessMode bank_conflict_access_mode,
    cudaStream_t stream)
{
    auto const tensor_shape{cute::make_shape(M, N)};
    auto const tensor_shape_transposed{cute::make_shape(N, M)};

    // Input matrix: row-major M x N matrix.
    auto const global_memory_layout_src{cute::make_layout(
        tensor_shape, cute::GenRowMajor{})}; // (M, N) : (N, 1)
    // Output matrix: row-major N x M matrix.
    auto const global_memory_layout_dst{cute::make_layout(
        tensor_shape_transposed, cute::GenRowMajor{})}; // (N, M) : (M, 1)
    // Same output matrix, but different view: column-major M x N matrix.
    auto const global_memory_layout_dst_transposed{cute::make_layout(
        tensor_shape, cute::GenColMajor{})}; // (M, N) : (1, M)

    auto const tensor_src{cute::make_tensor(cute::make_gmem_ptr(input_matrix),
                                            global_memory_layout_src)};
    auto const tensor_dst{cute::make_tensor(cute::make_gmem_ptr(output_matrix),
                                            global_memory_layout_dst)};
    auto const tensor_dst_transposed{
        cute::make_tensor(cute::make_gmem_ptr(output_matrix),
                          global_memory_layout_dst_transposed)};

    using TILE_SIZE_X = cute::Int<64>; // bN
    using TILE_SIZE_Y = cute::Int<32>; // bM

    constexpr auto block_shape{cute::make_shape(TILE_SIZE_Y{}, TILE_SIZE_X{})};
    constexpr auto block_shape_transposed{
        cute::make_shape(TILE_SIZE_X{}, TILE_SIZE_Y{})};

    auto const shared_memory_layout_src{cute::make_layout(
        block_shape, cute::GenRowMajor{})}; // (bM, bN) : (bN, 1)
    auto const shared_memory_layout_dst{cute::make_layout(
        block_shape_transposed, cute::GenRowMajor{})}; // (bN, bM) : (bM, 1)
    auto const shared_memory_layout_dst_transposed{cute::make_layout(
        block_shape, cute::GenColMajor{})}; // (bM, bN) : (1, bM)

    auto const tiled_tensor_src{cute::tiled_divide(
        tensor_src, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), M /
                                   // TILE_SIZE_Y, N / TILE_SIZE_X)
    auto const tiled_tensor_dst{cute::tiled_divide(
        tensor_dst, block_shape_transposed)}; // ((TILE_SIZE_X, TILE_SIZE_Y), N
                                              // / TILE_SIZE_X, M / TILE_SIZE_Y)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), M
                                              // / TILE_SIZE_Y, N / TILE_SIZE_X)

    using THREAD_BLOCK_SIZE_X = cute::Int<32>; // tN
    using THREAD_BLOCK_SIZE_Y = cute::Int<8>;  // tM

    CUTE_STATIC_ASSERT(TILE_SIZE_X::value % THREAD_BLOCK_SIZE_X::value == 0,
                       "TILE_SIZE_X must be divisible by THREAD_BLOCK_SIZE_X");
    CUTE_STATIC_ASSERT(TILE_SIZE_Y::value % THREAD_BLOCK_SIZE_Y::value == 0,
                       "TILE_SIZE_Y must be divisible by THREAD_BLOCK_SIZE_Y");

    constexpr auto thread_block_shape{
        cute::make_shape(THREAD_BLOCK_SIZE_Y{}, THREAD_BLOCK_SIZE_X{})};
    constexpr auto thread_block_shape_transposed{
        cute::make_shape(THREAD_BLOCK_SIZE_X{}, THREAD_BLOCK_SIZE_Y{})};
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};
    constexpr auto thread_layout_transposed{
        cute::make_layout(thread_block_shape_transposed, cute::GenColMajor{})};

    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{
        cute::size(THREAD_BLOCK_SIZE_X::value * THREAD_BLOCK_SIZE_Y::value)};

    if (bank_conflict_access_mode == SharedMemoryBankConflictAccessMode::Read)
    {
        transpose_shared_memory<<<grid_dim, thread_dim, 0, stream>>>(
            tiled_tensor_src, tiled_tensor_dst_transposed,
            shared_memory_layout_src, shared_memory_layout_src, thread_layout,
            thread_layout_transposed);
    }
    else
    {
        transpose_shared_memory<<<grid_dim, thread_dim, 0, stream>>>(
            tiled_tensor_src, tiled_tensor_dst_transposed,
            shared_memory_layout_dst_transposed,
            shared_memory_layout_dst_transposed, thread_layout,
            thread_layout_transposed);
    }

    return cudaGetLastError();
}

template <typename T>
cudaError_t launch_transpose_shared_memory_bank_conflict_read(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    cudaStream_t stream)
{
    return launch_transpose_shared_memory_bank_conflict_base(
        input_matrix, output_matrix, M, N,
        SharedMemoryBankConflictAccessMode::Read, stream);
}

template <typename T>
cudaError_t launch_transpose_shared_memory_bank_conflict_write(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    cudaStream_t stream)
{
    return launch_transpose_shared_memory_bank_conflict_base(
        input_matrix, output_matrix, M, N,
        SharedMemoryBankConflictAccessMode::Write, stream);
}

template <typename T>
cudaError_t
launch_transpose_shared_memory_padded(T const* input_matrix, T* output_matrix,
                                      unsigned int M, unsigned int N,
                                      cudaStream_t stream)
{
    auto const tensor_shape{cute::make_shape(M, N)};
    auto const tensor_shape_transposed{cute::make_shape(N, M)};

    // Input matrix: row-major M x N matrix.
    auto const global_memory_layout_src{cute::make_layout(
        tensor_shape, cute::GenRowMajor{})}; // (M, N) : (N, 1)
    // Output matrix: row-major N x M matrix.
    auto const global_memory_layout_dst{cute::make_layout(
        tensor_shape_transposed, cute::GenRowMajor{})}; // (N, M) : (M, 1)
    // Same output matrix, but different view: column-major M x N matrix.
    auto const global_memory_layout_dst_transposed{cute::make_layout(
        tensor_shape, cute::GenColMajor{})}; // (M, N) : (1, M)

    auto const tensor_src{cute::make_tensor(cute::make_gmem_ptr(input_matrix),
                                            global_memory_layout_src)};
    auto const tensor_dst{cute::make_tensor(cute::make_gmem_ptr(output_matrix),
                                            global_memory_layout_dst)};
    auto const tensor_dst_transposed{
        cute::make_tensor(cute::make_gmem_ptr(output_matrix),
                          global_memory_layout_dst_transposed)};

    using TILE_SIZE_X = cute::Int<64>;        // bN
    using TILE_SIZE_X_PADDED = cute::Int<65>; // bN + 1
    using TILE_SIZE_Y = cute::Int<32>;        // bM

    constexpr auto block_shape{cute::make_shape(TILE_SIZE_Y{}, TILE_SIZE_X{})};
    constexpr auto block_shape_transposed{
        cute::make_shape(TILE_SIZE_X{}, TILE_SIZE_Y{})};

    auto const shared_memory_layout_src{cute::make_layout(
        block_shape, cute::GenRowMajor{})}; // (bM, bN) : (bN, 1)
    auto const shared_memory_layout_src_padded{cute::make_layout(
        block_shape,
        cute::make_stride(TILE_SIZE_X_PADDED{},
                          cute::Int<1>{}))}; // (bM, bN) : (bN + 1, 1)
    auto const shared_memory_layout_dst{cute::make_layout(
        block_shape_transposed, cute::GenRowMajor{})}; // (bN, bM) : (bM, 1)
    auto const shared_memory_layout_dst_transposed{cute::make_layout(
        block_shape, cute::GenColMajor{})}; // (bM, bN) : (1, bM)

    auto const tiled_tensor_src{cute::tiled_divide(
        tensor_src, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), M /
                                   // TILE_SIZE_Y, N / TILE_SIZE_X)
    auto const tiled_tensor_dst{cute::tiled_divide(
        tensor_dst, block_shape_transposed)}; // ((TILE_SIZE_X, TILE_SIZE_Y), N
                                              // / TILE_SIZE_X, M / TILE_SIZE_Y)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), M
                                              // / TILE_SIZE_Y, N / TILE_SIZE_X)

    using THREAD_BLOCK_SIZE_X = cute::Int<32>; // tN
    using THREAD_BLOCK_SIZE_Y = cute::Int<8>;  // tM

    CUTE_STATIC_ASSERT(TILE_SIZE_X::value % THREAD_BLOCK_SIZE_X::value == 0,
                       "TILE_SIZE_X must be divisible by THREAD_BLOCK_SIZE_X");
    CUTE_STATIC_ASSERT(TILE_SIZE_Y::value % THREAD_BLOCK_SIZE_Y::value == 0,
                       "TILE_SIZE_Y must be divisible by THREAD_BLOCK_SIZE_Y");

    constexpr auto thread_block_shape{
        cute::make_shape(THREAD_BLOCK_SIZE_Y{}, THREAD_BLOCK_SIZE_X{})};
    constexpr auto thread_block_shape_transposed{
        cute::make_shape(THREAD_BLOCK_SIZE_X{}, THREAD_BLOCK_SIZE_Y{})};
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};
    constexpr auto thread_layout_transposed{
        cute::make_layout(thread_block_shape_transposed, cute::GenColMajor{})};

    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{
        cute::size(THREAD_BLOCK_SIZE_X::value * THREAD_BLOCK_SIZE_Y::value)};

    transpose_shared_memory<<<grid_dim, thread_dim, 0, stream>>>(
        tiled_tensor_src, tiled_tensor_dst_transposed,
        shared_memory_layout_src_padded, shared_memory_layout_src_padded,
        thread_layout, thread_layout_transposed);

    return cudaGetLastError();
}

template <class SHARED_MEMORY_LAYOUT>
void print_shared_memory_bank_ids(SHARED_MEMORY_LAYOUT shared_memory_layout)
{
    // Print the shared memory bank ids.
    for (unsigned int i{0}; i < cute::size<0>(shared_memory_layout); ++i)
    {
        for (unsigned int j{0}; j < cute::size<1>(shared_memory_layout); ++j)
        {
            std::cout << std::setw(2) << shared_memory_layout(i, j) % 32 << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
cudaError_t
launch_transpose_shared_memory_swizzled(T const* input_matrix, T* output_matrix,
                                        unsigned int M, unsigned int N,
                                        cudaStream_t stream)
{
    auto const tensor_shape{cute::make_shape(M, N)};
    auto const tensor_shape_transposed{cute::make_shape(N, M)};

    // Input matrix: row-major M x N matrix.
    auto const global_memory_layout_src{cute::make_layout(
        tensor_shape, cute::GenRowMajor{})}; // (M, N) : (N, 1)
    // Output matrix: row-major N x M matrix.
    auto const global_memory_layout_dst{cute::make_layout(
        tensor_shape_transposed, cute::GenRowMajor{})}; // (N, M) : (M, 1)
    // Same output matrix, but different view: column-major M x N matrix.
    auto const global_memory_layout_dst_transposed{cute::make_layout(
        tensor_shape, cute::GenColMajor{})}; // (M, N) : (1, M)

    auto const tensor_src{cute::make_tensor(cute::make_gmem_ptr(input_matrix),
                                            global_memory_layout_src)};
    auto const tensor_dst{cute::make_tensor(cute::make_gmem_ptr(output_matrix),
                                            global_memory_layout_dst)};
    auto const tensor_dst_transposed{
        cute::make_tensor(cute::make_gmem_ptr(output_matrix),
                          global_memory_layout_dst_transposed)};

    using TILE_SIZE_X = cute::Int<64>; // bN
    using TILE_SIZE_Y = cute::Int<32>; // bM
    constexpr int NUM_BITS_X{6};
    constexpr int NUM_BITS_Y{5};

    constexpr auto block_shape{cute::make_shape(TILE_SIZE_Y{}, TILE_SIZE_X{})};
    constexpr auto block_shape_transposed{
        cute::make_shape(TILE_SIZE_X{}, TILE_SIZE_Y{})};

    auto const shared_memory_layout_src{cute::make_layout(
        block_shape, cute::GenRowMajor{})}; // (bM, bN) : (bN, 1)
    auto const shared_memory_layout_dst{cute::make_layout(
        block_shape_transposed, cute::GenRowMajor{})}; // (bN, bM) : (bM, 1)
    auto const shared_memory_layout_dst_transposed{cute::make_layout(
        block_shape, cute::GenColMajor{})}; // (bM, bN) : (1, bM)

    auto const swizzle_src{cute::Swizzle<NUM_BITS_Y, 0, NUM_BITS_X>{}};
    auto const shared_memory_layout_swizzled_src{
        cute::composition(swizzle_src, shared_memory_layout_src)};

    // Inspect if the swizzling reduces the shared memory bank conflicts.
    // print_shared_memory_bank_ids(shared_memory_layout_swizzled_src);

    auto const tiled_tensor_src{cute::tiled_divide(
        tensor_src, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), M /
                                   // TILE_SIZE_Y, N / TILE_SIZE_X)
    auto const tiled_tensor_dst{cute::tiled_divide(
        tensor_dst, block_shape_transposed)}; // ((TILE_SIZE_X, TILE_SIZE_Y), N
                                              // / TILE_SIZE_X, M / TILE_SIZE_Y)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), M
                                              // / TILE_SIZE_Y, N / TILE_SIZE_X)

    using THREAD_BLOCK_SIZE_X = cute::Int<32>; // tN
    using THREAD_BLOCK_SIZE_Y = cute::Int<8>;  // tM

    CUTE_STATIC_ASSERT(TILE_SIZE_X::value % THREAD_BLOCK_SIZE_X::value == 0,
                       "TILE_SIZE_X must be divisible by THREAD_BLOCK_SIZE_X");
    CUTE_STATIC_ASSERT(TILE_SIZE_Y::value % THREAD_BLOCK_SIZE_Y::value == 0,
                       "TILE_SIZE_Y must be divisible by THREAD_BLOCK_SIZE_Y");

    constexpr auto thread_block_shape{
        cute::make_shape(THREAD_BLOCK_SIZE_Y{}, THREAD_BLOCK_SIZE_X{})};
    constexpr auto thread_block_shape_transposed{
        cute::make_shape(THREAD_BLOCK_SIZE_X{}, THREAD_BLOCK_SIZE_Y{})};
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};
    constexpr auto thread_layout_transposed{
        cute::make_layout(thread_block_shape_transposed, cute::GenColMajor{})};

    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{
        cute::size(THREAD_BLOCK_SIZE_X::value * THREAD_BLOCK_SIZE_Y::value)};

    transpose_shared_memory<<<grid_dim, thread_dim, 0, stream>>>(
        tiled_tensor_src, tiled_tensor_dst_transposed,
        shared_memory_layout_swizzled_src, shared_memory_layout_swizzled_src,
        thread_layout, thread_layout_transposed);

    return cudaGetLastError();
}

// Explicit instantiation.
template cudaError_t launch_transpose_shared_memory_bank_conflict_read<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t launch_transpose_shared_memory_bank_conflict_read<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

template cudaError_t launch_transpose_shared_memory_bank_conflict_write<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t launch_transpose_shared_memory_bank_conflict_write<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

template cudaError_t launch_transpose_shared_memory_padded<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t launch_transpose_shared_memory_padded<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

template cudaError_t launch_transpose_shared_memory_swizzled<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t launch_transpose_shared_memory_swizzled<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

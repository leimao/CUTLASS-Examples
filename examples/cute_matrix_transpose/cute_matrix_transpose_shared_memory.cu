#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "cute_matrix_transpose.hpp"

template <class TensorSrc, class TensorDst, class SharedMemoryLayoutSrc,
          class SharedMemoryLayoutDst, class ThreadLayoutSrc,
          class ThreadLayoutDst>
__global__ void static matrix_transpose_shared_memory(
    TensorSrc tensor_src, TensorDst tensor_dst, SharedMemoryLayoutSrc,
    SharedMemoryLayoutDst, ThreadLayoutSrc, ThreadLayoutDst)
{
    using Element = typename TensorSrc::value_type;
    CUTE_STATIC_ASSERT_V(cute::size(SharedMemoryLayoutSrc{}) ==
                             cute::size(SharedMemoryLayoutDst{}),
                         "SharedMemoryLayoutSrc and SharedMemoryLayoutDst "
                         "must have the same size.");
    __shared__ Element shared_memory[cute::cosize(SharedMemoryLayoutSrc{})];

    auto tensor_cache_src{cute::make_tensor(cute::make_smem_ptr(shared_memory),
                                            SharedMemoryLayoutSrc{})};
    auto tensor_cache_dst{cute::make_tensor(cute::make_smem_ptr(shared_memory),
                                            SharedMemoryLayoutDst{})};

    auto global_tile_src{tensor_src(cute::make_coord(cute::_, cute::_),
                                    blockIdx.y,
                                    blockIdx.x)}; // (TileSizeY, TileSizeX)
    auto global_tile_dst{tensor_dst(cute::make_coord(cute::_, cute::_),
                                    blockIdx.y,
                                    blockIdx.x)}; // (TileSizeY, TileSizeX)

    auto thread_global_tile_src{cute::local_partition(
        global_tile_src, ThreadLayoutSrc{},
        threadIdx.x)}; // (ThreadValueSizeY, ThreadValueSizeX)
    auto thread_global_tile_dst{cute::local_partition(
        global_tile_dst, ThreadLayoutDst{},
        threadIdx.x)}; // (ThreadValueSizeX, ThreadValueSizeY)

    auto thread_shared_tile_src{cute::local_partition(
        tensor_cache_src, ThreadLayoutSrc{},
        threadIdx.x)}; // (ThreadValueSizeY, ThreadValueSizeX)
    auto thread_shared_tile_dst{cute::local_partition(
        tensor_cache_dst, ThreadLayoutDst{},
        threadIdx.x)}; // (ThreadValueSizeX, ThreadValueSizeY)

    // A 2D array of tuples that maps (x, y) to (x, y).
    auto const identity_tensor_src{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(global_tile_src), cute::size<1>(global_tile_src)))};
    auto const thread_identity_tensor_src{cute::local_partition(
        identity_tensor_src, ThreadLayoutSrc{}, threadIdx.x)};
    auto predicator_src{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(thread_global_tile_src),
                         cute::size<1>(thread_global_tile_src)))};

    auto const identity_tensor_dst{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(global_tile_dst), cute::size<1>(global_tile_dst)))};
    auto const thread_identity_tensor_dst{cute::local_partition(
        identity_tensor_dst, ThreadLayoutDst{}, threadIdx.x)};
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

template <class TensorSrc, class TensorDst, class SharedMemoryLayoutSrc,
          class SharedMemoryLayoutDst, class ThreadLayoutSrc,
          class ThreadLayoutDst, class VectorLayout>
__global__ void static matrix_transpose_shared_memory_vectorized(
    TensorSrc tensor_src, TensorDst tensor_dst, SharedMemoryLayoutSrc,
    SharedMemoryLayoutDst, ThreadLayoutSrc, ThreadLayoutDst, VectorLayout)
{
    using Element = typename TensorSrc::value_type;
    CUTE_STATIC_ASSERT_V(cute::size(SharedMemoryLayoutSrc{}) ==
                             cute::size(SharedMemoryLayoutDst{}),
                         "SharedMemoryLayoutSrc and SharedMemoryLayoutDst "
                         "must have the same size.");
    __shared__ Element shared_memory[cute::cosize(SharedMemoryLayoutSrc{})];

    auto tensor_cache_src{cute::make_tensor(cute::make_smem_ptr(shared_memory),
                                            SharedMemoryLayoutSrc{})};
    auto tensor_cache_dst{cute::make_tensor(cute::make_smem_ptr(shared_memory),
                                            SharedMemoryLayoutDst{})};

    auto global_tile_src{tensor_src(cute::make_coord(cute::_, cute::_),
                                    blockIdx.y,
                                    blockIdx.x)}; // (TileSizeY, TileSizeX)
    auto global_tile_dst{tensor_dst(cute::make_coord(cute::_, cute::_),
                                    blockIdx.y,
                                    blockIdx.x)}; // (TileSizeY, TileSizeX)

    using AccessType =
        cutlass::AlignedArray<Element, cute::size(VectorLayout{})>;
    using CopyAtom = cute::Copy_Atom<cute::UniversalCopy<AccessType>, Element>;
    auto tiled_copy_src{
        cute::make_tiled_copy(CopyAtom{}, ThreadLayoutSrc{}, VectorLayout{})};
    auto thread_copy_src{tiled_copy_src.get_thread_slice(threadIdx.x)};

    auto thread_global_tile_src{thread_copy_src.partition_S(
        global_tile_src)}; // (CopyAtomShape, NumCopyTile)
    auto thread_shared_tile_src{thread_copy_src.partition_D(
        tensor_cache_src)}; // (CopyAtomShape, NumCopyTile)

    auto thread_global_tile_dst{cute::local_partition(
        global_tile_dst, ThreadLayoutDst{},
        threadIdx.x)}; // (ThreadValueSizeX, ThreadValueSizeY)
    auto thread_shared_tile_dst{cute::local_partition(
        tensor_cache_dst, ThreadLayoutDst{},
        threadIdx.x)}; // (ThreadValueSizeX, ThreadValueSizeY)

    auto const num_max_columns{cute::stride<0>(global_tile_src)};
    auto const num_max_rows{cute::stride<1>(global_tile_dst)};
    constexpr auto global_tile_columns{cute::size<1>(global_tile_src)};
    constexpr auto global_tile_rows{cute::size<0>(global_tile_src)};

    // A 2D array of tuples that maps (x, y) to (x, y).
    auto const identity_tensor_src{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(global_tile_src), cute::size<1>(global_tile_src)))};
    auto thread_identity_tensor_src{thread_copy_src.partition_S(
        identity_tensor_src)}; // (CopyAtomShape, NumCopyTile)
    auto predicator_src{cute::make_tensor<bool>(
        cute::make_shape(cute::size<1>(thread_global_tile_src),
                         cute::size<2>(thread_global_tile_src)))};

    CUTE_UNROLL
    for (unsigned int i{0}; i < cute::size<0>(predicator_src); ++i)
    {
        CUTE_UNROLL
        for (unsigned int j{0}; j < cute::size<1>(predicator_src); ++j)
        {
            auto const thread_identity{thread_identity_tensor_src(0, i, j)};
            bool const is_row_in_bound{cute::get<0>(thread_identity) +
                                           blockIdx.y * global_tile_rows <
                                       num_max_rows};
            bool const is_column_in_bound{cute::get<1>(thread_identity) +
                                              blockIdx.x * global_tile_columns <
                                          num_max_columns};
            predicator_src(i, j) = is_row_in_bound && is_column_in_bound;
        }
    }

    auto const identity_tensor_dst{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(global_tile_dst), cute::size<1>(global_tile_dst)))};
    auto const thread_identity_tensor_dst{cute::local_partition(
        identity_tensor_dst, ThreadLayoutDst{}, threadIdx.x)};
    auto predicator_dst{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(thread_global_tile_dst),
                         cute::size<1>(thread_global_tile_dst)))};

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

    cute::copy_if(tiled_copy_src, predicator_src, thread_global_tile_src,
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
static cudaError_t launch_matrix_transpose_shared_memory_bank_conflict_base(
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

    using TileSizeX = cute::Int<128>; // bN
    using TileSizeY = cute::Int<32>;  // bM

    constexpr auto block_shape{cute::make_shape(TileSizeY{}, TileSizeX{})};
    constexpr auto block_shape_transposed{
        cute::make_shape(TileSizeX{}, TileSizeY{})};

    auto const shared_memory_layout_src{cute::make_layout(
        block_shape, cute::GenRowMajor{})}; // (bM, bN) : (bN, 1)
    auto const shared_memory_layout_dst{cute::make_layout(
        block_shape_transposed, cute::GenRowMajor{})}; // (bN, bM) : (bM, 1)
    auto const shared_memory_layout_dst_transposed{cute::make_layout(
        block_shape, cute::GenColMajor{})}; // (bM, bN) : (1, bM)

    auto const tiled_tensor_src{cute::tiled_divide(
        tensor_src, block_shape)}; // ((TileSizeY, TileSizeX), M /
                                   // TileSizeY, N / TileSizeX)
    auto const tiled_tensor_dst{cute::tiled_divide(
        tensor_dst, block_shape_transposed)}; // ((TileSizeX, TileSizeY), N
                                              // / TileSizeX, M / TileSizeY)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TileSizeY, TileSizeX), M
                                              // / TileSizeY, N / TileSizeX)

    using ThreadBlockSizeX = cute::Int<32>; // tN
    using ThreadBlockSizeY = cute::Int<8>;  // tM

    CUTE_STATIC_ASSERT_V(TileSizeX{} % ThreadBlockSizeX{} == cute::Int<0>{},
                         "TileSizeX must be divisible by ThreadBlockSizeX");
    CUTE_STATIC_ASSERT_V(TileSizeY{} % ThreadBlockSizeY{} == cute::Int<0>{},
                         "TileSizeY must be divisible by ThreadBlockSizeY");

    constexpr auto thread_block_shape{
        cute::make_shape(ThreadBlockSizeY{}, ThreadBlockSizeX{})};
    constexpr auto thread_block_shape_transposed{
        cute::make_shape(ThreadBlockSizeX{}, ThreadBlockSizeY{})};
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};
    constexpr auto thread_layout_transposed{
        cute::make_layout(thread_block_shape_transposed, cute::GenColMajor{})};

    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{ThreadBlockSizeX::value * ThreadBlockSizeY::value};

    if (bank_conflict_access_mode == SharedMemoryBankConflictAccessMode::Read)
    {
        matrix_transpose_shared_memory<<<grid_dim, thread_dim, 0, stream>>>(
            tiled_tensor_src, tiled_tensor_dst_transposed,
            shared_memory_layout_src, shared_memory_layout_src, thread_layout,
            thread_layout_transposed);
    }
    else
    {
        matrix_transpose_shared_memory<<<grid_dim, thread_dim, 0, stream>>>(
            tiled_tensor_src, tiled_tensor_dst_transposed,
            shared_memory_layout_dst_transposed,
            shared_memory_layout_dst_transposed, thread_layout,
            thread_layout_transposed);
    }

    return cudaGetLastError();
}

template <typename T>
static cudaError_t
launch_matrix_transpose_shared_memory_vectorized_bank_conflict_base(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    SharedMemoryBankConflictAccessMode bank_conflict_access_mode,
    cudaStream_t stream)
{
    using VectorType = cute::uint128_t;
    CUTE_STATIC_ASSERT(sizeof(VectorType) % sizeof(T) == 0,
                       "sizeof(VectorType) must be a multiple of sizeof(T)");
    constexpr unsigned int NUM_VECTOR_ELEMENTS{sizeof(VectorType) / sizeof(T)};

    if (N % NUM_VECTOR_ELEMENTS != 0)
    {
        return cudaErrorInvalidValue;
    }

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

    using TileSizeX = cute::Int<128>; // bN
    using TileSizeY = cute::Int<32>;  // bM

    constexpr auto block_shape{cute::make_shape(TileSizeY{}, TileSizeX{})};
    constexpr auto block_shape_transposed{
        cute::make_shape(TileSizeX{}, TileSizeY{})};

    auto const shared_memory_layout_src{cute::make_layout(
        block_shape, cute::GenRowMajor{})}; // (bM, bN) : (bN, 1)
    auto const shared_memory_layout_dst{cute::make_layout(
        block_shape_transposed, cute::GenRowMajor{})}; // (bN, bM) : (bM, 1)
    auto const shared_memory_layout_dst_transposed{cute::make_layout(
        block_shape, cute::GenColMajor{})}; // (bM, bN) : (1, bM)

    auto const tiled_tensor_src{cute::tiled_divide(
        tensor_src, block_shape)}; // ((TileSizeY, TileSizeX), M /
                                   // TileSizeY, N / TileSizeX)
    auto const tiled_tensor_dst{cute::tiled_divide(
        tensor_dst, block_shape_transposed)}; // ((TileSizeX, TileSizeY), N
                                              // / TileSizeX, M / TileSizeY)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TileSizeY, TileSizeX), M
                                              // / TileSizeY, N / TileSizeX)

    using ThreadBlockSizeX = cute::Int<32>; // tN
    using ThreadBlockSizeY = cute::Int<8>;  // tM

    CUTE_STATIC_ASSERT_V(TileSizeX{} % ThreadBlockSizeX{} == cute::Int<0>{},
                         "TileSizeX must be divisible by ThreadBlockSizeX");
    CUTE_STATIC_ASSERT_V(TileSizeY{} % ThreadBlockSizeY{} == cute::Int<0>{},
                         "TileSizeY must be divisible by ThreadBlockSizeY");

    constexpr auto thread_block_shape{
        cute::make_shape(ThreadBlockSizeY{}, ThreadBlockSizeX{})};
    constexpr auto thread_block_shape_transposed{
        cute::make_shape(ThreadBlockSizeX{}, ThreadBlockSizeY{})};
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};
    constexpr auto thread_layout_transposed{
        cute::make_layout(thread_block_shape_transposed, cute::GenColMajor{})};

    using VECTOR_SIZE_X = cute::Int<NUM_VECTOR_ELEMENTS>;
    constexpr auto vector_shape{
        cute::make_shape(cute::Int<1>{}, VECTOR_SIZE_X{})};
    // Copy atom vector layout.
    constexpr auto vector_layout{
        cute::make_layout(vector_shape, cute::GenRowMajor{})};

    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{ThreadBlockSizeX::value * ThreadBlockSizeY::value};

    if (bank_conflict_access_mode == SharedMemoryBankConflictAccessMode::Read)
    {
        matrix_transpose_shared_memory_vectorized<<<grid_dim, thread_dim, 0,
                                                    stream>>>(
            tiled_tensor_src, tiled_tensor_dst_transposed,
            shared_memory_layout_src, shared_memory_layout_src, thread_layout,
            thread_layout_transposed, vector_layout);
    }
    else
    {
        return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

template <typename T>
cudaError_t launch_matrix_transpose_shared_memory_bank_conflict_read(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    cudaStream_t stream)
{
    return launch_matrix_transpose_shared_memory_bank_conflict_base(
        input_matrix, output_matrix, M, N,
        SharedMemoryBankConflictAccessMode::Read, stream);
}

template <typename T>
cudaError_t launch_matrix_transpose_shared_memory_bank_conflict_write(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    cudaStream_t stream)
{
    return launch_matrix_transpose_shared_memory_bank_conflict_base(
        input_matrix, output_matrix, M, N,
        SharedMemoryBankConflictAccessMode::Write, stream);
}

template <typename T>
cudaError_t launch_matrix_transpose_shared_memory_vectorized_bank_conflict_read(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    cudaStream_t stream)
{
    return launch_matrix_transpose_shared_memory_vectorized_bank_conflict_base<
        T>(input_matrix, output_matrix, M, N,
           SharedMemoryBankConflictAccessMode::Read, stream);
}

template <typename T>
cudaError_t launch_matrix_transpose_shared_memory_padded(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
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

    using TileSizeX = cute::Int<64>;          // bN
    using TILE_SIZE_X_PADDED = cute::Int<65>; // bN + 1
    using TileSizeY = cute::Int<32>;          // bM

    constexpr auto block_shape{cute::make_shape(TileSizeY{}, TileSizeX{})};
    constexpr auto block_shape_transposed{
        cute::make_shape(TileSizeX{}, TileSizeY{})};

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
        tensor_src, block_shape)}; // ((TileSizeY, TileSizeX), M /
                                   // TileSizeY, N / TileSizeX)
    auto const tiled_tensor_dst{cute::tiled_divide(
        tensor_dst, block_shape_transposed)}; // ((TileSizeX, TileSizeY), N
                                              // / TileSizeX, M / TileSizeY)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TileSizeY, TileSizeX), M
                                              // / TileSizeY, N / TileSizeX)

    using ThreadBlockSizeX = cute::Int<32>; // tN
    using ThreadBlockSizeY = cute::Int<8>;  // tM

    CUTE_STATIC_ASSERT_V(TileSizeX{} % ThreadBlockSizeX{} == cute::Int<0>{},
                         "TileSizeX must be divisible by ThreadBlockSizeX");
    CUTE_STATIC_ASSERT_V(TileSizeY{} % ThreadBlockSizeY{} == cute::Int<0>{},
                         "TileSizeY must be divisible by ThreadBlockSizeY");

    constexpr auto thread_block_shape{
        cute::make_shape(ThreadBlockSizeY{}, ThreadBlockSizeX{})};
    constexpr auto thread_block_shape_transposed{
        cute::make_shape(ThreadBlockSizeX{}, ThreadBlockSizeY{})};
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};
    constexpr auto thread_layout_transposed{
        cute::make_layout(thread_block_shape_transposed, cute::GenColMajor{})};

    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{ThreadBlockSizeX::value * ThreadBlockSizeY::value};

    matrix_transpose_shared_memory<<<grid_dim, thread_dim, 0, stream>>>(
        tiled_tensor_src, tiled_tensor_dst_transposed,
        shared_memory_layout_src_padded, shared_memory_layout_src_padded,
        thread_layout, thread_layout_transposed);

    return cudaGetLastError();
}

template <typename T>
cudaError_t launch_matrix_transpose_shared_memory_vectorized_padded(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    cudaStream_t stream)
{
    using VectorType = cute::uint128_t;
    CUTE_STATIC_ASSERT(sizeof(VectorType) % sizeof(T) == 0,
                       "sizeof(VectorType) must be a multiple of sizeof(T)");
    constexpr unsigned int NUM_VECTOR_ELEMENTS{sizeof(VectorType) / sizeof(T)};

    if (N % NUM_VECTOR_ELEMENTS != 0)
    {
        return cudaErrorInvalidValue;
    }

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

    using TileSizeX = cute::Int<128>; // bN
    // Such padding is necessary for the byte alignment of the vectorized
    // access. However, the shared memory bank conflict mitigation can be
    // compromised.
    using TILE_SIZE_X_PADDED =
        cute::Int<128 + NUM_VECTOR_ELEMENTS>; // bN + NUM_VECTOR_ELEMENTS
    using TileSizeY = cute::Int<32>;          // bM

    constexpr auto block_shape{cute::make_shape(TileSizeY{}, TileSizeX{})};
    constexpr auto block_shape_transposed{
        cute::make_shape(TileSizeX{}, TileSizeY{})};

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
        tensor_src, block_shape)}; // ((TileSizeY, TileSizeX), M /
                                   // TileSizeY, N / TileSizeX)
    auto const tiled_tensor_dst{cute::tiled_divide(
        tensor_dst, block_shape_transposed)}; // ((TileSizeX, TileSizeY), N
                                              // / TileSizeX, M / TileSizeY)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TileSizeY, TileSizeX), M
                                              // / TileSizeY, N / TileSizeX)

    using ThreadBlockSizeX = cute::Int<32>; // tN
    using ThreadBlockSizeY = cute::Int<8>;  // tM

    CUTE_STATIC_ASSERT_V(TileSizeX{} % ThreadBlockSizeX{} == cute::Int<0>{},
                         "TileSizeX must be divisible by ThreadBlockSizeX");
    CUTE_STATIC_ASSERT_V(TileSizeY{} % ThreadBlockSizeY{} == cute::Int<0>{},
                         "TileSizeY must be divisible by ThreadBlockSizeY");

    constexpr auto thread_block_shape{
        cute::make_shape(ThreadBlockSizeY{}, ThreadBlockSizeX{})};
    constexpr auto thread_block_shape_transposed{
        cute::make_shape(ThreadBlockSizeX{}, ThreadBlockSizeY{})};
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};
    constexpr auto thread_layout_transposed{
        cute::make_layout(thread_block_shape_transposed, cute::GenColMajor{})};

    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{ThreadBlockSizeX::value * ThreadBlockSizeY::value};

    using VECTOR_SIZE_X = cute::Int<NUM_VECTOR_ELEMENTS>;
    constexpr auto vector_shape{
        cute::make_shape(cute::Int<1>{}, VECTOR_SIZE_X{})};
    // Copy atom vector layout.
    constexpr auto vector_layout{
        cute::make_layout(vector_shape, cute::GenRowMajor{})};

    matrix_transpose_shared_memory_vectorized<<<grid_dim, thread_dim, 0,
                                                stream>>>(
        tiled_tensor_src, tiled_tensor_dst_transposed,
        shared_memory_layout_src_padded, shared_memory_layout_src_padded,
        thread_layout, thread_layout_transposed, vector_layout);

    return cudaGetLastError();
}

template <class SHARED_MEMORY_LAYOUT>
static void
print_shared_memory_bank_ids(SHARED_MEMORY_LAYOUT shared_memory_layout)
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

constexpr int constexpr_log2(int n)
{
    return ((n < 2) ? 0 : 1 + constexpr_log2(n / 2));
}

template <typename T>
cudaError_t launch_matrix_transpose_shared_memory_swizzled(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
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

    using TileSizeX = cute::Int<64>; // bN
    using TileSizeY = cute::Int<32>; // bM
    constexpr int NUM_BASE_BITS{constexpr_log2(1)};
    constexpr int NUM_MASK_BITS{constexpr_log2(32 * 4 / sizeof(T)) -
                                NUM_BASE_BITS};
    constexpr int NUM_SHIFT_BITS{constexpr_log2(TileSizeX::value) -
                                 NUM_BASE_BITS};

    constexpr auto block_shape{cute::make_shape(TileSizeY{}, TileSizeX{})};
    constexpr auto block_shape_transposed{
        cute::make_shape(TileSizeX{}, TileSizeY{})};

    auto const shared_memory_layout_src{cute::make_layout(
        block_shape, cute::GenRowMajor{})}; // (bM, bN) : (bN, 1)
    auto const shared_memory_layout_dst{cute::make_layout(
        block_shape_transposed, cute::GenRowMajor{})}; // (bN, bM) : (bM, 1)
    auto const shared_memory_layout_dst_transposed{cute::make_layout(
        block_shape, cute::GenColMajor{})}; // (bM, bN) : (1, bM)

    auto const swizzle_src{
        cute::Swizzle<NUM_MASK_BITS, NUM_BASE_BITS, NUM_SHIFT_BITS>{}};
    auto const shared_memory_layout_swizzled_src{
        cute::composition(swizzle_src, shared_memory_layout_src)};

    // Inspect if the swizzling reduces the shared memory bank conflicts.
    // print_shared_memory_bank_ids(shared_memory_layout_swizzled_src);

    auto const tiled_tensor_src{cute::tiled_divide(
        tensor_src, block_shape)}; // ((TileSizeY, TileSizeX), M /
                                   // TileSizeY, N / TileSizeX)
    auto const tiled_tensor_dst{cute::tiled_divide(
        tensor_dst, block_shape_transposed)}; // ((TileSizeX, TileSizeY), N
                                              // / TileSizeX, M / TileSizeY)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TileSizeY, TileSizeX), M
                                              // / TileSizeY, N / TileSizeX)

    using ThreadBlockSizeX = cute::Int<32>; // tN
    using ThreadBlockSizeY = cute::Int<8>;  // tM

    CUTE_STATIC_ASSERT_V(TileSizeX{} % ThreadBlockSizeX{} == cute::Int<0>{},
                         "TileSizeX must be divisible by ThreadBlockSizeX");
    CUTE_STATIC_ASSERT_V(TileSizeY{} % ThreadBlockSizeY{} == cute::Int<0>{},
                         "TileSizeY must be divisible by ThreadBlockSizeY");

    constexpr auto thread_block_shape{
        cute::make_shape(ThreadBlockSizeY{}, ThreadBlockSizeX{})};
    constexpr auto thread_block_shape_transposed{
        cute::make_shape(ThreadBlockSizeX{}, ThreadBlockSizeY{})};
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};
    constexpr auto thread_layout_transposed{
        cute::make_layout(thread_block_shape_transposed, cute::GenColMajor{})};

    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{ThreadBlockSizeX::value * ThreadBlockSizeY::value};

    matrix_transpose_shared_memory<<<grid_dim, thread_dim, 0, stream>>>(
        tiled_tensor_src, tiled_tensor_dst_transposed,
        shared_memory_layout_swizzled_src, shared_memory_layout_swizzled_src,
        thread_layout, thread_layout_transposed);

    return cudaGetLastError();
}

template <typename T>
cudaError_t launch_matrix_transpose_shared_memory_vectorized_swizzled(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    cudaStream_t stream)
{
    using VectorType = cute::uint128_t;
    CUTE_STATIC_ASSERT(sizeof(VectorType) % sizeof(T) == 0,
                       "sizeof(VectorType) must be a multiple of sizeof(T)");
    constexpr unsigned int NUM_VECTOR_ELEMENTS{sizeof(VectorType) / sizeof(T)};

    if (N % NUM_VECTOR_ELEMENTS != 0)
    {
        return cudaErrorInvalidValue;
    }

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

    using TileSizeX = cute::Int<128>; // bN
    using TileSizeY = cute::Int<32>;  // bM
    constexpr int NUM_BASE_BITS{constexpr_log2(NUM_VECTOR_ELEMENTS)};
    constexpr int NUM_MASK_BITS{constexpr_log2(32 * 4 / sizeof(T)) -
                                NUM_BASE_BITS};
    constexpr int NUM_SHIFT_BITS{constexpr_log2(TileSizeX::value) -
                                 NUM_BASE_BITS};

    constexpr auto block_shape{cute::make_shape(TileSizeY{}, TileSizeX{})};
    constexpr auto block_shape_transposed{
        cute::make_shape(TileSizeX{}, TileSizeY{})};

    auto const shared_memory_layout_src{cute::make_layout(
        block_shape, cute::GenRowMajor{})}; // (bM, bN) : (bN, 1)
    auto const shared_memory_layout_dst{cute::make_layout(
        block_shape_transposed, cute::GenRowMajor{})}; // (bN, bM) : (bM, 1)
    auto const shared_memory_layout_dst_transposed{cute::make_layout(
        block_shape, cute::GenColMajor{})}; // (bM, bN) : (1, bM)

    // Because of the vectorized access, NUM_BASE_BITS cannot be zero.
    // The shared memory bank conflict mitigation can be compromised.
    // Print the shared memory bank ids to see the details.
    auto const swizzle_src{
        cute::Swizzle<NUM_MASK_BITS, NUM_BASE_BITS, NUM_SHIFT_BITS>{}};
    auto const shared_memory_layout_swizzled_src{
        cute::composition(swizzle_src, shared_memory_layout_src)};

    // Inspect if the swizzling reduces the shared memory bank conflicts.
    // print_shared_memory_bank_ids(shared_memory_layout_swizzled_src);

    auto const tiled_tensor_src{cute::tiled_divide(
        tensor_src, block_shape)}; // ((TileSizeY, TileSizeX), M /
                                   // TileSizeY, N / TileSizeX)
    auto const tiled_tensor_dst{cute::tiled_divide(
        tensor_dst, block_shape_transposed)}; // ((TileSizeX, TileSizeY), N
                                              // / TileSizeX, M / TileSizeY)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TileSizeY, TileSizeX), M
                                              // / TileSizeY, N / TileSizeX)

    using ThreadBlockSizeX = cute::Int<32>; // tN
    using ThreadBlockSizeY = cute::Int<8>;  // tM

    CUTE_STATIC_ASSERT_V(TileSizeX{} % ThreadBlockSizeX{} == cute::Int<0>{},
                         "TileSizeX must be divisible by ThreadBlockSizeX");
    CUTE_STATIC_ASSERT_V(TileSizeY{} % ThreadBlockSizeY{} == cute::Int<0>{},
                         "TileSizeY must be divisible by ThreadBlockSizeY");

    constexpr auto thread_block_shape{
        cute::make_shape(ThreadBlockSizeY{}, ThreadBlockSizeX{})};
    constexpr auto thread_block_shape_transposed{
        cute::make_shape(ThreadBlockSizeX{}, ThreadBlockSizeY{})};
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};
    constexpr auto thread_layout_transposed{
        cute::make_layout(thread_block_shape_transposed, cute::GenColMajor{})};

    using VECTOR_SIZE_X = cute::Int<NUM_VECTOR_ELEMENTS>;
    constexpr auto vector_shape{
        cute::make_shape(cute::Int<1>{}, VECTOR_SIZE_X{})};
    // Copy atom vector layout.
    constexpr auto vector_layout{
        cute::make_layout(vector_shape, cute::GenRowMajor{})};

    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{ThreadBlockSizeX::value * ThreadBlockSizeY::value};

    matrix_transpose_shared_memory_vectorized<<<grid_dim, thread_dim, 0,
                                                stream>>>(
        tiled_tensor_src, tiled_tensor_dst_transposed,
        shared_memory_layout_swizzled_src, shared_memory_layout_swizzled_src,
        thread_layout, thread_layout_transposed, vector_layout);

    return cudaGetLastError();
}

// Explicit instantiation.
template cudaError_t
launch_matrix_transpose_shared_memory_bank_conflict_read<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t
launch_matrix_transpose_shared_memory_bank_conflict_read<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

template cudaError_t
launch_matrix_transpose_shared_memory_vectorized_bank_conflict_read<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t
launch_matrix_transpose_shared_memory_vectorized_bank_conflict_read<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

template cudaError_t
launch_matrix_transpose_shared_memory_bank_conflict_write<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t
launch_matrix_transpose_shared_memory_bank_conflict_write<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

template cudaError_t launch_matrix_transpose_shared_memory_padded<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t launch_matrix_transpose_shared_memory_padded<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

template cudaError_t
launch_matrix_transpose_shared_memory_vectorized_padded<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t
launch_matrix_transpose_shared_memory_vectorized_padded<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

template cudaError_t launch_matrix_transpose_shared_memory_swizzled<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t launch_matrix_transpose_shared_memory_swizzled<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

template cudaError_t
launch_matrix_transpose_shared_memory_vectorized_swizzled<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t
launch_matrix_transpose_shared_memory_vectorized_swizzled<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

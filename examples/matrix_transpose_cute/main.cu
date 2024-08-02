// #include <cutlass/detail/layout.hpp>

// #include <cute/pointer.hpp>
#include <cute/tensor.hpp>

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const* file, int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class TensorSrc, class TensorDst, class ThreadLayoutSrc, class ThreadLayoutDst>
__global__ void transpose_kernel(
    TensorSrc tensor_src,
    TensorDst tensor_dst,
    ThreadLayoutSrc thread_layout_src,
    ThreadLayoutDst thread_layout_dst)
{
    static_assert(std::is_same_v<typename TensorSrc::value_type, typename TensorDst::value_type>);
    // Get block tile by indexing.
    // tensor_src[:, :, blockIdx.x, blockIdx.y]
    // (BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N)
    auto const tensor_block_tile_src{tensor_src(cute::make_coord(cute::_, cute::_), blockIdx.x, blockIdx.y)};
    // tensor_dst[:, :, blockIdx.x, blockIdx.y]
    // (BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_M)
    auto tensor_block_tile_dst{tensor_dst(cute::make_coord(cute::_, cute::_), blockIdx.x, blockIdx.y)};

    // auto const tensor_thread_tile_src{cute::local_partition(tensor_block_tile_src, thread_layout_src, threadIdx.x, threadIdx.y)};

    // auto tensor_thread_tile_dst{cute::local_partition(tensor_block_tile_dst, thread_layout_dst, threadIdx.x, threadIdx.y)};

    auto const tensor_thread_tile_src{cute::local_partition(tensor_block_tile_src, thread_layout_src, threadIdx.x)};

    auto tensor_thread_tile_dst{cute::local_partition(tensor_block_tile_dst, thread_layout_dst, threadIdx.x)};

    auto register_buffer{cute::make_tensor_like(tensor_thread_tile_src)};

    cute::copy(tensor_thread_tile_src, register_buffer);
    cute::copy(register_buffer, tensor_thread_tile_dst);
}



int main()
{
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // M fast dimension?
    size_t const M{1024};
    size_t const N{1024};

    size_t const matrix_size{M * N};
    std::vector<float> matrix(matrix_size, 0.0f);
    std::vector<float> matrix_transposed(matrix_size, 1.0f);
    std::vector<float> matrix_transposed_reference(matrix_size, 2.0f);

    float const* const h_src{matrix.data()};
    float* const h_dst{matrix_transposed.data()};

    for (size_t i{0}; i < N; ++i)
    {
        for (size_t j{0}; j < M; ++j)
        {
            matrix[i * M + j] = i * M + j;
        }
    }

    // Compute reference tensor.
    for (size_t i{0}; i < M; ++i)
    {
        for (size_t j{0}; j < N; ++j)
        {
            size_t src_idx{j * M + i};
            size_t dst_idx{i * N + j};
            matrix_transposed_reference[dst_idx] = matrix[src_idx];
        }
    }

    float* d_src;
    float* d_dst;

    CHECK_CUDA_ERROR(cudaMalloc(&d_src, matrix_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dst, matrix_size * sizeof(float)));

    // Copy data to device.
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_src, h_src, matrix_size * sizeof(float), cudaMemcpyHostToDevice, stream));


    // Dynamic shapes.
    auto const tensor_shape{cute::make_shape(M, N)};
    auto const tensor_shape_transposed{cute::make_shape(N, M)};

    auto const gmem_layout_row_major_src{cute::make_layout(tensor_shape, cute::GenRowMajor{})};
    // Same tensor content, different views.
    // This is the row-major layout of the tensor the user expects.
    auto const gmem_layout_row_major_dst{cute::make_layout(tensor_shape_transposed, cute::GenRowMajor{})};
    // This is an equivalent layout of the tensor, even though it is column-major.
    auto const gemm_layout_col_major_dst{cute::make_layout(tensor_shape, cute::GenColMajor{})};

    cute::gmem_ptr<float*> const gemm_ptr_src{cute::make_gmem_ptr(d_src)};
    cute::gmem_ptr<float*> const gemm_ptr_dst{cute::make_gmem_ptr(d_dst)};

    auto const tensor_src{cute::make_tensor(gemm_ptr_src, gmem_layout_row_major_src)};
    auto tensor_dst{cute::make_tensor(gemm_ptr_dst, gemm_layout_col_major_dst)};

    using BLOCK_TILE_SIZE_M = cute::Int<64>;
    using BLOCK_TILE_SIZE_N = cute::Int<64>;

    // Static shapes known at compile time.
    auto const block_tile_shape{cute::make_shape(BLOCK_TILE_SIZE_M{}, BLOCK_TILE_SIZE_N{})};
    auto const block_tile_shape_transposed{cute::make_shape(BLOCK_TILE_SIZE_N{}, BLOCK_TILE_SIZE_M{})};

    // Create a divided tensor.
    // The values in the divided tensor which has the same tile id are the same.
    auto const block_tiled_tensor_src{cute::tiled_divide(tensor_src, block_tile_shape)};
    auto block_tiled_tensor_dst{cute::tiled_divide(tensor_dst, block_tile_shape_transposed)};

    std::cout << cute::size<1>(block_tiled_tensor_src) << std::endl;
    std::cout << cute::size<2>(block_tiled_tensor_src) << std::endl;

    using NUM_THREADS_M = cute::Int<8>;
    using NUM_THREADS_N = cute::Int<32>;

    // Static shapes known at compile time.
    auto const thread_layout_shape{cute::make_shape(NUM_THREADS_M{}, NUM_THREADS_N{})};
    auto const thread_layout_row_major_src{cute::make_layout(thread_layout_shape, cute::GenRowMajor{})};
    auto const thread_layout_row_major_dst{cute::make_layout(thread_layout_shape, cute::GenRowMajor{})};

    dim3 const grid_size{cute::size<1>(block_tiled_tensor_src), cute::size<2>(block_tiled_tensor_src)};

    // dim3 const block_size{cute::size<0>(thread_layout_row_major_src), cute::size<1>(thread_layout_row_major_src)};

    dim3 const block_size{cute::size(thread_layout_row_major_src)};

    transpose_kernel<<<grid_size, block_size, 0, stream>>>(block_tiled_tensor_src, block_tiled_tensor_dst, thread_layout_row_major_src, thread_layout_row_major_dst);

    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_dst, d_dst, matrix_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // for (size_t i{0}; i < matrix_size; ++i)
    // {
    //     std::cout << "Idx " << i << ": " << matrix_transposed_reference[i] << " " << matrix_transposed[i] << std::endl;
    // }

    // Verify the result.
    for (size_t i{0}; i < matrix_size; ++i)
    {
        if (matrix_transposed_reference[i] != matrix_transposed[i])
        {
            std::cerr << "Mismatch at index " << i << " expected " << matrix_transposed_reference[i] << " got " << matrix_transposed[i] << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }


    // auto const tensor1{cute::local_partition(block_tiled_tensor_src(cute::make_coord(cute::_, cute::_), 0, 0), thread_layout_row_major_src, 0, 0)};
    // std::cout << cute::size<0>(tensor1) << std::endl;
    // std::cout << cute::size<1>(tensor1) << std::endl;
    // // std::cout << cute::size<2>(tensor1) << std::endl;


    CHECK_CUDA_ERROR(cudaFree(d_src));
    CHECK_CUDA_ERROR(cudaFree(d_dst));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

}
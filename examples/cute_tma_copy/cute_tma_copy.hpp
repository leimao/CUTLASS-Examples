#ifndef CUTE_TMA_COPY_HPP
#define CUTE_TMA_COPY_HPP

#include <cuda_runtime.h>

/**
 * @brief Launch matrix copy operation using TMA (Tensor Memory Accelerator)
 *
 * This function performs a matrix copy operation using NVIDIA's Tensor Memory
 * Accelerator (TMA) available on Hopper architecture and later. TMA provides
 * efficient data movement between global memory and shared memory.
 *
 * @tparam DataType The data type for matrix elements (float, double, etc.)
 *
 * @param input_matrix Pointer to input matrix in global memory
 * @param output_matrix Pointer to output matrix in global memory
 * @param m Number of rows in the matrix
 * @param n Number of columns in the matrix
 * @param stream CUDA stream for asynchronous execution
 * @return cudaError_t CUDA error status
 */
template <class DataType>
cudaError_t launch_matrix_copy(DataType const* input_matrix,
                               DataType* output_matrix, unsigned int m,
                               unsigned int n, cudaStream_t stream);

#endif // CUTE_TMA_COPY_HPP

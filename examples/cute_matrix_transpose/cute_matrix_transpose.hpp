#ifndef CUTE_MATRIX_TRANSPOSE_HPP
#define CUTE_MATRIX_TRANSPOSE_HPP

#include <cuda_runtime.h>

template <typename T>
cudaError_t
launch_transpose_naive_coalesced_read(T const* input_matrix, T* output_matrix,
                                      unsigned int M, unsigned int N,
                                      cudaStream_t stream);

template <typename T>
cudaError_t
launch_transpose_naive_coalesced_write(T const* input_matrix, T* output_matrix,
                                       unsigned int M, unsigned int N,
                                       cudaStream_t stream);

template <typename T>
cudaError_t launch_transpose_shared_memory_bank_conflict_read(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    cudaStream_t stream);

template <typename T>
cudaError_t launch_transpose_shared_memory_bank_conflict_read_vectorized(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    cudaStream_t stream);

template <typename T>
cudaError_t launch_transpose_shared_memory_bank_conflict_write(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    cudaStream_t stream);

template <typename T>
cudaError_t
launch_transpose_shared_memory_padded(T const* input_matrix, T* output_matrix,
                                      unsigned int M, unsigned int N,
                                      cudaStream_t stream);

template <typename T>
cudaError_t
launch_transpose_shared_memory_swizzled(T const* input_matrix, T* output_matrix,
                                        unsigned int M, unsigned int N,
                                        cudaStream_t stream);

#endif // CUTE_MATRIX_TRANSPOSE_HPP
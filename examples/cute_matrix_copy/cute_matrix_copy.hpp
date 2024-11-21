#ifndef CUTE_MATRIX_COPY_HPP
#define CUTE_MATRIX_COPY_HPP

#include <cuda_runtime.h>

template <typename T>
cudaError_t
launch_matrix_copy(T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N, cudaStream_t stream);

template <typename T>
cudaError_t
launch_matrix_copy_vectorized(T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N, cudaStream_t stream);

#endif // CUTE_MATRIX_COPY_HPP
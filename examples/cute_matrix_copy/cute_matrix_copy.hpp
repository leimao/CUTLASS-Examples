#ifndef CUTE_MATRIX_COPY_HPP
#define CUTE_MATRIX_COPY_HPP

#include <cuda_runtime.h>

template <typename T>
cudaError_t
launch_matrix_copy(T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N, cudaStream_t stream);


// Vectorized matrix copy assumes the leading dimension of the matrices is aligned to 128 bytes.
// For the matrices that cannot align to 128 bytes, they can be allocated with additional padding.
template <typename T>
cudaError_t
launch_matrix_copy_vectorized(T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N, cudaStream_t stream);

#endif // CUTE_MATRIX_COPY_HPP
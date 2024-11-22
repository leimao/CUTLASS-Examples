cmake_minimum_required(VERSION 3.28)

project(CUTE-TRANSPOSE-NAIVE VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_TRANSPOSE cute_matrix_transpose)
add_library(
    ${CUTE_TRANSPOSE}
    SHARED
    cute_matrix_transpose_naive.cu
    cute_matrix_transpose_shared_memory.cu
    cute_matrix_transpose_shared_memory_vectorized.cu
)
target_include_directories(${CUTE_TRANSPOSE} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_TRANSPOSE} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_TRANSPOSE} PRIVATE --expt-relaxed-constexpr)

add_subdirectory(tests)
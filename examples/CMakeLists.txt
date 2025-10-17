cmake_minimum_required(VERSION 3.28)

project(CUTLASS-Examples VERSION 0.0.1 LANGUAGES CXX CUDA)

# Disabling the subdirectories that are not interesting can reduce the build time.
add_subdirectory(cute_matrix_transpose)
add_subdirectory(cute_vector_copy)
add_subdirectory(cute_swizzle_preview)
add_subdirectory(cute_tiled_mma_preview)
add_subdirectory(cute_general_matrix_multiplication)
add_subdirectory(cute_tma_copy)

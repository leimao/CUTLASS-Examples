cmake_minimum_required(VERSION 3.28)

project(CUTLASS-Examples VERSION 0.0.1 LANGUAGES CXX CUDA)

add_subdirectory(gemm_api_v2)
add_subdirectory(gemm_api_v3)
add_subdirectory(cute_tiled_copy)
add_subdirectory(cute_matrix_transpose)
add_subdirectory(sgemm)

cmake_minimum_required(VERSION 3.28)

project(CUTLASS-Examples VERSION 0.0.1 LANGUAGES CXX CUDA)

# Disabling the subdirectories that are not interesting can reduce the build time.
add_subdirectory(cute_matrix_transpose)
add_subdirectory(cute_vector_copy)
add_subdirectory(cute_swizzle_preview)
add_subdirectory(cute_tiled_mma_preview)
add_subdirectory(cute_general_matrix_multiplication)

# cute_tma_copy requires SM90+ architecture support
# Check if CUDA version supports SM90 (requires CUDA 11.8+) and if SM90 is in the target architectures
if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
    # Check if CMAKE_CUDA_ARCHITECTURES contains SM90 variants (90, 90a)
    if (CMAKE_CUDA_ARCHITECTURES MATCHES "90")
        add_subdirectory(cute_tma_copy)
        message(STATUS "Building cute_tma_copy for SM90+ architectures")
    else()
        message(STATUS "Skipping cute_tma_copy: SM90+ not in CMAKE_CUDA_ARCHITECTURES (${CMAKE_CUDA_ARCHITECTURES})")
    endif()
else()
    message(STATUS "Skipping cute_tma_copy: CUDA ${CUDA_VERSION} does not support SM90 (requires 11.8+)")
endif()
cmake_minimum_required(VERSION 3.28)

project(CUTE-TMA VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
# Create a library for testing
set(CUTE_TMA_COPY cute_tma_copy)
add_library(
    ${CUTE_TMA_COPY}
    SHARED
    cute_tma_copy.cu
)
target_include_directories(${CUTE_TMA_COPY} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_TMA_COPY} PROPERTIES CUDA_ARCHITECTURES 90a)
target_compile_options(${CUTE_TMA_COPY} PRIVATE --expt-relaxed-constexpr)

add_subdirectory(tests)
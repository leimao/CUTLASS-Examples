cmake_minimum_required(VERSION 3.28)

project(CUTE-VECTOR-COPY VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_VECTOR_COPY cute_vector_copy)
add_library(
    ${CUTE_VECTOR_COPY}
    SHARED
    cute_vector_copy.cu
    cute_vector_copy_vectorized.cu
)
target_include_directories(${CUTE_VECTOR_COPY} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
# set_target_properties(${CUTE_VECTOR_COPY} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_VECTOR_COPY} PRIVATE --expt-relaxed-constexpr)

add_subdirectory(tests)
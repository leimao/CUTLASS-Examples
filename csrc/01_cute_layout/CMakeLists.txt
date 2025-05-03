cmake_minimum_required(VERSION 3.28)

project(CUTE_LAYOUT VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_LAYOUT cute_layout)
add_executable(
    ${CUTE_LAYOUT}
    cute_layout.cu
)
target_include_directories(${CUTE_LAYOUT} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_LAYOUT} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_LAYOUT} PRIVATE --expt-relaxed-constexpr)

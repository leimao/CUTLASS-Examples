cmake_minimum_required(VERSION 3.28)

project(CUTE-TMA VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_TMA cute_tma)
add_executable(
    ${CUTE_TMA}
    cute_tma_copy.cu
)
target_include_directories(${CUTE_TMA} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_TMA} PROPERTIES CUDA_ARCHITECTURES 90a)
target_compile_options(${CUTE_TMA} PRIVATE --expt-relaxed-constexpr)

# add_subdirectory(tests)
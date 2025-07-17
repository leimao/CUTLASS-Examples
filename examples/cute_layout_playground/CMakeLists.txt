cmake_minimum_required(VERSION 3.28)

project(CUTE-LAYOUT-PLAYGROUND VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_LAYOUT_PLAYGROUND cute_layout_playground)
add_executable(
    ${CUTE_LAYOUT_PLAYGROUND}
    cute_layout_playground.cu
)
target_include_directories(${CUTE_LAYOUT_PLAYGROUND} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_LAYOUT_PLAYGROUND} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_LAYOUT_PLAYGROUND} PRIVATE --expt-relaxed-constexpr)

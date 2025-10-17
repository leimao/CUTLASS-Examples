cmake_minimum_required(VERSION 3.28)

project(CUTE-SWIZZLE-PREVIEW VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_SWIZZLE_PREVIEW cute_swizzle_preview)
add_executable(
    ${CUTE_SWIZZLE_PREVIEW}
    cute_swizzle_preview.cu
)
target_include_directories(${CUTE_SWIZZLE_PREVIEW} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
# set_target_properties(${CUTE_SWIZZLE_PREVIEW} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_SWIZZLE_PREVIEW} PRIVATE --expt-relaxed-constexpr)

cmake_minimum_required(VERSION 3.28)

project(CUTE-TILED-MMA-PREVIEW VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_TILED_MMA_PREVIEW cute_tiled_mma_preview)
add_executable(
    ${CUTE_TILED_MMA_PREVIEW}
    cute_tiled_mma_preview.cu
)
target_include_directories(${CUTE_TILED_MMA_PREVIEW} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
# set_target_properties(${CUTE_TILED_MMA_PREVIEW} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_TILED_MMA_PREVIEW} PRIVATE --expt-relaxed-constexpr)

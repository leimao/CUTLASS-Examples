cmake_minimum_required(VERSION 3.28)

project(CUTE-MMA-PROFILER VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_MMA_PROFILER_SM80 cute_mma_profiler_sm80)
add_executable(
    ${CUTE_MMA_PROFILER_SM80}
    cute_mma_profiler_sm80.cu
)
target_include_directories(${CUTE_MMA_PROFILER_SM80} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_MMA_PROFILER_SM80} PROPERTIES CUDA_ARCHITECTURES "80;86;89")
target_compile_options(${CUTE_MMA_PROFILER_SM80} PRIVATE --expt-relaxed-constexpr)

set(CUTE_MMA_PROFILER_SM89 cute_mma_profiler_sm89)
add_executable(
    ${CUTE_MMA_PROFILER_SM89}
    cute_mma_profiler_sm89.cu
)
target_include_directories(${CUTE_MMA_PROFILER_SM89} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_MMA_PROFILER_SM89} PROPERTIES CUDA_ARCHITECTURES "89;90")
target_compile_options(${CUTE_MMA_PROFILER_SM89} PRIVATE --expt-relaxed-constexpr)

set(CUTE_MMA_PROFILER_SM120 cute_mma_profiler_sm120)
add_executable(
    ${CUTE_MMA_PROFILER_SM120}
    cute_mma_profiler_sm120.cu
)
target_include_directories(${CUTE_MMA_PROFILER_SM120} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_MMA_PROFILER_SM120} PROPERTIES CUDA_ARCHITECTURES "120a")
target_compile_options(${CUTE_MMA_PROFILER_SM120} PRIVATE --expt-relaxed-constexpr)

set(CUTE_MMA_PROFILER_SM120_SPARSE cute_mma_profiler_sm120_sparse)
add_executable(
    ${CUTE_MMA_PROFILER_SM120_SPARSE}
    cute_mma_profiler_sm120_sparse.cu
)
target_include_directories(${CUTE_MMA_PROFILER_SM120_SPARSE} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_MMA_PROFILER_SM120_SPARSE} PROPERTIES CUDA_ARCHITECTURES "120a")
target_compile_options(${CUTE_MMA_PROFILER_SM120_SPARSE} PRIVATE --expt-relaxed-constexpr)
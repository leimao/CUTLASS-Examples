cmake_minimum_required(VERSION 3.28)

project(CUTE-MMA-BENCHMARK VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_MMA_BENCHMARK_SM70 cute_mma_benchmark_sm70)
add_executable(
    ${CUTE_MMA_BENCHMARK_SM70}
    cute_mma_benchmark_sm70.cu
)
target_include_directories(${CUTE_MMA_BENCHMARK_SM70} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_MMA_BENCHMARK_SM70} PROPERTIES CUDA_ARCHITECTURES "75")
target_compile_options(${CUTE_MMA_BENCHMARK_SM70} PRIVATE --expt-relaxed-constexpr)

set(CUTE_MMA_BENCHMARK_SM75 cute_mma_benchmark_sm75)
add_executable(
    ${CUTE_MMA_BENCHMARK_SM75}
    cute_mma_benchmark_sm75.cu
)
target_include_directories(${CUTE_MMA_BENCHMARK_SM75} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_MMA_BENCHMARK_SM75} PROPERTIES CUDA_ARCHITECTURES "75")
target_compile_options(${CUTE_MMA_BENCHMARK_SM75} PRIVATE --expt-relaxed-constexpr)

set(CUTE_MMA_BENCHMARK_SM80 cute_mma_benchmark_sm80)
add_executable(
    ${CUTE_MMA_BENCHMARK_SM80}
    cute_mma_benchmark_sm80.cu
)
target_include_directories(${CUTE_MMA_BENCHMARK_SM80} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_MMA_BENCHMARK_SM80} PROPERTIES CUDA_ARCHITECTURES "80;86;89")
target_compile_options(${CUTE_MMA_BENCHMARK_SM80} PRIVATE --expt-relaxed-constexpr)

set(CUTE_MMA_BENCHMARK_SM89 cute_mma_benchmark_sm89)
add_executable(
    ${CUTE_MMA_BENCHMARK_SM89}
    cute_mma_benchmark_sm89.cu
)
target_include_directories(${CUTE_MMA_BENCHMARK_SM89} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_MMA_BENCHMARK_SM89} PROPERTIES CUDA_ARCHITECTURES "89;90")
target_compile_options(${CUTE_MMA_BENCHMARK_SM89} PRIVATE --expt-relaxed-constexpr)

set(CUTE_MMA_BENCHMARK_SM120 cute_mma_benchmark_sm120)
add_executable(
    ${CUTE_MMA_BENCHMARK_SM120}
    cute_mma_benchmark_sm120.cu
)
target_include_directories(${CUTE_MMA_BENCHMARK_SM120} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_MMA_BENCHMARK_SM120} PROPERTIES CUDA_ARCHITECTURES "120a")
target_compile_options(${CUTE_MMA_BENCHMARK_SM120} PRIVATE --expt-relaxed-constexpr)

set(CUTE_MMA_BENCHMARK_SM120_SPARSE cute_mma_benchmark_sm120_sparse)
add_executable(
    ${CUTE_MMA_BENCHMARK_SM120_SPARSE}
    cute_mma_benchmark_sm120_sparse.cu
)
target_include_directories(${CUTE_MMA_BENCHMARK_SM120_SPARSE} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_MMA_BENCHMARK_SM120_SPARSE} PROPERTIES CUDA_ARCHITECTURES "120a")
target_compile_options(${CUTE_MMA_BENCHMARK_SM120_SPARSE} PRIVATE --expt-relaxed-constexpr)
cmake_minimum_required(VERSION 3.28)

project(SGEMM VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
add_executable(sgemm_1 sgemm_1.cu)
target_include_directories(sgemm_1 PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(sgemm_1 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(sgemm_1 PRIVATE --expt-relaxed-constexpr)

add_executable(sgemm_2 sgemm_2.cu)
target_include_directories(sgemm_2 PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(sgemm_2 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(sgemm_2 PRIVATE --expt-relaxed-constexpr)

add_executable(sgemm_sm70 sgemm_sm70.cu)
target_include_directories(sgemm_sm70 PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(sgemm_sm70 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(sgemm_sm70 PRIVATE --expt-relaxed-constexpr)

add_executable(sgemm_sm80 sgemm_sm80.cu)
target_include_directories(sgemm_sm80 PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(sgemm_sm80 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(sgemm_sm80 PRIVATE --expt-relaxed-constexpr)

add_executable(tiled_copy tiled_copy.cu)
target_include_directories(tiled_copy PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(tiled_copy PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(tiled_copy PRIVATE --expt-relaxed-constexpr)
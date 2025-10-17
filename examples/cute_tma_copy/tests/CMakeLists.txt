cmake_minimum_required(VERSION 3.28)

project(CUTE-TMA-COPY-TEST VERSION 0.0.1 LANGUAGES CXX CUDA)

find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(CUTE_TMA_COPY_INCLUDE cute_tma_copy.hpp HINTS ${CMAKE_SOURCE_DIR}/examples/cute_tma_copy/)

add_executable(test_tma_copy test_tma_copy.cu)
target_include_directories(test_tma_copy PRIVATE ${CUTE_TMA_COPY_INCLUDE} ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
target_link_libraries(test_tma_copy GTest::gtest GTest::gtest_main ${CUTE_TMA_COPY})
set_target_properties(test_tma_copy PROPERTIES CUDA_ARCHITECTURES native)

add_test(TestAllTmaCopy test_tma_copy)

add_executable(profile_tma_copy profile_tma_copy.cu)
target_include_directories(profile_tma_copy PRIVATE ${CUTE_TMA_COPY_INCLUDE} ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
target_link_libraries(profile_tma_copy GTest::gtest GTest::gtest_main ${CUTE_TMA_COPY})
set_target_properties(profile_tma_copy PROPERTIES CUDA_ARCHITECTURES native)

add_test(ProfileAllTmaCopy profile_tma_copy)
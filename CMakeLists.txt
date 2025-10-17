cmake_minimum_required(VERSION 3.28)

# This has to be placed before the project() call
# Set CUDA architectures before enabling CUDA language
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 90a)
endif()

project(CUTLASS-Examples VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find GoogleTest
find_package(GTest REQUIRED)

# Enable testing at the root level so that we could run all the tests at once
enable_testing()

# Set CUTLASS include directories
find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS cutlass/include)
find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS cutlass/tools/util/include)

add_subdirectory(examples)

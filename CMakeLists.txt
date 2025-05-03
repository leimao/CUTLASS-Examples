cmake_minimum_required(VERSION 3.28)

project(CUTLASS-Examples VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip 
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing at the root level so that we could run all the tests at once
enable_testing()

# Set CUTLASS include directories
find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS cutlass/include)
find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS cutlass/tools/util/include)

# add_subdirectory(examples)
add_subdirectory(csrc)

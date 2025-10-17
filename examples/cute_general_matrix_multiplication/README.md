# CuTe General Matrix Multiplication

## Introduction

These examples demonstrate the implementation of general matrix transpose kernels using CuTe. They follow the cuBLAS interface design and supports matrices that are transposed or not transposed. The naive kernels perform boundary checks and can be used for any matrix dimension size. The kernels that performs tiled copy assumes each matrix dimension size is a multiple of certain size, usually 32, depending on the data type.

The performance was mainly optimized for the NT case where the matrix A is a M x K column-major matrix, the matrix B is a K x N row-major matrix, and the matrix C is a M x N column-major matrix. I will try coming up with a general implementation that optimizes for all the NN, NT, TN, and TT cases in the future. An idea is to use shared memory layouts that are compatible with the global memory layouts so that global memory vectorized access can always be enabled.

## Usages

### Build Specialized CUDA Kernels

High performance CUDA kernels are usually implemented in a way that is specialized for a certain problem sizes. Generic CUDA kernels might not be able to achieve the best performance. So instead of using one generic CUDA kernel to solve all the problems, accelerated computing libraries usually provide a set of specialized CUDA kernels that are optimized for different problem sizes or have some strict assumptions and requirements on the problem sizes.

To build specialized CUDA kernels for performance measurements, please run the following commands.

```bash
$ export NUM_CMAKE_JOBS=4
$ cmake -B build -DNO_BOUNDS_CHECK=ON
$ cmake --build build --config Release --parallel ${NUM_CMAKE_JOBS}
```

To build generic CUDA kernels which are suitable for a wide range of problem sizes, please run the following commands.

```bash
$ export NUM_CMAKE_JOBS=4
$ cmake -B build -DNO_BOUNDS_CHECK=OFF
$ cmake --build build --config Release --parallel ${NUM_CMAKE_JOBS}
```

### Run Unit Tests

```bash
$ ctest --test-dir build/ --tests-regex "TestAllGeneralMatrixMultiplication.*" --verbose
The following tests passed:
        TestAllGeneralMatrixMultiplicationNaive
        TestAllGeneralMatrixMultiplicationNaiveTiledCopyTiledMma
        TestAllGeneralMatrixMultiplicationTensorCoreTiledCopyTiledMma

100% tests passed, 0 tests failed out of 3
```

### Run Performance Measurement

The performance were measured using specialized CUDA kernels.

```bash
$ ctest --test-dir build/ --tests-regex "ProfileAllGeneralMatrixMultiplication.*" --verbose
```

Only the general matrix multiplication kernels that consume half typed data and compute in half precision are documented here for a problem size of 4096 x 4096 x 4096 on NVIDIA GeForce RTX 3090.

|                                   Kernel Name                                    | Trans A | Trans B | Latency (ms) |  TOPs   | Performance VS cuBLAS (%) |
| :------------------------------------------------------------------------------: | :-----: | :-----: | :----------: | :-----: | :-----------------------: |
|                                      cuBLAS                                      |    N    |    T    |   1.12927    | 121.706 |             -             |
|                                      Naive                                       |    N    |    T    |   15.1479    | 9.07312 |          6.96212          |
|                           Gmem Tiled Copy + Tiled MMA                            |    N    |    T    |   8.02777    | 17.1204 |          14.067           |
|                  Gmem Tiled Copy + Tiled MMA + SM80 Tensor Core                  |    N    |    T    |   2.29437    | 59.9026 |          56.0556          |
|          Gmem Tiled Copy + Tiled MMA + SM80 Tensor Core + SM80 Pipeline          |    N    |    T    |   1.63768    | 83.9228 |          77.8466          |
|         Gmem Tiled Copy + Smem Tiled Copy + Tiled MMA + SM80 Tensor Core         |    N    |    T    |   1.93464    | 71.041  |          65.1556          |
| Gmem Tiled Copy + Smem Tiled Copy + Tiled MMA + SM80 Tensor Core + SM80 Pipeline |    N    |    T    |   1.16808    | 117.663 |          108.67           |

### Run Nsight Compute Profiling

```bash
for file in build/examples/cute_general_matrix_multiplication/tests/profile_*; do
    filename=$(basename -- "$file")
    ncu --set full -f -o ncu_reports/"$filename" "$file"
done
```

### Run Compute Sanitizer

```bash
for file in build/examples/cute_general_matrix_multiplication/tests/test_*; do
    compute-sanitizer --leak-check full "$file"
done
```

## References

- [Cooperative GEMM - CuTe](https://github.com/NVIDIA/cutlass/blob/bf9da7b76c766d7ee7d536afc77880a4ef1f1156/include/cute/algorithm/cooperative_gemm.hpp)
- [SM80 MMA Multi-Stage - CUTLASS](https://github.com/NVIDIA/cutlass/blob/bf9da7b76c766d7ee7d536afc77880a4ef1f1156/include/cutlass/gemm/collective/sm80_mma_multistage.hpp)

# CuTe Matrix Transpose

## Introduction

These examples demonstrate the implementation of general matrix transpose kernels using CuTe. Each kernel does boundary checks and can be used for any matrix size. The matrices are assumed to be stored in row-major order. The general matrix transpose vectorized kernel also assumes the row size is a multiple of certain size depending on the data type.

## Usages

### Run Unit Tests

```bash
$ ctest --test-dir build/ --tests-regex "TestAllMatrixTranspose.*" --verbose

The following tests passed:
        TestAllMatrixTransposeNaiveCoalescedRead
        TestAllMatrixTransposeNaiveCoalescedWrite
        TestAllMatrixTransposeSharedMemoryBankConflictRead
        TestAllMatrixTransposeSharedMemoryBankConflictWrite
        TestAllMatrixTransposeSharedMemoryPadded
        TestAllMatrixTransposeSharedMemorySwizzled

100% tests passed, 0 tests failed out of 6
```

### Run Performance Measurement

```bash
$ ctest --test-dir build/ --tests-regex "ProfileAllMatrixTranspose.*" --verbose
```

The following tables show the performance measurements of a 16384 x 16384 matrix transpose using the matrix transpose kernels on NVIDIA GeForce RTX 3090.

|                 Kernel Name                 | Latency (ms) | Effective Bandwidth (GB/s) | Peak Bandwidth Percentage (%) |
| :-----------------------------------------: | :----------: | :------------------------: | :---------------------------: |
|            Naive Coalesced Read             |   7.43096    |          269.144           |            28.7518            |
|            Naive Coalesced Write            |   3.12054    |          640.915           |            68.4668            |
|      Shared Memory Bank Conflict Read       |    3.1184    |          641.355           |            68.5138            |
| Shared Memory Vectorized Bank Conflict Read |   3.23225    |          618.764           |            66.1005            |
|      Shared Memory Bank Conflict Write      |   3.18751    |           627.45           |            67.0283            |
|            Shared Memory Padded             |   3.12975    |          639.028           |            68.2652            |
|       Shared Memory Vectorized Padded       |   3.19688    |          625.611           |            66.8319            |
|           Shared Memory Swizzled            |    2.9489    |           678.22           |            72.452             |
|      Shared Memory Vectorized Swizzled      |   3.23204    |          618.804           |            66.1047            |

### Run Nsight Compute Profiling

```bash
for file in build/examples/cute_matrix_transpose/tests/profile_*; do
    filename=$(basename -- "$file")
    ncu --set full -f -o ncu_reports/"$filename" "$file"
done
```

### Run Compute Sanitizer

```bash
for file in build/examples/cute_matrix_transpose/tests/test_*; do
    compute-sanitizer --leak-check full "$file"
done
```

## References

- [CuTe Matrix Transpose](https://leimao.github.io/article/CuTe-Matrix-Transpose/)

# CuTe Vector Copy

## Introduction

These examples demonstrate the implementation of vector copy kernels using CuTe. The general vector copy kernel does boundary checks and can be used for any matrix size. The vector copy vectorized kernel assumes the vector size is a multiple of certain size depending on the data type.

## Usages

### Run Unit Tests

```bash
$ ctest --test-dir build/ --tests-regex "TestAllVectorCopy.*" --verbose

The following tests passed:
        TestAllVectorCopy
        TestAllVectorCopyVectorized

100% tests passed, 0 tests failed out of 2
```

### Run Performance Measurement

```bash
$ ctest --test-dir build/ --tests-regex "ProfileAllVectorCopy.*" --verbose
```

The following tables show the performance measurements of copying 1 GB floating point values using the vector copy kernels on NVIDIA GeForce RTX 3090.

|      Kernel Name       | Latency (ms) | Effective Bandwidth (GB/s) | Peak Bandwidth Percentage (%) |
| :--------------------: | :----------: | :------------------------: | :---------------------------: |
|      Vector Copy       |   2.81923    |          709.413           |            75.7842            |
| Vector Copy Vectorized |   2.83133    |          706.382           |            75.4604            |

### Run Nsight Compute Profiling

```bash
for file in build/examples/cute_vector_copy/tests/profile_*; do
    filename=$(basename -- "$file")
    ncu --set full -f -o ncu_reports/"$filename" "$file"
done
```

### Run Compute Sanitizer

```bash
for file in build/examples/cute_vector_copy/tests/test_*; do
    compute-sanitizer --leak-check full "$file"
done
```

## References

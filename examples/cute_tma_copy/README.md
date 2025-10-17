# CuTe TMA Copy

## Introduction

These examples demonstrate the implementation of matrix copy kernels using CuTe and TMA. The general matrix copy kernel does boundary checks and can be used for any matrix size. The matrix copy vectorized kernel assumes the matrix size is a multiple of certain size depending on the data type.

## Usages

### Run Unit Tests

```bash
$ ctest --test-dir build/ --tests-regex "TestAllTmaCopy.*" --verbose

The following tests passed:
        TestAllTmaCopy
        TestAllTmaCopyVectorized

100% tests passed, 0 tests failed out of 2
```

### Run Performance Measurement

```bash
$ ctest --test-dir build/ --tests-regex "ProfileAllTmaCopy.*" --verbose
```

### Run Nsight Compute Profiling

```bash
for file in build/examples/cute_tma_copy/tests/profile_*; do
    filename=$(basename -- "$file")
    ncu --set full -f -o ncu_reports/"$filename" "$file"
done
```

### Run Compute Sanitizer

```bash
for file in build/examples/cute_tma_copy/tests/test_*; do
    compute-sanitizer --leak-check full "$file"
done
```

## References

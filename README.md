# CUTLASS Examples

## Introduction

The CUDA kernel examples using [CUTLASS](https://github.com/NVIDIA/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md) abstractions.

## Examples

### CuTe Matrix Transpose

- CuTe Matrix Transpose Global Memory Access Coalesced Read
- CuTe Matrix Transpose Global Memory Access Coalesced Write
- CuTe Matrix Transpose Global Shared Memory Bank Read Conflicts
- CuTe Matrix Transpose Global Shared Memory Bank Write Conflicts
- CuTe Matrix Transpose Global Shared Memory Bank Conflicts Free with Padding

## Usages

To download the CUTLASS-Examples repository, please run the following command.

```bash
$ git clone --recursive https://github.com/leimao/CUTLASS-Examples
$ cd CUTLASS-Examples
# If you are updating the submodules of an existing checkout.
$ git submodule sync
$ git submodule update --init --recursive
```

## CUTLASS Docker Container

Docker is used to build and run CUTLASS CUDA kernels. The custom Docker container is built based on the [NVIDIA NGC CUDA](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) 12.4.1 Docker container.

Please adjust the base Docker container CUDA version if the host computer has a different CUDA version. Otherwise, weird compilation errors and runtime errors may occur.

### Build Docker Images

To build the custom Docker image, please run the following command.

```bash
$ docker build -f docker/cuda.Dockerfile --no-cache --tag cuda:12.4.1 .
```

### Run Docker Container

To run the custom Docker container, please run the following command.

```bash
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt -w /mnt cuda:12.4.1
```

To run the custom Docker container with NVIDIA Nsight Compute, please run the following command.

```bash
$ xhost +
$ docker run -it --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --cap-add=SYS_ADMIN --security-opt seccomp=unconfined -v $(pwd):/mnt -w /mnt --network=host cuda:12.4.1
$ xhost +
```

## CUTLASS CMake Examples

### Build Examples

To build the CUDA kernels, please run the following commands.

```bash
$ cmake -B build
$ cmake --build build --config Release --parallel
```

### Run Examples

To run the FP32 and FP16 GEMM CUDA kernels, please run the following commands.

```bash
$ ./build/examples/gemm_api_v2/CUTLASS-GEMM-API-V2
$ ./build/examples/gemm_api_v3/CUTLASS-GEMM-API-V3
$ ./build/examples/cute_tiled_copy/CUTE-TILED-COPY
```

### Run Unit Tests

To run the unit tests, please run the following command.

```bash
$ ctest --test-dir build/ --tests-regex "Test.*" --verbose
```

### Run Performance Measurements

To run the performance measurements, please run the following command.

```bash
$ ctest --test-dir build/ --tests-regex "Profile.*" --verbose
```

Performance measurements will run selected CUDA kernels for large problems multiple times and therefore might take a long time to complete.

## References

- [CuTe Layout Algebra](https://leimao.github.io/article/CuTe-Layout-Algebra/)
- [Build and Develop CUTLASS CUDA Kernels](https://leimao.github.io/blog/Build-Develop-CUTLASS-CUDA-Kernels/)

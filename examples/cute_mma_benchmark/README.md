# CuTe MMA Benchmark

## Introduction

This application benchmarks various CuTe MMA (Matrix Multiply-Accumulate) configurations on NVIDIA GPUs. It measures the AI peak performance in TOPS (Tera Operations Per Second) for different data types and layouts using NVIDIA Tensor Cores. Hopefully, this could help developers to reproduce the AI peak performance NVIDIA advertises for GPUs.

## Usages

The dense MMA performance of SM120 can be benchmarked using the following command.

```bash
$ ./build/examples/cute_mma_benchmark/cute_mma_benchmark_sm120
========================================
CUTLASS SM120 MMA Atom Benchmark
========================================
Device: NVIDIA GeForce RTX 5080
Compute Capability: 12.0
Multiprocessors: 84
Max Threads per SM: 1536
Max Blocks per SM: 24
========================================

Benchmarking SM120 MMA Atoms (FP8 Instructions):
Configuration: 84 SMs × 8 blocks/SM × 4 warps/block = 2688 total warps
----------------------------------------

=== FP32 Output (E2M1 Input) ===
(SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>) :   0.092480 ms |    238.107 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m1_t, float_e3m2_t, float>) :   0.092675 ms |    237.605 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m1_t, float_e2m3_t, float>) :   0.092320 ms |    238.519 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m1_t, float_e4m3_t, float>) :   0.092314 ms |    238.536 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m1_t, float_e5m2_t, float>) :   0.092394 ms |    238.329 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (E3M2 Input) ===
(SM120_16x8x32_TN<float_e3m2_t, float_e2m1_t, float>) :   0.092390 ms |    238.337 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e3m2_t, float_e3m2_t, float>) :   0.092486 ms |    238.090 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e3m2_t, float_e2m3_t, float>) :   0.092422 ms |    238.255 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e3m2_t, float_e4m3_t, float>) :   0.092861 ms |    237.130 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e3m2_t, float_e5m2_t, float>) :   0.092733 ms |    237.457 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (E2M3 Input) ===
(SM120_16x8x32_TN<float_e2m3_t, float_e2m1_t, float>) :   0.092442 ms |    238.205 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m3_t, float_e3m2_t, float>) :   0.092646 ms |    237.679 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m3_t, float_e2m3_t, float>) :   0.092288 ms |    238.602 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m3_t, float_e4m3_t, float>) :   0.092352 ms |    238.437 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m3_t, float_e5m2_t, float>) :   0.092205 ms |    238.817 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (E4M3 Input) ===
(SM120_16x8x32_TN<float_e4m3_t, float_e2m1_t, float>) :   0.092182 ms |    238.875 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e4m3_t, float_e3m2_t, float>) :   0.092490 ms |    238.082 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e4m3_t, float_e2m3_t, float>) :   0.092339 ms |    238.470 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>) :   0.092717 ms |    237.498 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e4m3_t, float_e5m2_t, float>) :   0.092323 ms |    238.511 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (E5M2 Input) ===
(SM120_16x8x32_TN<float_e5m2_t, float_e2m1_t, float>) :   0.092243 ms |    238.718 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e5m2_t, float_e3m2_t, float>) :   0.092694 ms |    237.556 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e5m2_t, float_e2m3_t, float>) :   0.092637 ms |    237.704 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e5m2_t, float_e4m3_t, float>) :   0.092224 ms |    238.768 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e5m2_t, float_e5m2_t, float>) :   0.092131 ms |    239.008 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled E2M1, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.092250 ms |    238.701 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e3m2_t, float, float_ue8m0_t, 32>) :   0.093789 ms |    234.784 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m3_t, float, float_ue8m0_t, 32>) :   0.092294 ms |    238.585 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e4m3_t, float, float_ue8m0_t, 32>) :   0.047123 ms |    467.288 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e5m2_t, float, float_ue8m0_t, 32>) :   0.047133 ms |    467.193 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled E3M2, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.092253 ms |    238.693 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e3m2_t, float, float_ue8m0_t, 32>) :   0.092234 ms |    238.743 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e2m3_t, float, float_ue8m0_t, 32>) :   0.092131 ms |    239.008 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e4m3_t, float, float_ue8m0_t, 32>) :   0.047750 ms |    461.150 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e5m2_t, float, float_ue8m0_t, 32>) :   0.047114 ms |    467.383 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled E2M3, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.092198 ms |    238.834 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e3m2_t, float, float_ue8m0_t, 32>) :   0.092195 ms |    238.842 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e2m3_t, float, float_ue8m0_t, 32>) :   0.092838 ms |    237.187 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e4m3_t, float, float_ue8m0_t, 32>) :   0.047149 ms |    467.034 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e5m2_t, float, float_ue8m0_t, 32>) :   0.047069 ms |    467.828 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled E4M3, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.047126 ms |    467.256 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e3m2_t, float, float_ue8m0_t, 32>) :   0.047194 ms |    466.591 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e2m3_t, float, float_ue8m0_t, 32>) :   0.046989 ms |    468.624 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, 32>) :   0.047162 ms |    466.907 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e5m2_t, float, float_ue8m0_t, 32>) :   0.047130 ms |    467.224 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled E5M2, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.047053 ms |    467.987 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e3m2_t, float, float_ue8m0_t, 32>) :   0.047165 ms |    466.876 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e2m3_t, float, float_ue8m0_t, 32>) :   0.047066 ms |    467.860 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e4m3_t, float, float_ue8m0_t, 32>) :   0.047178 ms |    466.749 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e5m2_t, float, float_ue8m0_t, 32>) :   0.047094 ms |    467.574 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled 16x8x64 E2M1, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.047155 ms |    933.941 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled 16x8x64 E2M1, VS=32, UE4M3) ===
(SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, 32>) :   0.047152 ms |    934.005 TOPS (672 blocks × 4 warps/block = 2688 warps)

========================================
Benchmarking Complete
========================================
```

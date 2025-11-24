# CuTe MMA Profiler

## Introduction

This application profiles various CuTe MMA (Matrix Multiply-Accumulate) configurations on NVIDIA GPUs. It measures the AI peak performance in TOPS (Tera Operations Per Second) for different data types and layouts using NVIDIA Tensor Cores. Hopefully, this could help developers to reproduce the AI peak performance NVIDIA advertises for GPUs.

## Usages

The dense MMA performance of SM120 can be profiled using the following command.

```bash
./build/examples/cute_mma_profiler/cute_mma_profiler_sm120
========================================
CUTLASS SM120 MMA Atom Profiler
========================================
Device: NVIDIA GeForce RTX 5080
Compute Capability: 12.0
Multiprocessors: 84
Max Threads per SM: 1536
Max Blocks per SM: 24
========================================

Profiling SM120 MMA Atoms (FP8 Instructions):
Configuration: 84 SMs × 8 blocks/SM × 4 warps/block = 2688 total warps
----------------------------------------

=== FP32 Output (E2M1 Input) ===
(SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>) :   0.092467 ms |    238.140 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m1_t, float_e3m2_t, float>) :   0.092592 ms |    237.819 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m1_t, float_e2m3_t, float>) :   0.092598 ms |    237.802 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m1_t, float_e4m3_t, float>) :   0.092278 ms |    238.627 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m1_t, float_e5m2_t, float>) :   0.092272 ms |    238.643 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (E3M2 Input) ===
(SM120_16x8x32_TN<float_e3m2_t, float_e2m1_t, float>) :   0.092253 ms |    238.693 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e3m2_t, float_e3m2_t, float>) :   0.092234 ms |    238.743 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e3m2_t, float_e2m3_t, float>) :   0.092435 ms |    238.222 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e3m2_t, float_e4m3_t, float>) :   0.092266 ms |    238.660 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e3m2_t, float_e5m2_t, float>) :   0.092102 ms |    239.083 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (E2M3 Input) ===
(SM120_16x8x32_TN<float_e2m3_t, float_e2m1_t, float>) :   0.092157 ms |    238.942 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m3_t, float_e3m2_t, float>) :   0.092150 ms |    238.958 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m3_t, float_e2m3_t, float>) :   0.092186 ms |    238.867 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m3_t, float_e4m3_t, float>) :   0.092099 ms |    239.091 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e2m3_t, float_e5m2_t, float>) :   0.092102 ms |    239.083 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (E4M3 Input) ===
(SM120_16x8x32_TN<float_e4m3_t, float_e2m1_t, float>) :   0.092288 ms |    238.602 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e4m3_t, float_e3m2_t, float>) :   0.092141 ms |    238.983 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e4m3_t, float_e2m3_t, float>) :   0.092330 ms |    238.494 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>) :   0.092128 ms |    239.016 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e4m3_t, float_e5m2_t, float>) :   0.092230 ms |    238.751 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (E5M2 Input) ===
(SM120_16x8x32_TN<float_e5m2_t, float_e2m1_t, float>) :   0.092182 ms |    238.875 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e5m2_t, float_e3m2_t, float>) :   0.092253 ms |    238.693 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e5m2_t, float_e2m3_t, float>) :   0.092122 ms |    239.033 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e5m2_t, float_e4m3_t, float>) :   0.092176 ms |    238.892 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120_16x8x32_TN<float_e5m2_t, float_e5m2_t, float>) :   0.092138 ms |    238.991 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled E2M1, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.092246 ms |    238.710 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e3m2_t, float, float_ue8m0_t, 32>) :   0.092250 ms |    238.701 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m3_t, float, float_ue8m0_t, 32>) :   0.092240 ms |    238.726 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e4m3_t, float, float_ue8m0_t, 32>) :   0.047171 ms |    466.812 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e5m2_t, float, float_ue8m0_t, 32>) :   0.047606 ms |    462.545 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled E3M2, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.092250 ms |    238.701 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e3m2_t, float, float_ue8m0_t, 32>) :   0.092141 ms |    238.983 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e2m3_t, float, float_ue8m0_t, 32>) :   0.092128 ms |    239.016 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e4m3_t, float, float_ue8m0_t, 32>) :   0.047066 ms |    467.860 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e5m2_t, float, float_ue8m0_t, 32>) :   0.047082 ms |    467.701 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled E2M3, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.092234 ms |    238.743 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e3m2_t, float, float_ue8m0_t, 32>) :   0.092134 ms |    239.000 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e2m3_t, float, float_ue8m0_t, 32>) :   0.092483 ms |    238.098 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e4m3_t, float, float_ue8m0_t, 32>) :   0.047142 ms |    467.097 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e5m2_t, float, float_ue8m0_t, 32>) :   0.047072 ms |    467.796 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled E4M3, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.047123 ms |    467.288 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e3m2_t, float, float_ue8m0_t, 32>) :   0.047091 ms |    467.605 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e2m3_t, float, float_ue8m0_t, 32>) :   0.047091 ms |    467.605 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, 32>) :   0.047146 ms |    467.066 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e5m2_t, float, float_ue8m0_t, 32>) :   0.047123 ms |    467.288 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled E5M2, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.047194 ms |    466.591 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e3m2_t, float, float_ue8m0_t, 32>) :   0.047139 ms |    467.129 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e2m3_t, float, float_ue8m0_t, 32>) :   0.047072 ms |    467.796 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e4m3_t, float, float_ue8m0_t, 32>) :   0.047078 ms |    467.732 TOPS (672 blocks × 4 warps/block = 2688 warps)
(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e5m2_t, float, float_ue8m0_t, 32>) :   0.047187 ms |    466.654 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled 16x8x64 E2M1, VS=32, UE8M0) ===
(SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, 32>) :   0.047197 ms |    933.118 TOPS (672 blocks × 4 warps/block = 2688 warps)

=== FP32 Output (Block-Scaled 16x8x64 E2M1, VS=32, UE4M3) ===
(SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, 32>) :   0.047072 ms |    935.592 TOPS (672 blocks × 4 warps/block = 2688 warps)

========================================
Profiling Complete
========================================
```

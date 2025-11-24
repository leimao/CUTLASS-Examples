#include <cstring>
#include <cuda_runtime.h>
#include <cute/arch/mma_sm120.hpp>
#include <cute/tensor.hpp>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <type_traits>

using namespace cute;

// Template helpers to determine array sizes
template <typename T>
struct ArraySize;

template <typename T, size_t N>
struct ArraySize<T[N]>
{
    static constexpr size_t value = N;
};

// Helper to check if MMA has scaling factor registers
template <typename MMA, typename = void>
struct HasScalingFactors : std::false_type
{
};

template <typename MMA>
struct HasScalingFactors<
    MMA, std::void_t<typename MMA::SFARegisters, typename MMA::SFBRegisters>>
    : std::true_type
{
};

// Generic MMA profiler kernel using template expansion
template <typename MMA, int NUM_ITERS>
__global__ void profile_mma_kernel()
{
    typename MMA::DRegisters d{};
    typename MMA::ARegisters a{};
    typename MMA::BRegisters b{};
    typename MMA::CRegisters c{};

    // Initialize with non-zero values to prevent optimization
    constexpr int D_SIZE = ArraySize<typename MMA::DRegisters>::value;
    constexpr int A_SIZE = ArraySize<typename MMA::ARegisters>::value;
    constexpr int B_SIZE = ArraySize<typename MMA::BRegisters>::value;
    constexpr int C_SIZE = ArraySize<typename MMA::CRegisters>::value;

    for (int idx = 0; idx < A_SIZE; ++idx)
        a[idx] = 1;
    for (int idx = 0; idx < B_SIZE; ++idx)
        b[idx] = 1;
    for (int idx = 0; idx < C_SIZE; ++idx)
        c[idx] = 1;

    // Initialize scaling factors if they exist
    if constexpr (HasScalingFactors<MMA>::value)
    {
        typename MMA::SFARegisters sfa{};
        typename MMA::SFBRegisters sfb{};
        constexpr int SFA_SIZE = ArraySize<typename MMA::SFARegisters>::value;
        constexpr int SFB_SIZE = ArraySize<typename MMA::SFBRegisters>::value;

        for (int idx = 0; idx < SFA_SIZE; ++idx)
            sfa[idx] = 1;
        for (int idx = 0; idx < SFB_SIZE; ++idx)
            sfb[idx] = 1;

#pragma unroll 1
        for (int i = 0; i < NUM_ITERS; ++i)
        {
            // Block-scaled MMA with scaling factors
            if constexpr (D_SIZE == 4 && A_SIZE == 4 && B_SIZE == 2 &&
                          C_SIZE == 4)
            {
                MMA::fma(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0],
                         b[1], c[0], c[1], c[2], c[3], sfa[0], sfb[0]);
            }

// Feedback loop
#pragma unroll
            for (int idx = 0; idx < C_SIZE; ++idx)
                c[idx] = d[idx];
        }
    }
    else
    {
#pragma unroll 1
        for (int i = 0; i < NUM_ITERS; ++i)
        {
            // Call fma with expanded parameters based on array sizes
            // SM120 non-block-scaled MMA atoms all use: D_SIZE=4, A_SIZE=4,
            // B_SIZE=2, C_SIZE=4
            if constexpr (D_SIZE == 4 && A_SIZE == 4 && B_SIZE == 2 &&
                          C_SIZE == 4)
            {
                // Pattern: All SM120 16x8x32 FP32 output MMA atoms
                MMA::fma(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0],
                         b[1], c[0], c[1], c[2], c[3]);
            }
            else if constexpr (D_SIZE == 2 && A_SIZE == 4 && B_SIZE == 2 &&
                               C_SIZE == 2)
            {
                // Pattern: All SM120 16x8x32 FP16 output MMA atoms
                MMA::fma(d[0], d[1], a[0], a[1], a[2], a[3], b[0], b[1], c[0],
                         c[1]);
            }

// Feedback loop: copy d to c to prevent optimization
#pragma unroll
            for (int idx = 0; idx < C_SIZE; ++idx)
                c[idx] = d[idx];
        }
    }

    // Use result to prevent complete elimination
    if (threadIdx.x == 0 && d[0] == 0)
    {
        printf("Should not print\n");
    }
}

// ====================================
// Profiling Helper Function
// ====================================
template <typename MMA, int NUM_ITERS = 1000>
float profile_mma(const char* name, int num_sms, int blocks_per_sm = 8,
                  int warps_per_block = 4)
{
    const int num_runs = 10;
    const int warmup_runs = 2;

    // Determine M, N, K from MMA type name
    // Check if it's 16x8x64 or 16x8x32
    const bool is_64k = (std::strstr(name, "16x8x64") != nullptr);
    const int m = 16;
    const int n = 8;
    const int k = is_64k ? 64 : 32;

    // Use multiple blocks to saturate GPU
    dim3 block(32 * warps_per_block);   // warps_per_block warps per block
    dim3 grid(num_sms * blocks_per_sm); // Multiple blocks per SM

    // Warmup
    for (int i = 0; i < warmup_runs; ++i)
    {
        profile_mma_kernel<MMA, NUM_ITERS><<<grid, block>>>();
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_runs; ++i)
    {
        profile_mma_kernel<MMA, NUM_ITERS><<<grid, block>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float avg_time = milliseconds / num_runs;

    // Calculate TOPS (Tera Operations Per Second)
    // Each MMA instruction is executed by a warp (32 threads) cooperatively
    // and computes one M×N×K matrix multiplication
    // Total operations = 2 * M * N * K * NUM_ITERS * num_warps
    int num_warps = (block.x * grid.x) / 32; // Each warp performs one MMA
    double total_ops = 2.0 * m * n * k * NUM_ITERS * num_warps;
    double time_seconds = avg_time / 1000.0;
    double tops = (total_ops / time_seconds) / 1e12;

    std::cout << std::setw(50) << std::left << name << " : " << std::setw(10)
              << std::right << std::fixed << std::setprecision(6) << avg_time
              << " ms" << " | " << std::setw(10) << std::fixed
              << std::setprecision(3) << tops << " TOPS" << " (" << grid.x
              << " blocks × " << warps_per_block
              << " warps/block = " << num_warps << " warps)" << std::endl;

    return avg_time;
}

// Helper macro to profile an MMA atom - use variadic to handle template commas
#define PROFILE_MMA(...)                                                       \
    profile_mma<__VA_ARGS__>(#__VA_ARGS__, num_sms, blocks_per_sm,             \
                             warps_per_block)

// ====================================
// Main Function
// ====================================
int main()
{
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "========================================" << std::endl;
    std::cout << "CUTLASS SM120 MMA Atom Profiler" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor
              << std::endl;
    std::cout << "Max Blocks per SM: " << prop.maxBlocksPerMultiProcessor
              << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    if (prop.major * 10 + prop.minor < 120)
    {
        std::cerr << "Error: This profiler requires SM120 or later (Blackwell "
                     "architecture)"
                  << std::endl;
        return 1;
    }

    int num_sms = prop.multiProcessorCount;
    int blocks_per_sm = 8; // Launch multiple blocks per SM for better occupancy
    int warps_per_block = 4; // Multiple warps per block to hide latency

    std::cout << "Profiling SM120 MMA Atoms (FP8 Instructions):" << std::endl;
    std::cout << "Configuration: " << num_sms << " SMs × " << blocks_per_sm
              << " blocks/SM × " << warps_per_block << " warps/block = "
              << (num_sms * blocks_per_sm * warps_per_block) << " total warps"
              << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // FP32 Output (FP8 Input) MMA Atoms - 16x8x32
    std::cout << std::endl << "=== FP32 Output (E2M1 Input) ===" << std::endl;
    PROFILE_MMA(SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e2m1_t, float_e3m2_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e2m1_t, float_e2m3_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e2m1_t, float_e4m3_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e2m1_t, float_e5m2_t, float>, 1000);

    std::cout << std::endl << "=== FP32 Output (E3M2 Input) ===" << std::endl;
    PROFILE_MMA(SM120_16x8x32_TN<float_e3m2_t, float_e2m1_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e3m2_t, float_e3m2_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e3m2_t, float_e2m3_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e3m2_t, float_e4m3_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e3m2_t, float_e5m2_t, float>, 1000);

    std::cout << std::endl << "=== FP32 Output (E2M3 Input) ===" << std::endl;
    PROFILE_MMA(SM120_16x8x32_TN<float_e2m3_t, float_e2m1_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e2m3_t, float_e3m2_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e2m3_t, float_e2m3_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e2m3_t, float_e4m3_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e2m3_t, float_e5m2_t, float>, 1000);

    std::cout << std::endl << "=== FP32 Output (E4M3 Input) ===" << std::endl;
    PROFILE_MMA(SM120_16x8x32_TN<float_e4m3_t, float_e2m1_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e4m3_t, float_e3m2_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e4m3_t, float_e2m3_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e4m3_t, float_e5m2_t, float>, 1000);

    std::cout << std::endl << "=== FP32 Output (E5M2 Input) ===" << std::endl;
    PROFILE_MMA(SM120_16x8x32_TN<float_e5m2_t, float_e2m1_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e5m2_t, float_e3m2_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e5m2_t, float_e2m3_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e5m2_t, float_e4m3_t, float>, 1000);
    PROFILE_MMA(SM120_16x8x32_TN<float_e5m2_t, float_e5m2_t, float>, 1000);

    // Block-scaled MMA Atoms with scaling factors - 16x8x32 VS=32
    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled E2M1, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m1_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e3m2_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m3_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e4m3_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e5m2_t,
                                                float, float_ue8m0_t, 32>,
        1000);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled E3M2, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e2m1_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e3m2_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e2m3_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e4m3_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e5m2_t,
                                                float, float_ue8m0_t, 32>,
        1000);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled E2M3, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e2m1_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e3m2_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e2m3_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e4m3_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e5m2_t,
                                                float, float_ue8m0_t, 32>,
        1000);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled E4M3, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e2m1_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e3m2_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e2m3_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e5m2_t,
                                                float, float_ue8m0_t, 32>,
        1000);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled E5M2, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e2m1_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e3m2_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e2m3_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e4m3_t,
                                                float, float_ue8m0_t, 32>,
        1000);
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e5m2_t,
                                                float, float_ue8m0_t, 32>,
        1000);

    // 16x8x64 Block-scaled MMA with E2M1 (FP4 format)
    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled 16x8x64 E2M1, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t,
                                                float, float_ue8m0_t, 32>,
        1000);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled 16x8x64 E2M1, VS=32, UE4M3) ==="
              << std::endl;
    PROFILE_MMA(
        SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t,
                                                float, float_ue4m3_t, 32>,
        1000);

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Profiling Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

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
template <typename MMA, size_t NUM_ITERS>
__global__ void profile_mma_kernel()
{
    typename MMA::DRegisters d{};
    typename MMA::ARegisters a{};
    typename MMA::BRegisters b{};
    typename MMA::CRegisters c{};

    // Initialize with non-zero values to prevent optimization
    constexpr int D_SIZE{ArraySize<typename MMA::DRegisters>::value};
    constexpr int A_SIZE{ArraySize<typename MMA::ARegisters>::value};
    constexpr int B_SIZE{ArraySize<typename MMA::BRegisters>::value};
    constexpr int C_SIZE{ArraySize<typename MMA::CRegisters>::value};

    for (int idx{0}; idx < A_SIZE; ++idx)
        a[idx] = 1;
    for (int idx{0}; idx < B_SIZE; ++idx)
        b[idx] = 1;
    for (int idx{0}; idx < C_SIZE; ++idx)
        c[idx] = 1;

    // Initialize scaling factors if they exist
    if constexpr (HasScalingFactors<MMA>::value)
    {
        typename MMA::SFARegisters sfa{};
        typename MMA::SFBRegisters sfb{};
        constexpr int SFA_SIZE{ArraySize<typename MMA::SFARegisters>::value};
        constexpr int SFB_SIZE{ArraySize<typename MMA::SFBRegisters>::value};

        for (int idx{0}; idx < SFA_SIZE; ++idx)
            sfa[idx] = 1;
        for (int idx{0}; idx < SFB_SIZE; ++idx)
            sfb[idx] = 1;

#pragma unroll 1
        for (size_t i{0}; i < NUM_ITERS; ++i)
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
            for (int idx{0}; idx < C_SIZE; ++idx)
                c[idx] = d[idx];
        }
    }
    else
    {
#pragma unroll 1
        for (size_t i{0}; i < NUM_ITERS; ++i)
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
            for (int idx{0}; idx < C_SIZE; ++idx)
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
template <typename MMA, size_t NUM_ITERS = 1000>
float profile_mma(char const* name, size_t num_sms, size_t blocks_per_sm = 8,
                  size_t warps_per_block = 4)
{
    size_t const num_runs{10};
    size_t const warmup_runs{2};

    // Determine M, N, K from MMA type name
    // Check if it's 16x8x64 or 16x8x32
    bool const is_64k{std::strstr(name, "16x8x64") != nullptr};
    size_t const m{16};
    size_t const n{8};
    size_t const k{is_64k ? 64u : 32u};

    // Use multiple blocks to saturate GPU
    size_t const total_threads_per_block{32u * warps_per_block};
    size_t const total_blocks{num_sms * blocks_per_sm};
    dim3 block{static_cast<unsigned int>(
        total_threads_per_block)}; // warps_per_block warps per block
    dim3 grid{
        static_cast<unsigned int>(total_blocks)}; // Multiple blocks per SM

    // Warmup
    for (size_t i{0}; i < warmup_runs; ++i)
    {
        profile_mma_kernel<MMA, NUM_ITERS><<<grid, block>>>();
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (size_t i{0}; i < num_runs; ++i)
    {
        profile_mma_kernel<MMA, NUM_ITERS><<<grid, block>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds{0};
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float avg_time{milliseconds / num_runs};

    // Calculate TOPS (Tera Operations Per Second)
    // Each MMA instruction is executed by a warp (32 threads) cooperatively
    // and computes one M×N×K matrix multiplication
    // Total operations = 2 * M * N * K * NUM_ITERS * num_warps
    size_t const num_warps{
        (static_cast<size_t>(block.x) * static_cast<size_t>(grid.x)) /
        32u}; // Each warp performs one MMA
    double total_ops{2.0 * m * n * k * NUM_ITERS * num_warps};
    double time_seconds{avg_time / 1000.0};
    double tops{(total_ops / time_seconds) / 1e12};

    std::cout << std::setw(50) << std::left << name << " : " << std::setw(10)
              << std::right << std::fixed << std::setprecision(6) << avg_time
              << " ms" << " | " << std::setw(10) << std::fixed
              << std::setprecision(3) << tops << " TOPS" << " (" << grid.x
              << " blocks × " << warps_per_block
              << " warps/block = " << num_warps << " warps)" << std::endl;

    return avg_time;
}

// Helper macro to profile an MMA atom - use variadic to handle template commas
// The MMA_TYPE should be wrapped in parentheses when it contains template
// commas Usage: PROFILE_MMA((MMA_TYPE<with, template, args>), NUM_ITERS)
#define PROFILE_MMA(MMA_TYPE, NUM_ITERS)                                       \
    profile_mma<PROFILE_MMA_UNWRAP MMA_TYPE, NUM_ITERS>(                       \
        #MMA_TYPE, num_sms, blocks_per_sm, warps_per_block)

#define PROFILE_MMA_UNWRAP(...) __VA_ARGS__

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

    constexpr size_t NUM_ITERS{1000};
    size_t num_sms{static_cast<size_t>(prop.multiProcessorCount)};
    size_t blocks_per_sm{
        8}; // Launch multiple blocks per SM for better occupancy
    size_t warps_per_block{4}; // Multiple warps per block to hide latency

    std::cout << "Profiling SM120 MMA Atoms (FP8 Instructions):" << std::endl;
    std::cout << "Configuration: " << num_sms << " SMs × " << blocks_per_sm
              << " blocks/SM × " << warps_per_block << " warps/block = "
              << (num_sms * blocks_per_sm * warps_per_block) << " total warps"
              << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // FP32 Output (FP8 Input) MMA Atoms - 16x8x32
    std::cout << std::endl << "=== FP32 Output (E2M1 Input) ===" << std::endl;
    PROFILE_MMA((SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e2m1_t, float_e3m2_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e2m1_t, float_e2m3_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e2m1_t, float_e4m3_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e2m1_t, float_e5m2_t, float>),
                NUM_ITERS);

    std::cout << std::endl << "=== FP32 Output (E3M2 Input) ===" << std::endl;
    PROFILE_MMA((SM120_16x8x32_TN<float_e3m2_t, float_e2m1_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e3m2_t, float_e3m2_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e3m2_t, float_e2m3_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e3m2_t, float_e4m3_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e3m2_t, float_e5m2_t, float>),
                NUM_ITERS);

    std::cout << std::endl << "=== FP32 Output (E2M3 Input) ===" << std::endl;
    PROFILE_MMA((SM120_16x8x32_TN<float_e2m3_t, float_e2m1_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e2m3_t, float_e3m2_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e2m3_t, float_e2m3_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e2m3_t, float_e4m3_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e2m3_t, float_e5m2_t, float>),
                NUM_ITERS);

    std::cout << std::endl << "=== FP32 Output (E4M3 Input) ===" << std::endl;
    PROFILE_MMA((SM120_16x8x32_TN<float_e4m3_t, float_e2m1_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e4m3_t, float_e3m2_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e4m3_t, float_e2m3_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e4m3_t, float_e5m2_t, float>),
                NUM_ITERS);

    std::cout << std::endl << "=== FP32 Output (E5M2 Input) ===" << std::endl;
    PROFILE_MMA((SM120_16x8x32_TN<float_e5m2_t, float_e2m1_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e5m2_t, float_e3m2_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e5m2_t, float_e2m3_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e5m2_t, float_e4m3_t, float>),
                NUM_ITERS);
    PROFILE_MMA((SM120_16x8x32_TN<float_e5m2_t, float_e5m2_t, float>),
                NUM_ITERS);

    // Block-scaled MMA Atoms with scaling factors - 16x8x32 VS=32
    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled E2M1, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m1_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e3m2_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m3_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e4m3_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e5m2_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled E3M2, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e2m1_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e3m2_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e2m3_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e4m3_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e3m2_t, float_e5m2_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled E2M3, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e2m1_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e3m2_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e2m3_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e4m3_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m3_t, float_e5m2_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled E4M3, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e2m1_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e3m2_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e2m3_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e5m2_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled E5M2, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e2m1_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e3m2_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e2m3_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e4m3_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e5m2_t, float_e5m2_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);

    // 16x8x64 Block-scaled MMA with E2M1 (FP4 format)
    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled 16x8x64 E2M1, VS=32, UE8M0) ==="
              << std::endl;
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t,
                                                 float, float_ue8m0_t, 32>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled 16x8x64 E2M1, VS=32, UE4M3) ==="
              << std::endl;
    PROFILE_MMA(
        (SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t,
                                                 float, float_ue4m3_t, 32>),
        NUM_ITERS);

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Profiling Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

#include <cuda_runtime.h>
#include <cute/arch/mma_sm80.hpp>
#include <cute/tensor.hpp>
#include <iomanip>
#include <iostream>
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

#pragma unroll 1
    for (size_t i{0}; i < NUM_ITERS; ++i)
    {
        // Call fma with expanded parameters based on array sizes
        if constexpr (D_SIZE == 2 && A_SIZE == 2 && B_SIZE == 1 && C_SIZE == 2)
        {
            // Pattern: SM80_16x8x8_F16F16F16F16_TN
            MMA::fma(d[0], d[1], a[0], a[1], b[0], c[0], c[1]);
        }
        else if constexpr (D_SIZE == 2 && A_SIZE == 4 && B_SIZE == 2 &&
                           C_SIZE == 2)
        {
            // Pattern: SM80_16x8x16_F16F16F16F16_TN
            MMA::fma(d[0], d[1], a[0], a[1], a[2], a[3], b[0], b[1], c[0],
                     c[1]);
        }
        else if constexpr (D_SIZE == 4 && A_SIZE == 2 && B_SIZE == 1 &&
                           C_SIZE == 4)
        {
            // Pattern: SM80_16x8x8_F32F16F16F32_TN,
            // SM80_16x8x4_F32TF32TF32F32_TN
            MMA::fma(d[0], d[1], d[2], d[3], a[0], a[1], b[0], c[0], c[1], c[2],
                     c[3]);
        }
        else if constexpr (D_SIZE == 4 && A_SIZE == 4 && B_SIZE == 2 &&
                           C_SIZE == 4)
        {
            // Pattern: SM80_16x8x16_F32F16F16F32_TN,
            // SM80_16x8x8_F32TF32TF32F32_TN
            MMA::fma(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0], b[1],
                     c[0], c[1], c[2], c[3]);
        }
        else if constexpr (D_SIZE == 2 && A_SIZE == 1 && B_SIZE == 1 &&
                           C_SIZE == 2)
        {
            // Pattern: SM80_8x8x16_S32S8S8S32_TN and similar INT8/INT4 variants
            MMA::fma(d[0], d[1], a[0], b[0], c[0], c[1]);
        }
        else if constexpr (D_SIZE == 4 && A_SIZE == 1 && B_SIZE == 1 &&
                           C_SIZE == 4)
        {
            // Pattern: SM80_16x8x4_F32BF16BF16F32_TN
            MMA::fma(d[0], d[1], d[2], d[3], a[0], b[0], c[0], c[1], c[2],
                     c[3]);
        }
        else if constexpr (D_SIZE == 4 && A_SIZE == 2 && B_SIZE == 2 &&
                           C_SIZE == 4)
        {
            // Pattern: SM80_16x8x8_F32BF16BF16F32_TN and FP64 variants
            MMA::fma(d[0], d[1], d[2], d[3], a[0], a[1], b[0], b[1], c[0], c[1],
                     c[2], c[3]);
        }

// Feedback loop: copy d to c to prevent optimization
#pragma unroll
        for (int idx{0}; idx < C_SIZE; ++idx)
            c[idx] = d[idx];
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
float profile_mma(char const* name, size_t m, size_t n, size_t k,
                  size_t num_sms, size_t blocks_per_sm = 8,
                  size_t warps_per_block = 4)
{
    size_t const num_runs{10};
    size_t const warmup_runs{2};

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

    std::cout << std::setw(40) << std::left << name << " : " << std::setw(10)
              << std::right << std::fixed << std::setprecision(6) << avg_time
              << " ms" << " | " << std::setw(10) << std::fixed
              << std::setprecision(3) << tops << " TOPS" << " (" << grid.x
              << " blocks × " << warps_per_block
              << " warps/block = " << num_warps << " warps)" << std::endl;

    return avg_time;
}

// Helper macro to profile an MMA atom
#define PROFILE_MMA(MMA_TYPE, M, N, K, NUM_ITERS)                              \
    profile_mma<MMA_TYPE, NUM_ITERS>(#MMA_TYPE, M, N, K, num_sms,              \
                                     blocks_per_sm, warps_per_block)

// ====================================
// Main Function
// ====================================
int main()
{
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "========================================" << std::endl;
    std::cout << "CUTLASS SM80 MMA Atom Profiler" << std::endl;
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

    if (prop.major < 8)
    {
        std::cerr << "Error: This profiler requires SM80 or later (Ampere "
                     "architecture)"
                  << std::endl;
        return 1;
    }

    constexpr size_t NUM_ITERS{1000};
    size_t num_sms{static_cast<size_t>(prop.multiProcessorCount)};
    size_t blocks_per_sm{
        8}; // Launch multiple blocks per SM for better occupancy
    size_t warps_per_block{4}; // Multiple warps per block to hide latency

    std::cout << "Profiling SM80 MMA Atoms:" << std::endl;
    std::cout << "Configuration: " << num_sms << " SMs × " << blocks_per_sm
              << " blocks/SM × " << warps_per_block << " warps/block = "
              << (num_sms * blocks_per_sm * warps_per_block) << " total warps"
              << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // FP16 Output MMA Atoms
    std::cout << std::endl << "=== FP16 Output ===" << std::endl;
    PROFILE_MMA(SM80_16x8x8_F16F16F16F16_TN, 16, 8, 8, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x16_F16F16F16F16_TN, 16, 8, 16, NUM_ITERS);

    // FP32 Output (FP16 Input) MMA Atoms
    std::cout << std::endl << "=== FP32 Output (FP16 Input) ===" << std::endl;
    PROFILE_MMA(SM80_16x8x8_F32F16F16F32_TN, 16, 8, 8, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x16_F32F16F16F32_TN, 16, 8, 16, NUM_ITERS);

    // BF16 to FP32 MMA Atoms
    std::cout << std::endl << "=== FP32 Output (BF16 Input) ===" << std::endl;
    PROFILE_MMA(SM80_16x8x8_F32BF16BF16F32_TN, 16, 8, 8, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x16_F32BF16BF16F32_TN, 16, 8, 16, NUM_ITERS);

    // TF32 to FP32 MMA Atoms
    std::cout << std::endl << "=== FP32 Output (TF32 Input) ===" << std::endl;
    PROFILE_MMA(SM80_16x8x4_F32TF32TF32F32_TN, 16, 8, 4, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x8_F32TF32TF32F32_TN, 16, 8, 8, NUM_ITERS);

    // FP64 MMA Atom
    std::cout << std::endl << "=== FP64 Output ===" << std::endl;
    PROFILE_MMA(SM80_8x8x4_F64F64F64F64_TN, 8, 8, 4, NUM_ITERS);

    // INT8 MMA Atoms (S32S8S8S32)
    std::cout << std::endl << "=== INT32 Output (S8×S8 Input) ===" << std::endl;
    PROFILE_MMA(SM80_8x8x16_S32S8S8S32_TN, 8, 8, 16, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x16_S32S8S8S32_TN, 16, 8, 16, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x32_S32S8S8S32_TN, 16, 8, 32, NUM_ITERS);
    PROFILE_MMA(SM80_8x8x16_S32S8S8S32_TN_SATURATE, 8, 8, 16, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x16_S32S8S8S32_TN_SATURATE, 16, 8, 16, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x32_S32S8S8S32_TN_SATURATE, 16, 8, 32, NUM_ITERS);

    // INT8 MMA Atoms (S32U8S8S32)
    std::cout << std::endl << "=== INT32 Output (U8×S8 Input) ===" << std::endl;
    PROFILE_MMA(SM80_8x8x16_S32U8S8S32_TN, 8, 8, 16, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x16_S32U8S8S32_TN, 16, 8, 16, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x32_S32U8S8S32_TN, 16, 8, 32, NUM_ITERS);

    // INT8 MMA Atoms (S32S8U8S32)
    std::cout << std::endl << "=== INT32 Output (S8×U8 Input) ===" << std::endl;
    PROFILE_MMA(SM80_8x8x16_S32S8U8S32_TN, 8, 8, 16, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x16_S32S8U8S32_TN, 16, 8, 16, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x32_S32S8U8S32_TN, 16, 8, 32, NUM_ITERS);

    // INT8 MMA Atoms (S32U8U8S32)
    std::cout << std::endl << "=== INT32 Output (U8×U8 Input) ===" << std::endl;
    PROFILE_MMA(SM80_8x8x16_S32U8U8S32_TN, 8, 8, 16, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x16_S32U8U8S32_TN, 16, 8, 16, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x32_S32U8U8S32_TN, 16, 8, 32, NUM_ITERS);

    // INT4 MMA Atoms (S32S4S4S32)
    std::cout << std::endl << "=== INT32 Output (S4×S4 Input) ===" << std::endl;
    PROFILE_MMA(SM80_8x8x32_S32S4S4S32_TN, 8, 8, 32, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x32_S32S4S4S32_TN, 16, 8, 32, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x64_S32S4S4S32_TN, 16, 8, 64, NUM_ITERS);

    // INT4 MMA Atoms (S32U4S4S32)
    std::cout << std::endl << "=== INT32 Output (U4×S4 Input) ===" << std::endl;
    PROFILE_MMA(SM80_8x8x32_S32U4S4S32_TN, 8, 8, 32, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x32_S32U4S4S32_TN, 16, 8, 32, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x64_S32U4S4S32_TN, 16, 8, 64, NUM_ITERS);

    // INT4 MMA Atoms (S32S4U4S32)
    std::cout << std::endl << "=== INT32 Output (S4×U4 Input) ===" << std::endl;
    PROFILE_MMA(SM80_8x8x32_S32S4U4S32_TN, 8, 8, 32, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x32_S32S4U4S32_TN, 16, 8, 32, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x64_S32S4U4S32_TN, 16, 8, 64, NUM_ITERS);

    // INT4 MMA Atoms (S32U4U4S32)
    std::cout << std::endl << "=== INT32 Output (U4×U4 Input) ===" << std::endl;
    PROFILE_MMA(SM80_8x8x32_S32U4U4S32_TN, 8, 8, 32, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x32_S32U4U4S32_TN, 16, 8, 32, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x64_S32U4U4S32_TN, 16, 8, 64, NUM_ITERS);

    // Binary MMA Atoms (ANDPOPC)
    std::cout << std::endl
              << "=== INT32 Output (U1×U1 ANDPOPC) ===" << std::endl;
    PROFILE_MMA(SM80_8x8x128_S32U1U1S32_TN_ANDPOPC, 8, 8, 128, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x128_S32U1U1S32_TN_ANDPOPC, 16, 8, 128, NUM_ITERS);
    PROFILE_MMA(SM80_16x8x256_S32U1U1S32_TN_ANDPOPC, 16, 8, 256, NUM_ITERS);

    // Binary MMA Atoms (XORPOPC) - Only available on SM80-SM89, not on SM90+
    if (prop.major * 10 + prop.minor <= 89)
    {
        std::cout << std::endl
                  << "=== INT32 Output (U1×U1 XORPOPC) ===" << std::endl;
        PROFILE_MMA(SM80_8x8x128_S32U1U1S32_TN_XORPOPC, 8, 8, 128, NUM_ITERS);
        PROFILE_MMA(SM80_16x8x128_S32U1U1S32_TN_XORPOPC, 16, 8, 128, NUM_ITERS);
        PROFILE_MMA(SM80_16x8x256_S32U1U1S32_TN_XORPOPC, 16, 8, 256, NUM_ITERS);
    }
    else
    {
        std::cout
            << std::endl
            << "=== INT32 Output (U1×U1 XORPOPC) [SKIPPED - not available on SM"
            << prop.major << prop.minor << "] ===" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Profiling Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

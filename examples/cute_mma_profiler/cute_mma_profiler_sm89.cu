#include <cuda_runtime.h>
#include <cute/arch/mma_sm89.hpp>
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

#pragma unroll 1
    for (int i = 0; i < NUM_ITERS; ++i)
    {
        // Call fma with expanded parameters based on array sizes
        // SM89 has only one pattern: D_SIZE=4, A_SIZE=4, B_SIZE=2, C_SIZE=4 for
        // F32 output and D_SIZE=2, A_SIZE=4, B_SIZE=2, C_SIZE=2 for F16 output
        if constexpr (D_SIZE == 4 && A_SIZE == 4 && B_SIZE == 2 && C_SIZE == 4)
        {
            // Pattern: All SM89 FP32 output MMA atoms (16x8x32)
            MMA::fma(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0], b[1],
                     c[0], c[1], c[2], c[3]);
        }
        else if constexpr (D_SIZE == 2 && A_SIZE == 4 && B_SIZE == 2 &&
                           C_SIZE == 2)
        {
            // Pattern: All SM89 FP16 output MMA atoms (16x8x32)
            MMA::fma(d[0], d[1], a[0], a[1], a[2], a[3], b[0], b[1], c[0],
                     c[1]);
        }

// Feedback loop: copy d to c to prevent optimization
#pragma unroll
        for (int idx = 0; idx < C_SIZE; ++idx)
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
template <typename MMA, int NUM_ITERS = 1000>
float profile_mma(const char* name, int m, int n, int k, int num_sms,
                  int blocks_per_sm = 8, int warps_per_block = 4)
{
    const int num_runs = 10;
    const int warmup_runs = 2;

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

    std::cout << std::setw(45) << std::left << name << " : " << std::setw(10)
              << std::right << std::fixed << std::setprecision(6) << avg_time
              << " ms" << " | " << std::setw(10) << std::fixed
              << std::setprecision(3) << tops << " TOPS" << " (" << grid.x
              << " blocks × " << warps_per_block
              << " warps/block = " << num_warps << " warps)" << std::endl;

    return avg_time;
}

// Helper macro to profile an MMA atom
#define PROFILE_MMA(MMA_TYPE, M, N, K)                                         \
    profile_mma<MMA_TYPE, 1000>(#MMA_TYPE, M, N, K, num_sms, blocks_per_sm,    \
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
    std::cout << "CUTLASS SM89 MMA Atom Profiler" << std::endl;
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

    if (prop.major * 10 + prop.minor < 89)
    {
        std::cerr << "Error: This profiler requires SM89 or later (Ada "
                     "Lovelace architecture)"
                  << std::endl;
        return 1;
    }

    int num_sms = prop.multiProcessorCount;
    int blocks_per_sm = 8; // Launch multiple blocks per SM for better occupancy
    int warps_per_block = 4; // Multiple warps per block to hide latency

    std::cout << "Profiling SM89 MMA Atoms (FP8 Instructions):" << std::endl;
    std::cout << "Configuration: " << num_sms << " SMs × " << blocks_per_sm
              << " blocks/SM × " << warps_per_block << " warps/block = "
              << (num_sms * blocks_per_sm * warps_per_block) << " total warps"
              << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // FP32 Output (FP8 Input) MMA Atoms - 16x8x32
    std::cout << std::endl
              << "=== FP32 Output (FP8 E4M3/E5M2 Input) ===" << std::endl;
    PROFILE_MMA(SM89_16x8x32_F32E4M3E4M3F32_TN, 16, 8, 32);
    PROFILE_MMA(SM89_16x8x32_F32E4M3E5M2F32_TN, 16, 8, 32);
    PROFILE_MMA(SM89_16x8x32_F32E5M2E4M3F32_TN, 16, 8, 32);
    PROFILE_MMA(SM89_16x8x32_F32E5M2E5M2F32_TN, 16, 8, 32);

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Profiling Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

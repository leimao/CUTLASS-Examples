#include <cuda_runtime.h>
#include <cute/arch/mma_sm70.hpp>
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

// Generic MMA benchmark kernel using template expansion
template <typename MMA, size_t NUM_ITERS>
__global__ void benchmark_mma_kernel()
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
        if constexpr (D_SIZE == 4 && A_SIZE == 2 && B_SIZE == 2 && C_SIZE == 4)
        {
            // Pattern: SM70_8x8x4_F16F16F16F16 (all layouts)
            MMA::fma(d[0], d[1], d[2], d[3], a[0], a[1], b[0], b[1], c[0], c[1],
                     c[2], c[3]);
        }
        else if constexpr (D_SIZE == 8 && A_SIZE == 2 && B_SIZE == 2 &&
                           C_SIZE == 8)
        {
            // Pattern: SM70_8x8x4_F32F16F16F32 (all layouts)
            MMA::fma(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], a[0], a[1],
                     b[0], b[1], c[0], c[1], c[2], c[3], c[4], c[5], c[6],
                     c[7]);
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
// Benchmark Helper Function
// ====================================
template <typename MMA, size_t NUM_ITERS = 1000>
float benchmark_mma(char const* name, size_t m, size_t n, size_t k,
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
        benchmark_mma_kernel<MMA, NUM_ITERS><<<grid, block>>>();
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (size_t i{0}; i < num_runs; ++i)
    {
        benchmark_mma_kernel<MMA, NUM_ITERS><<<grid, block>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds{0};
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float avg_time{milliseconds / num_runs};

    // Calculate TOPS (Tera Operations Per Second)
    // SM70 MMA instructions operate on 8-thread groups, not full warps
    // Each warp (32 threads) has 4 groups of 8 threads
    // Each group computes one M×N×K matrix multiplication
    // Total operations = 2 * M * N * K * NUM_ITERS * num_warps * 4
    size_t const num_warps{
        (static_cast<size_t>(block.x) * static_cast<size_t>(grid.x)) / 32u};
    double total_ops{2.0 * m * n * k * NUM_ITERS * num_warps * 4.0};
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

// Helper macro to benchmark an MMA atom
#define BENCHMARK_MMA(MMA_TYPE, M, N, K, NUM_ITERS)                            \
    benchmark_mma<MMA_TYPE, NUM_ITERS>(#MMA_TYPE, M, N, K, num_sms,            \
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
    std::cout << "CUTLASS SM70 MMA Atom Benchmark" << std::endl;
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

    if (prop.major < 7)
    {
        std::cerr << "Error: This benchmark requires SM70 or later (Volta "
                     "architecture)"
                  << std::endl;
        return 1;
    }

    constexpr size_t NUM_ITERS{1000};
    size_t num_sms{static_cast<size_t>(prop.multiProcessorCount)};
    size_t blocks_per_sm{
        8}; // Launch multiple blocks per SM for better occupancy
    size_t warps_per_block{4}; // Multiple warps per block to hide latency

    std::cout << "Benchmarking SM70 MMA Atoms:" << std::endl;
    std::cout << "Configuration: " << num_sms << " SMs × " << blocks_per_sm
              << " blocks/SM × " << warps_per_block << " warps/block = "
              << (num_sms * blocks_per_sm * warps_per_block) << " total warps"
              << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // FP16 Output MMA Atoms (8x8x4)
    std::cout << std::endl << "=== FP16 Output ===" << std::endl;
    BENCHMARK_MMA(SM70_8x8x4_F16F16F16F16_TN, 8, 8, 4, NUM_ITERS);
    BENCHMARK_MMA(SM70_8x8x4_F16F16F16F16_NT, 8, 8, 4, NUM_ITERS);
    BENCHMARK_MMA(SM70_8x8x4_F16F16F16F16_NN, 8, 8, 4, NUM_ITERS);
    BENCHMARK_MMA(SM70_8x8x4_F16F16F16F16_TT, 8, 8, 4, NUM_ITERS);

    // FP32 Output (FP16 Input) MMA Atoms (8x8x4)
    std::cout << std::endl << "=== FP32 Output (FP16 Input) ===" << std::endl;
    BENCHMARK_MMA(SM70_8x8x4_F32F16F16F32_TN, 8, 8, 4, NUM_ITERS);
    BENCHMARK_MMA(SM70_8x8x4_F32F16F16F32_NT, 8, 8, 4, NUM_ITERS);
    BENCHMARK_MMA(SM70_8x8x4_F32F16F16F32_NN, 8, 8, 4, NUM_ITERS);
    BENCHMARK_MMA(SM70_8x8x4_F32F16F16F32_TT, 8, 8, 4, NUM_ITERS);

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Benchmarking Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

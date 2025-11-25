#include <cstring>
#include <cuda_runtime.h>
#include <cute/arch/mma_sm120_sparse.hpp>
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
    static constexpr size_t value{N};
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

// Generic Sparse MMA benchmark kernel using template expansion
template <typename MMA, size_t NUM_ITERS>
__global__ void benchmark_sparse_mma_kernel()
{
    typename MMA::DRegisters d{};
    typename MMA::ARegisters a{};
    typename MMA::BRegisters b{};
    typename MMA::CRegisters c{};
    typename MMA::ERegisters e{}; // Sparse metadata

    // Initialize with non-zero values to prevent optimization
    constexpr size_t D_SIZE{ArraySize<typename MMA::DRegisters>::value};
    constexpr size_t A_SIZE{ArraySize<typename MMA::ARegisters>::value};
    constexpr size_t B_SIZE{ArraySize<typename MMA::BRegisters>::value};
    constexpr size_t C_SIZE{ArraySize<typename MMA::CRegisters>::value};
    constexpr size_t E_SIZE{ArraySize<typename MMA::ERegisters>::value};

#pragma unroll
    for (size_t idx{0}; idx < A_SIZE; ++idx)
        a[idx] = 1;
#pragma unroll
    for (size_t idx{0}; idx < B_SIZE; ++idx)
        b[idx] = 1;
#pragma unroll
    for (size_t idx{0}; idx < C_SIZE; ++idx)
        c[idx] = 1;
#pragma unroll
    for (size_t idx{0}; idx < E_SIZE; ++idx)
        e[idx] = 0xAAAAAAAA; // 2:4 sparsity pattern

    // Initialize scaling factors if they exist
    if constexpr (HasScalingFactors<MMA>::value)
    {
        typename MMA::SFARegisters sfa{};
        typename MMA::SFBRegisters sfb{};
        constexpr size_t SFA_SIZE{ArraySize<typename MMA::SFARegisters>::value};
        constexpr size_t SFB_SIZE{ArraySize<typename MMA::SFBRegisters>::value};

#pragma unroll
        for (size_t idx{0}; idx < SFA_SIZE; ++idx)
            sfa[idx] = 1;
#pragma unroll
        for (size_t idx{0}; idx < SFB_SIZE; ++idx)
            sfb[idx] = 1;

#pragma unroll 1
        for (size_t i{0}; i < NUM_ITERS; ++i)
        {
            // Block-scaled sparse MMA with scaling factors
            if constexpr (D_SIZE == 4 && A_SIZE == 4 && B_SIZE == 4 &&
                          C_SIZE == 4 && E_SIZE == 1)
            {
                // 16x8x64 or 16x8x128 block-scaled sparse
                MMA::fma(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0],
                         b[1], b[2], b[3], c[0], c[1], c[2], c[3], e[0], sfa[0],
                         sfb[0]);
            }

// Feedback loop
#pragma unroll
            for (size_t idx{0}; idx < C_SIZE; ++idx)
                c[idx] = d[idx];
        }
    }
    else
    {
#pragma unroll 1
        for (size_t i{0}; i < NUM_ITERS; ++i)
        {
            // Call fma with expanded parameters based on array sizes
            // SM120 sparse non-block-scaled MMA atoms: 16x8x64
            if constexpr (D_SIZE == 4 && A_SIZE == 4 && B_SIZE == 4 &&
                          C_SIZE == 4 && E_SIZE == 1)
            {
                // Pattern: All SM120 sparse 16x8x64 FP32 output MMA atoms
                MMA::fma(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0],
                         b[1], b[2], b[3], c[0], c[1], c[2], c[3], e[0]);
            }

// Feedback loop: copy d to c to prevent optimization
#pragma unroll
            for (size_t idx{0}; idx < C_SIZE; ++idx)
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
// Benchmark Helper Function
// ====================================
template <typename MMA, size_t NUM_ITERS = 1000>
float benchmark_sparse_mma(char const* name, size_t num_sms,
                           size_t blocks_per_sm = 8, size_t warps_per_block = 4)
{
    size_t const num_runs{10};
    size_t const warmup_runs{2};

    // Determine M, N, K from MMA type name
    // Check if it's 16x8x128 or 16x8x64
    bool const is_128k{std::strstr(name, "16x8x128") != nullptr};
    size_t const m{16};
    size_t const n{8};
    size_t const k{is_128k ? 128u : 64u};

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
        benchmark_sparse_mma_kernel<MMA, NUM_ITERS><<<grid, block>>>();
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (size_t i{0}; i < num_runs; ++i)
    {
        benchmark_sparse_mma_kernel<MMA, NUM_ITERS><<<grid, block>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds{0};
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float avg_time{milliseconds / num_runs};

    // Calculate TOPS (Tera Operations Per Second)
    // For sparse (2:4) MMA: NVIDIA reports "effective TOPS" based on full
    // matrix dimensions Even though 50% of values are zero, a 16x8x64 sparse
    // MMA is reported as performing the same 2*M*N*K operations as a dense
    // 16x8x64 MMA for throughput purposes Each MMA instruction is executed by a
    // warp (32 threads) cooperatively Total effective operations = 2 * M * N *
    // K * NUM_ITERS * num_warps
    size_t const num_warps{
        (static_cast<size_t>(block.x) * static_cast<size_t>(grid.x)) /
        32u}; // Each warp performs one MMA
    double total_ops{2.0 * m * n * k * NUM_ITERS * num_warps};
    double time_seconds{avg_time / 1000.0};
    double tops{(total_ops / time_seconds) / 1e12};

    std::cout << std::setw(90) << std::left << name << " : " << std::setw(10)
              << std::right << std::fixed << std::setprecision(6) << avg_time
              << " ms" << " | " << std::setw(10) << std::fixed
              << std::setprecision(3) << tops << " TOPS" << " (" << grid.x
              << " blocks × " << warps_per_block
              << " warps/block = " << num_warps << " warps)" << std::endl;

    return avg_time;
}

// Helper macro to benchmark a sparse MMA atom - use variadic to handle template
// commas. The MMA_TYPE should be wrapped in parentheses when it contains
// commas. Usage: BENCHMARK_SPARSE_MMA((MMA_TYPE<with, template, args>),
// NUM_ITERS)
#define BENCHMARK_SPARSE_MMA(MMA_TYPE, NUM_ITERS)                              \
    benchmark_sparse_mma<BENCHMARK_SPARSE_MMA_UNWRAP MMA_TYPE, NUM_ITERS>(     \
        #MMA_TYPE, num_sms, blocks_per_sm, warps_per_block)

#define BENCHMARK_SPARSE_MMA_UNWRAP(...) __VA_ARGS__

// ====================================
// Main Function
// ====================================
int main()
{
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "========================================" << std::endl;
    std::cout << "CUTLASS SM120 Sparse MMA Atom Benchmark" << std::endl;
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
        std::cerr << "Error: This benchmark requires SM120 or later (Blackwell "
                     "architecture)"
                  << std::endl;
        return 1;
    }

    constexpr size_t NUM_ITERS{1000};
    size_t num_sms{static_cast<size_t>(prop.multiProcessorCount)};
    size_t blocks_per_sm{
        8}; // Launch multiple blocks per SM for better occupancy
    size_t warps_per_block{4}; // Multiple warps per block to hide latency

    std::cout
        << "Benchmarking SM120 Sparse MMA Atoms (FP8 2:4 Structured Sparsity):"
        << std::endl;
    std::cout << "Configuration: " << num_sms << " SMs × " << blocks_per_sm
              << " blocks/SM × " << warps_per_block << " warps/block = "
              << (num_sms * blocks_per_sm * warps_per_block) << " total warps"
              << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Non-block-scaled Sparse MMA Atoms - 16x8x64
    std::cout << std::endl
              << "=== FP32 Output (Sparse E2M1 Input, 16x8x64) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e2m1_t, float_e2m1_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e2m1_t, float_e3m2_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e2m1_t, float_e2m3_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e2m1_t, float_e4m3_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e2m1_t, float_e5m2_t,
                                                float>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Sparse E3M2 Input, 16x8x64) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e3m2_t, float_e2m1_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e3m2_t, float_e3m2_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e3m2_t, float_e2m3_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e3m2_t, float_e4m3_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e3m2_t, float_e5m2_t,
                                                float>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Sparse E2M3 Input, 16x8x64) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e2m3_t, float_e2m1_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e2m3_t, float_e3m2_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e2m3_t, float_e2m3_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e2m3_t, float_e4m3_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e2m3_t, float_e5m2_t,
                                                float>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Sparse E4M3 Input, 16x8x64) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e4m3_t, float_e2m1_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e4m3_t, float_e3m2_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e4m3_t, float_e2m3_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e4m3_t, float_e4m3_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e4m3_t, float_e5m2_t,
                                                float>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Sparse E5M2 Input, 16x8x64) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e5m2_t, float_e2m1_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e5m2_t, float_e3m2_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e5m2_t, float_e2m3_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e5m2_t, float_e4m3_t,
                                                float>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::SPARSE::SM120_SPARSE_16x8x64_TN<float_e5m2_t, float_e5m2_t,
                                                float>),
        NUM_ITERS);

    // Block-scaled Sparse MMA Atoms with scaling factors - 16x8x64 VS=64
    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled Sparse E2M1, 16x8x64, VS=64, "
                 "UE8M0) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e2m1_t, float_e3m2_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e2m1_t, float_e2m3_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e2m1_t, float_e4m3_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e2m1_t, float_e5m2_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled Sparse E3M2, 16x8x64, VS=64, "
                 "UE8M0) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e3m2_t, float_e2m1_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e3m2_t, float_e3m2_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e3m2_t, float_e2m3_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e3m2_t, float_e4m3_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e3m2_t, float_e5m2_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled Sparse E2M3, 16x8x64, VS=64, "
                 "UE8M0) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e2m3_t, float_e2m1_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e2m3_t, float_e3m2_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e2m3_t, float_e2m3_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e2m3_t, float_e4m3_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e2m3_t, float_e5m2_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled Sparse E4M3, 16x8x64, VS=64, "
                 "UE8M0) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e4m3_t, float_e2m1_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e4m3_t, float_e3m2_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e4m3_t, float_e2m3_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e4m3_t, float_e5m2_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled Sparse E5M2, 16x8x64, VS=64, "
                 "UE8M0) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e5m2_t, float_e2m1_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e5m2_t, float_e3m2_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e5m2_t, float_e2m3_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e5m2_t, float_e4m3_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x64_TN_VS<
            float_e5m2_t, float_e5m2_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);

    // 16x8x128 Block-scaled Sparse MMA with E2M1 (FP4 format) - VS=64 or VS=32
    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled Sparse 16x8x128 E2M1, VS=64, "
                 "UE8M0) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x128_TN_VS<
            float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, 64>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled Sparse 16x8x128 E2M1, VS=32, "
                 "UE8M0) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x128_TN_VS<
            float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, 32>),
        NUM_ITERS);

    std::cout << std::endl
              << "=== FP32 Output (Block-Scaled Sparse 16x8x128 E2M1, VS=32, "
                 "UE4M3) ==="
              << std::endl;
    BENCHMARK_SPARSE_MMA(
        (SM120::BLOCKSCALED::SPARSE::SM120_SPARSE_16x8x128_TN_VS<
            float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, 32>),
        NUM_ITERS);

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Benchmarking Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

// #include "cutlass/util/GPU_Clock.hpp"
// #include "cutlass/util/helper_cuda.hpp"
// #include "cutlass/util/print_error.hpp"

template <class ProblemShape, class CtaTiler, class TA, class AStride,
          class ASmemLayout, class TiledCopyA, class TB, class BStride,
          class BSmemLayout, class TiledCopyB, class TC, class CStride,
          class CSmemLayout, class TiledMma, class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(
    TiledMma{}))::value) void gemm_device(ProblemShape shape_MNK,
                                          CtaTiler cta_tiler, TA const* A,
                                          AStride dA, ASmemLayout sA_layout,
                                          TiledCopyA copy_a, TB const* B,
                                          BStride dB, BSmemLayout sB_layout,
                                          TiledCopyB copy_b, TC* C, CStride dC,
                                          CSmemLayout, TiledMma mma,
                                          Alpha alpha, Beta beta)
{
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma)); // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma)); // NumThreads

    CUTE_STATIC_ASSERT(is_static<ASmemLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BSmemLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

    CUTE_STATIC_ASSERT_V(
        congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(
        congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(
        congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MN

    //
    // Full and Tiled Tensors
    //

    // Represent the full tensors
    Tensor mA =
        make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
    Tensor mB =
        make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
    Tensor mC =
        make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord,
                           Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord,
                           Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord,
                           Step<_1, _1, X>{}); // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K)

    //
    // Partition the copying of A and B tiles across the threads
    //

    // TUTORIAL: Example of partitioning via a TiledCopy

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K)
    // Allocate registers same shape/layout as partitioned data
    Tensor tArA = make_fragment_like(tAsA); // (CPY,CPY_M,CPY_K)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K)
    // Allocate registers same shape/layout as partitioned data
    Tensor tBrB = make_fragment_like(tBsB); // (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB)); // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB)); // CPY_K

    // Copy gmem to rmem for k_tile=0
    copy(copy_a, tAgA(_, _, _, 0), tArA);
    copy(copy_b, tBgB(_, _, _, 0), tBrB);
    //
    // Define A/B partitioning and C accumulators
    //

    // TUTORIAL: Example of partitioning via a TiledMMA

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

    // Allocate the accumulators -- same size as the projected data
    // Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)
    Tensor tCrC = make_fragment_like(tCgC); // (MMA,MMA_M,MMA_N)

    CUTE_STATIC_ASSERT_V(shape(tCrC) == shape(tCgC));     // (MMA,MMA_M,MMA_N)
    CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA)); // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB)); // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB)); // MMA_K

    // Clear the accumulators
    clear(tCrC);

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1

    // TUTORIAL: Example of an inner loop that pipelines compute with reads
    //           from global memory by staging through register and shared
    //           memory.
    //   Data is read from global to registers, then to shared via the TiledCopy
    //   partitions gemm(.) operates on the shared memory directly via the
    //   TiledMMA partitions

    auto K_TILE_MAX = size<3>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        // Copy rmem to smem with tA|tB thread-partitioned tensors
        __syncthreads(); // Wait for all threads to consume smem
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads(); // Wait for all threads to consume smem

        // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors
        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
        copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);
        // TUTORIAL: The above call to copy(copy_a, tAgA(_,_,_,k_tile_next),
        // tArA) is equivalent to
        //   CUTE_UNROLL
        //   for (int k = 0; k < size<1>(tCsA); ++k) {
        //     CUTE_UNROLL
        //     for (int m = 0; m < size<0>(tCrC); ++m) {
        //       copy_a.call(tAgA(_,m,k), tArA(_,m,k);
        //     }
        //   }

        // Compute gemm on mma-partitioned smem
        gemm(mma, tCsA, tCsB, tCrC);
        // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
        //   CUTE_UNROLL
        //   for (int k = 0; k < size<1>(tCsA); ++k) {
        //     CUTE_UNROLL
        //     for (int m = 0; m < size<0>(tCrC); ++m) {
        //       CUTE_UNROLL
        //       for (int n = 0; n < size<1>(tCrC); ++n) {
        //         mma.call(tCsA(_,m,k), tCsB(_,n,k), tCrC(_,m,n);
        //       }
        //     }
        //   }
    }

#endif

    //
    // Epilogue
    //

    axpby(alpha, tCrC, beta, tCgC);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC, class Alpha, class Beta>
static void gemm_nt(int m, int n, int k, Alpha alpha, TA const* A, int ldA,
                    TB const* B, int ldB, Beta beta, TC* C, int ldC,
                    cudaStream_t stream = 0)
{
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K); // (M, N, K)

    // Define NT strides (mixed)
    auto dA = make_stride(Int<1>{}, ldA); // (dM, dK)
    auto dB = make_stride(Int<1>{}, ldB); // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128 * 4 / sizeof(TA)>{};
    auto bN = Int<128 * 4 / sizeof(TB)>{};
    auto bK = Int<32>{};
    auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

    // Define the smem layouts (static)
    auto sA = make_layout(make_shape(bM, bK)); // (m,k) -> smem_idx; m-major
    auto sB = make_layout(make_shape(bN, bK)); // (n,k) -> smem_idx; n-major
    auto sC = make_layout(make_shape(bM, bN)); // (m,n) -> smem_idx; m-major

    // Define the thread layouts (static)

    // TUTORIAL: Construct TiledCopy with a particular Copy_Atom to use and
    //           define the partitioning pattern to apply.
    // Each thread will (try to) copy 4x1 elements of type TA using 128-bit
    // copy. Use 32x8 of these threads.

    TiledCopy copyA =
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
                        Layout<Shape<_32, _8>>{}, // Thr layout 32x8 m-major
                        Layout<Shape<cute::Int<sizeof(uint128_t) / sizeof(TA)>,
                                     _1>>{}); // Val layout  4x1 m-major
    TiledCopy copyB =
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
                        Layout<Shape<_32, _8>>{}, // Thr layout 32x8 n-major
                        Layout<Shape<cute::Int<sizeof(uint128_t) / sizeof(TB)>,
                                     _1>>{}); // Val layout  4x1 n-major

    // TUTORIAL: Construct TiledMMA with a particular MMA_Atom to use and
    //           define the partitioning pattern to apply.
    // Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single
    // thread. Reproduce that atom 16x16x1 times (m-major) across threads so
    // that we use 256 threads.

    // TiledMMA mmaC =
    //     make_tiled_mma(UniversalFMA<TC, TA, TB>{},
    //                    Layout<Shape<_16, _16, _1>>{}); // 16x16x1
    //                    UniversalFMA

    // TiledMMA mmaC = make_tiled_mma(SM70_8x8x4_F16F16F16F16_NT{},
    //                               Layout<Shape <_4,_8>,
    //                                      Stride<_1,_4>>{});   // 2x2 n-major
    //                                      layout of Atoms

    // TiledMMA mmaC = make_tiled_mma(SM70_8x8x4_F16F16F16F16_NT{},
    //                               Layout<Shape <_4,_8>, Stride<_1,_4>>{},
    //                               Layout<Shape <_2,_2>>{});   // 2x2 n-major
    //                               layout of Atoms

    // TiledMMA mmaC = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
    //                               Layout<Shape <_4,_8>,
    //                                      Stride<_1,_4>>{});   // 2x2 n-major
    //                                      layout of Atoms

    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    // using MMA = decltype(make_tiled_mma(mma_atom{},
    //                     make_layout(Shape<_4, _2, _1>{}),
    //                     make_layout(Shape<_1, _2, _1>{})));

    using MMA =
        decltype(make_tiled_mma(mma_atom{}, make_layout(Shape<_4, _2, _1>{})));

    TiledMMA mmaC = MMA{};

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA,
                                                  sA, copyA, B, dB, sB, copyB,
                                                  C, dC, sC, mmaC, alpha, beta);
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC, class Alpha, class Beta>
static void gemm_tn(int m, int n, int k, Alpha alpha, TA const* A, int ldA,
                    TB const* B, int ldB, Beta beta, TC* C, int ldC,
                    cudaStream_t stream = 0)
{
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K); // (M, N, K)

    // Define TN strides (mixed)
    auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
    auto dB = make_stride(ldB, Int<1>{}); // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

    // Define the smem layouts (static)
    auto sA = make_layout(
        make_shape(bM, bK),
        make_stride(Int<1>{},
                    bM + Int<1>{})); // (m,k) -> smem_idx; padded m-major
    auto sB = make_layout(
        make_shape(bN, bK),
        make_stride(Int<1>{},
                    bN + Int<1>{})); // (n,k) -> smem_idx; padded n-major
    auto sC = make_layout(make_shape(bM, bN)); // (m,n) -> smem_idx

    // TUTORIAL: Construct TiledCopy to define the Copy_Atom to use and the
    //           partitioning pattern to apply.
    // Each thread will copy 1x1 elements of type TA.
    // Use 32x8 of these threads arranged in k-major.

    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<TA>, TA>{},
        Layout<Shape<_32, _8>, Stride<_8, _1>>{}, // Thr layout 32x8 k-major
        Layout<Shape<_1, _1>>{});                 // Val layout  1x1
    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<UniversalCopy<TB>, TB>{},
        Layout<Shape<_32, _8>, Stride<_8, _1>>{}, // Thr layout 32x8 k-major
        Layout<Shape<_1, _1>>{});                 // Val layout  1x1

    // TUTORIAL: Construct TiledMMA to define the MMA_Atom to use and the
    //           partitioning pattern to apply.
    // Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single
    // thread. Reproduce that atom 16x16x1 times (m-major) across threads so
    // that we use 256 threads.

    TiledMMA mmaC =
        make_tiled_mma(UniversalFMA<TC, TA, TB>{},
                       Layout<Shape<_16, _16, _1>>{}); // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA,
                                                  sA, copyA, B, dB, sB, copyB,
                                                  C, dC, sC, mmaC, alpha, beta);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
cudaError_t launch_sgemm_2(char transA, char transB, int m, int n, int k,
                           Alpha alpha, TA const* A, int ldA, TB const* B,
                           int ldB, Beta beta, TC* C, int ldC,
                           cudaStream_t stream = 0)
{
    if (transA == 'N' && transB == 'T')
    {
        gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    // else if (transA == 'T' && transB == 'N')
    // {
    //     gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    // }
    else
    {
        assert(false && "Not implemented");
    }
    return cudaGetLastError();
}

// Explicit instantiation
// template cudaError_t launch_sgemm_2<float, float, float, float, float>(
//     char transA, char transB, int m, int n, int k, float alpha, float const*
//     A, int ldA, float const* B, int ldB, float beta, float* C, int ldC,
//     cudaStream_t stream);
// template cudaError_t launch_sgemm_2<double, double, double, double, double>(
//     char transA, char transB, int m, int n, int k, double alpha,
//     double const* A, int ldA, double const* B, int ldB, double beta, double*
//     C, int ldC, cudaStream_t stream);
template cudaError_t
launch_sgemm_2<cute::half_t, cute::half_t, cute::half_t, float, float>(
    char transA, char transB, int m, int n, int k, float alpha,
    cute::half_t const* A, int ldA, cute::half_t const* B, int ldB, float beta,
    cute::half_t* C, int ldC, cudaStream_t stream);
template cudaError_t
launch_sgemm_2<cute::half_t, cute::half_t, cute::half_t, cute::half_t,
               cute::half_t>(char transA, char transB, int m, int n, int k,
                             cute::half_t alpha, cute::half_t const* A, int ldA,
                             cute::half_t const* B, int ldB, cute::half_t beta,
                             cute::half_t* C, int ldC, cudaStream_t stream);

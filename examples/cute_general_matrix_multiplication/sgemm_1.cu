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
          class ASmemLayout, class AThreadLayout, class TB, class BStride,
          class BSmemLayout, class BThreadLayout, class TC, class CStride,
          class CSmemLayout, class CThreadLayout, class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(
    CThreadLayout{}))::value) void gemm_device(ProblemShape shape_MNK,
                                               CtaTiler cta_tiler, TA const* A,
                                               AStride dA,
                                               ASmemLayout sA_layout,
                                               AThreadLayout tA, TB const* B,
                                               BStride dB,
                                               BSmemLayout sB_layout,
                                               BThreadLayout tB, TC* C,
                                               CStride dC, CSmemLayout,
                                               CThreadLayout tC, Alpha alpha,
                                               Beta beta)
{
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

    static_assert(is_static<AThreadLayout>::value);
    static_assert(is_static<BThreadLayout>::value);
    static_assert(is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size(tA) == size(tB)); // NumThreads
    CUTE_STATIC_ASSERT_V(size(tC) == size(tA)); // NumThreads

    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) ==
                         Int<0>{}); // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) ==
                         Int<0>{}); // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) ==
                         Int<0>{}); // BLK_N / THR_N
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) ==
                         Int<0>{}); // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) ==
                         Int<0>{}); // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) ==
                         Int<0>{}); // BLK_N / THR_N

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    // ASmemLayout: (BLK_M, BLK_K)
    // BSmemLayout: (BLK_N, BLK_K)
    // CSmemLayout: (BLK_M, BLK_N)
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
    // cta_coord: (m, n, :)
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,:)
    // cta_tiler: (BLK_M, BLK_N, BLK_K)
    // With Step<_1, X, _1>{}, the second mode in the cta tiler is ignored, thus
    // the tiler becomes (BLK_M, BLK_K) In addition, because the the second mode
    // is ignored, the cta_coord becomes (m, :) The resulting local_tile is
    // (BLK_M, BLK_K, k) where BLK_K * k = K.
    Tensor gA = local_tile(mA, cta_tiler, cta_coord,
                           Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
    // With Step<X, _1, _1>{}, the first mode in the cta tiler is ignored, thus
    // the tiler becomes (BLK_N, BLK_K) In addition, because the the first mode
    // is ignored, the cta_coord becomes (n, :) The resulting local_tile is
    // (BLK_N, BLK_K, k) where BLK_K * k = K.
    Tensor gB = local_tile(mB, cta_tiler, cta_coord,
                           Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    // With Step<_1, _1, X>{}, the third mode in the cta tiler is ignored, thus
    // the tiler becomes (BLK_M, BLK_N) In addition, because the the third mode
    // is ignored, the cta_coord becomes (m, n) The resulting local_tile is
    // (BLK_M, BLK_N).
    Tensor gC = local_tile(mC, cta_tiler, cta_coord,
                           Step<_1, _1, X>{}); // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    // sA and sB can be column major or row major.
    // Whether it is column major or row major depends on the gA and gB layout.
    // If gA is column major, then sA is column major.
    // If gA is row major, then sA is row major.
    // If gB is column major, then sB is column major.
    // If gB is row major, then sB is row major.
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K)

    //
    // Partition the copying of A and B tiles across the threads
    //

    // TUTORIAL: Example of simple raked partitioning of ThreadLayouts tA|tB
    // over data A|B tiles

    Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M,THR_K,k)
    Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M,THR_K)

    Tensor tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N,THR_K,k)
    Tensor tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N,THR_K)

    CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA)); // THR_M
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // THR_K
    CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB)); // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // THR_K

    //
    // Define A/B partitioning and C accumulators
    //

    // TUTORIAL: Example of partitioning via projections of a ThreadLayout tC

    // Each thread needs to read some data from A and B, depending on its
    // coordinates in tC.

    // Partition sA (M,K) by the rows of tC
    // Different threads in the same column of tC will read the same data from
    // sA. Because of the Step<_1, X>{} projection, the second mode in the tC
    // layout is ignored.

    Tensor tCsA =
        local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // (THR_M,BLK_K)
    // Partition sB (N,K) by the cols of tC
    // Different threads in the same row of tC will read the same data from sB.
    // Because of the Step<X, _1>{} projection, the first mode in the tC layout
    // is ignored.
    Tensor tCsB =
        local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // (THR_N,BLK_K)
    // Partition gC (M,N) by the tile of tC
    Tensor tCgC =
        local_partition(gC, tC, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)

    // Allocate the accumulators -- same shape/layout as the partitioned data
    // The layout is automatically compacted to the smallest possible layout to
    // avoid unnecessary memory/register usage.
    Tensor tCrC = make_tensor_like(tCgC); // (THR_M,THR_N)
    // auto tCrC = make_fragment_like(tCgC); // (THR_M,(THR_N))

    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC)); // THR_M
    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA)); // THR_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC)); // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB)); // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB)); // BLK_K

    // Clear the accumulators
    clear(tCrC);

    // if (thread0())
    // {
    //     print("cta_coord:");
    //     print(cta_coord);
    //     print("\n");
    //     print("  mA : ");
    //     print(mA);
    //     print("\n");
    //     print("  gA : ");
    //     print(gA);
    //     print("\n");
    //     print("  sA : ");
    //     print(sA);
    //     print("\n");
    //     print("tAgA : ");
    //     print(tAgA);
    //     print("\n");
    //     print("tAsA : ");
    //     print(tAsA);
    //     print("\n");
    // }

    // if (thread0())
    // {
    //     print("  mB : ");
    //     print(mB);
    //     print("\n");
    //     print("  gB : ");
    //     print(gB);
    //     print("\n");
    //     print("  sB : ");
    //     print(sB);
    //     print("\n");
    //     print("tBgB : ");
    //     print(tBgB);
    //     print("\n");
    //     print("tBsB : ");
    //     print(tBsB);
    //     print("\n");
    // }

    // if (thread0())
    // {
    //     print("  mC : ");
    //     print(mC);
    //     print("\n");
    //     print("  gC : ");
    //     print(gC);
    //     print("\n");
    //     print("tCsA : ");
    //     print(tCsA);
    //     print("\n");
    //     print("tCsB : ");
    //     print(tCsB);
    //     print("\n");
    //     print("tCgC : ");
    //     print(tCgC);
    //     print("\n");
    //     print("tCrC : ");
    //     print(tCrC);
    //     print("\n");
    // }

#if 1

    // TUTORIAL: Example of a simple mainloop that read tiles of data into
    // shared memory,
    //           and then computes on those tiles.
    //   copy(.) operates on the global and shared memory via the tA|tB
    //   partitioning gemm(.) operates on the shared and register memory via the
    //   tC partitioning

    auto K_TILE_MAX = size<2>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        // Copy gmem to smem with tA|tB thread-partitioned tensors
        copy(tAgA(_, _, k_tile), tAsA); // A   (THR_M,THR_K) -> (THR_M,THR_K)
        copy(tBgB(_, _, k_tile), tBsB); // B   (THR_N,THR_K) -> (THR_N,THR_K)

        // TUTORIAL: The above call to copy(tAgA(_,_,k_tile), tAsA) is
        // equivalent to
        //   Tensor tAgAk = tAgA(_,_,k_tile);
        //   CUTE_UNROLL
        //   for (int i = 0; i < size(tAsA); ++i) {
        //     tAsA(i) = tAgAk(i);
        //   }

        cp_async_fence(); // Label the end of (potential) cp.async instructions
        cp_async_wait<0>(); // Sync on all (potential) cp.async instructions
        __syncthreads();    // Wait for all threads to write to smem

        // Compute gemm on tC thread-partitioned smem
        gemm(tCsA, tCsB,
             tCrC); // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)

        // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
        //   CUTE_UNROLL
        //   for (int k = 0; k < size<1>(tCsA); ++k) {
        //     CUTE_UNROLL
        //     for (int m = 0; m < size<0>(tCrC); ++m) {
        //       CUTE_UNROLL
        //       for (int n = 0; n < size<1>(tCrC); ++n) {
        //         tCrC(m,n) += tCsA(m,k) * tCsB(n,k);
        //       }
        //     }
        //   }

        __syncthreads(); // Wait for all threads to read from smem
    }

#endif

    //
    // Epilogue
    //

    axpby(alpha, tCrC, beta, tCgC);

    // TUTORIAL: The above call to axpby(alpha, tCrC, beta, tCgC) is equivalent
    // to
    //   CUTE_UNROLL
    //   for (int i = 0; i < size(tCsA); ++i) {
    //     tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
    //   }
}

// Setup params for an NT GEMM
// Use m-major smem sA, n-major smem sB, and mn-major threads tA|tB
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
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

    // Define the smem layouts (static)
    auto sA = make_layout(make_shape(bM, bK)); // (m,k) -> smem_idx; m-major
    auto sB = make_layout(make_shape(bN, bK)); // (n,k) -> smem_idx; n-major
    auto sC = make_layout(make_shape(bM, bN)); // (m,n) -> smem_idx; m-major

    // Define the thread layouts (static)
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (m,k) -> thr_idx
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (n,k) -> thr_idx
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{})); // (m,n) -> thr_idx

    dim3 dimBlock(size(tC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA,
                                                  sA, tA, B, dB, sB, tB, C, dC,
                                                  sC, tC, alpha, beta);
}

// Setup params for a TN GEMM
// Use padded m-major smem sA, padded n-major smem sB, and k-major threads tA|tB
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
    auto sA = make_layout(make_shape(bM, bK),
                          LayoutRight{}); // (m,k) -> smem_idx; k-major
    auto sB = make_layout(make_shape(bN, bK),
                          LayoutRight{});      // (n,k) -> smem_idx; k-major
    auto sC = make_layout(make_shape(bM, bN)); // (m,n) -> smem_idx; m-major

    // Define the thread layouts (static)
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}),
                          LayoutRight{}); // (m,k) -> thr_idx; k-major
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}),
                          LayoutRight{}); // (n,k) -> thr_idx; k-major
    auto tC = make_layout(
        make_shape(Int<16>{}, Int<16>{})); // (m,n) -> thr_idx; m-major

    dim3 dimBlock(size(tC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA,
                                                  sA, tA, B, dB, sB, tB, C, dC,
                                                  sC, tC, alpha, beta);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
cudaError_t launch_sgemm_1(char transA, char transB, int m, int n, int k,
                           Alpha alpha, TA const* A, int ldA, TB const* B,
                           int ldB, Beta beta, TC* C, int ldC,
                           cudaStream_t stream = 0)
{
    if (transA == 'N' && transB == 'T')
    {
        gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    else if (transA == 'T' && transB == 'N')
    {
        gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    else
    {
        assert(false && "Not implemented");
    }

    // auto test_layout = cute::make_layout(cute::make_shape(cute::Int<8>{},
    //                                                       cute::Int<8>{}),
    //                                                       cute::make_stride(cute::Int<16>{},
    //                                                       cute::Int<16384>{}));
    // auto test_layout_like = cute::make_layout_like(test_layout);
    // auto test_tensor = cute::make_tensor(cute::make_gmem_ptr(A),
    // test_layout); auto test_tensor_like =
    // cute::make_tensor_like(test_tensor); std::cout << "test_layout: ";
    // cute::print(test_layout);
    // std::cout << std::endl;
    // std::cout << "test_layout_like: ";
    // cute::print(test_layout_like);
    // std::cout << std::endl;
    // std::cout << "test_tensor: ";
    // cute::print(test_tensor);
    // std::cout << std::endl;
    // std::cout << "test_tensor_like: ";
    // cute::print(test_tensor_like);
    // std::cout << std::endl;

    return cudaGetLastError();
}

// Explicit instantiation
template cudaError_t launch_sgemm_1<float, float, float, float, float>(
    char transA, char transB, int m, int n, int k, float alpha, float const* A,
    int ldA, float const* B, int ldB, float beta, float* C, int ldC,
    cudaStream_t stream);
template cudaError_t launch_sgemm_1<double, double, double, double, double>(
    char transA, char transB, int m, int n, int k, double alpha,
    double const* A, int ldA, double const* B, int ldB, double beta, double* C,
    int ldC, cudaStream_t stream);
template cudaError_t
launch_sgemm_1<cute::half_t, cute::half_t, cute::half_t, float, float>(
    char transA, char transB, int m, int n, int k, float alpha,
    cute::half_t const* A, int ldA, cute::half_t const* B, int ldB, float beta,
    cute::half_t* C, int ldC, cudaStream_t stream);
template cudaError_t
launch_sgemm_1<cute::half_t, cute::half_t, cute::half_t, cute::half_t,
               cute::half_t>(char transA, char transB, int m, int n, int k,
                             cute::half_t alpha, cute::half_t const* A, int ldA,
                             cute::half_t const* B, int ldB, cute::half_t beta,
                             cute::half_t* C, int ldC, cudaStream_t stream);

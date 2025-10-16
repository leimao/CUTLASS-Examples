#include <iostream>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/numeric_types.h"
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

// #include "cuda_launch.hpp"
// #include "shared_storage.h"
// #include "smem_helper.hpp"

// template <typename T, int CTA_M, int CTA_N, class TmaLoad, class GmemTensor>
// __global__ void tma_load_kernel(__grid_constant__ const TmaLoad tma_load,
// GmemTensor gmem_tensor) {
//   using namespace cute;
//   constexpr int tma_transaction_bytes = CTA_M * CTA_N * sizeof(T);

//   __shared__ T smem_data[CTA_M * CTA_N];
//   __shared__ uint64_t tma_load_mbar;

//   auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
//   auto smem_tensor = make_tensor(make_smem_ptr(smem_data), smem_layout);

//   if (threadIdx.x == 0) {
//     auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));

//     auto gmem_tensor_coord_cta = local_tile(
//         gmem_tensor_coord,
//         Tile<Int<CTA_M>, Int<CTA_N>>{},
//         make_coord(blockIdx.x, blockIdx.y));

//     initialize_barrier(tma_load_mbar, /* arrival count */ 1);

//     set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);

//     auto tma_load_per_cta = tma_load.get_slice(0);
//     copy(tma_load.with(tma_load_mbar),
//          tma_load_per_cta.partition_S(gmem_tensor_coord_cta),
//          tma_load_per_cta.partition_D(smem_tensor));
//   }
//   __syncthreads();
//   wait_barrier(tma_load_mbar, /* phase */ 0);

//   // after this line, the TMA load is finished
// }

template <typename T, int CTA_M, int CTA_N>
void host_fn(T* data, int M, int N)
{
    using namespace cute;

    // create the GMEM tensor
    auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
    auto gmem_tensor = make_tensor(make_gmem_ptr(data), gmem_layout);

    // create the SMEM layout
    auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});

    // create the TMA object
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);

    //   // invoke the kernel
    //   tma_load_kernel<CTA_M, CTA_N>
    //                  <<<dim3{M / CTA_M, N / CTA_N, 1}, 1>>>
    //                  (tma_load, gmem_tensor, smem_layout);
}

int main()
{
    using namespace cute;

    printf("Copy with TMA load and store -- no swizzling.\n");

    using Element = float;

    int M = 16384;
    int N = 16384;

    auto tensor_shape = make_shape(M, N);

    // Allocate and initialize
    thrust::host_vector<Element> h_S(size(tensor_shape)); // (M, N)
    thrust::host_vector<Element> h_D(size(tensor_shape)); // (M, N)

    host_fn<Element, 32, 32>(h_S.data(), M, N);
    return 0;
}
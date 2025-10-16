#include <cuda_runtime.h>

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/container/array_aligned.hpp>
#include <cute/tensor.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/detail/layout.hpp>
#include <cutlass/fast_math.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

cudaError_t set_smem_size(void const* kernel_func, int smem_size)
{
    // 48 KB
    if (smem_size >= 48 << 10)
    {
        return cudaFuncSetAttribute(kernel_func,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);
    }
    else
    {
        return cudaSuccess;
    }
}

template <class DataType, class SmemLayout>
struct SharedStorageTMA
{
    cute::array_aligned<DataType, cute::cosize_v<SmemLayout>,
                        cutlass::detail::alignment_for_swizzle(SmemLayout{})>
        smem;
    // CUTLASS barrier wrapper.
    cutlass::arch::ClusterTransactionBarrier mbarrier;
};

template <class DataType, class TMACopyS, class TMACopyD, class GmemLayout,
          class SmemLayout, class TileShape>
__global__ static void matrix_copy(CUTE_GRID_CONSTANT TMACopyS tma_load,
                                   CUTE_GRID_CONSTANT TMACopyD tma_store,
                                   GmemLayout gmem_layout,
                                   SmemLayout smem_layout, TileShape tile_shape)
{
    // This is supposed to be 0 for all thread blocks, since we only have one
    // cluster.
    constexpr uint32_t cluster_size{1U};
    constexpr uint16_t tma_mcast_mask{
        (static_cast<uint16_t>(1U) << cluster_size) - 1};

    // Shared memory storage
    extern __shared__ uint8_t smem[];
    auto& smem_storage{
        *reinterpret_cast<SharedStorageTMA<DataType, SmemLayout>*>(smem)};

    // Create shared memory tensor.
    auto smem_tensor{cute::make_tensor(
        cute::make_smem_ptr(smem_storage.smem.data()), smem_layout)};

    // Get CUTLASS barrier wrapper object.
    auto& mbarrier{smem_storage.mbarrier};
    using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
    constexpr int tma_transaction_bytes{
        sizeof(cute::ArrayEngine<DataType, size(SmemLayout{})>)};

    // Ensure only one thread from a thread block will execute the TMA
    // operation.
    int const warp_idx{cutlass::canonical_warp_idx_sync()};
    bool const lane_predicate{static_cast<bool>(cute::elect_one_sync())};

    // Prefetch TMA descriptors for load and store from global memory to cache.
    // This is an optimization for performance.
    // https://pytorch.org/blog/hopper-tma-unit/
    if (warp_idx == 0 && lane_predicate)
    {
        cute::prefetch_tma_descriptor(tma_load.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_store.get_tma_descriptor());
    }

    // TMA tensor shape can be smaller than the tensor shape, which might be
    // useful when some data on the global memory is not needed to be copied. In
    // our case, we want to copy the whole tensor. The coordinates are 2D and it
    // will result in using the TMA 2D copy operation.
    // Even though it produces an implicit tensor that consists of many
    // coordinates, Only a small fraction of them will be used as the starting
    // coordinate for TMA copy PTX instruction, such as
    // cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
    cute::Tensor const global_tma_coord_src_tensor{
        tma_load.get_tma_tensor(cute::shape(gmem_layout))};
    auto const block_coord_src{cute::make_coord(blockIdx.x, blockIdx.y)};
    auto const block_tma_coord_src_tensor{cute::local_tile(
        global_tma_coord_src_tensor, tile_shape, block_coord_src)};

    cute::Tensor const global_tma_coord_dst_tensor{
        tma_store.get_tma_tensor(cute::shape(gmem_layout))};
    auto const block_coord_dst{cute::make_coord(blockIdx.x, blockIdx.y)};
    auto const block_tma_coord_dst_tensor{cute::local_tile(
        global_tma_coord_dst_tensor, tile_shape, block_coord_dst)};

    // This borrows the TiledCopy interface to get the ThreadCopy object for
    // thread 0. Because the TMA TiledCopy object has the "number of threads"
    // equal to the multicast size, which is 1 in our case, we will always get
    // the ThreadCopy object for thread 0.
    auto block_tma_load{tma_load.get_slice(0)};
    // Depending on the layouts, multiple TMA instructions might be issued in
    // cute::copy to complete the copy of the whole tile.
    auto const partitioned_block_tma_coord_src_tensor{
        block_tma_load.partition_S(block_tma_coord_src_tensor)};
    auto partitioned_smem_src_tensor{block_tma_load.partition_D(smem_tensor)};

    auto block_tma_store{tma_store.get_slice(0)};
    auto const partitioned_smem_dst_tensor{
        block_tma_store.partition_S(smem_tensor)};
    auto partitioned_block_tma_coord_dst_tensor{
        block_tma_store.partition_D(block_tma_coord_dst_tensor)};

    // Perform TMA load from global memory to shared memory.
    // Only one thread in a thread block will perform the TMA operation.
    if (warp_idx == 0 && lane_predicate)
    {
        // Initialize the barrier for TMA load.
        // The arrival count is 1, since each thread block only performs one TMA
        // load.
        mbarrier.init(1);
        // Set the expected transaction bytes for this barrier.
        mbarrier.arrive_and_expect_tx(tma_transaction_bytes);
        // TMA load does not support the
        // cutlass::arch::ClusterTransactionBarrier type. Need to cast it to its
        // underlying integer type.
        cute::copy(tma_load.with(reinterpret_cast<BarrierType&>(mbarrier),
                                 tma_mcast_mask),
                   partitioned_block_tma_coord_src_tensor,
                   partitioned_smem_src_tensor);
    }

    // Ensure mbarrier is initialized correctly before all threads start to wait
    // on it. Otherwise, some threads might wait on an uninitialized barrier,
    // which can lead to undefined behavior.
    __syncthreads();
    // 0 is the phase bit for the first TMA operation.
    // A phase bit will be flipped after all arriving threads arrive at the
    // barrier. 0 -> 1 - > 0 - > 1 ... Wait until this barrier completes its
    // phase 0 synchronization
    mbarrier.wait(0);
    // Accessing the same memory location across multiple proxies needs a
    // cross-proxy fence. For the async proxy, fence.proxy.async should be used
    // to synchronize memory between generic proxy and the async proxy.
    cutlass::arch::fence_view_async_shared();

    // Data becomes visible to all threads in the generic proxy.
    // We could perform operations on the data in the shared memory here.

    // Perform TMA store from shared memory to global memory.
    if (warp_idx == 0 && lane_predicate)
    {
        cute::copy(tma_store, partitioned_smem_dst_tensor,
                   partitioned_block_tma_coord_dst_tensor);
        cute::tma_store_arrive();
    }
    cute::tma_store_wait<0>();
}

// Assuming row-major matrix first.
template <class DataType>
static cudaError_t launch_matrix_copy(DataType const* input_matrix,
                                      DataType* output_matrix, unsigned int m,
                                      unsigned int n, cudaStream_t stream)
{
    auto const tensor_shape{
        cute::make_shape(static_cast<int>(m), static_cast<int>(n))};
    // Row-major global memory layout
    // auto const global_memory_layout{
    //     cute::make_layout(tensor_shape, cute::LayoutRight{})};
    auto const global_memory_layout{
        cute::make_layout(tensor_shape, cute::LayoutLeft{})};

    auto const tensor_src{cute::make_tensor(cute::make_gmem_ptr(input_matrix),
                                            global_memory_layout)};
    auto const tensor_dst{cute::make_tensor(cute::make_gmem_ptr(output_matrix),
                                            global_memory_layout)};

    constexpr auto TILE_M{128};
    constexpr auto TILE_N{128};

    using bM = cute::Int<TILE_M>;
    using bN = cute::Int<TILE_N>;

    constexpr auto tile_shape{cute::make_shape(bM{}, bN{})};
    // Swizzle is especially useful for matrix transpose.
    // Not useful for simple identity copy.
    // Cannot use arbitrary swizzled layout for TMA, because TMA only supports
    // certain swizzles. Row-major swizzled layout for TMA
    // constexpr auto swizzled_atom_tile{
    //     cute::SM90::GMMA::Layout_K_SW128_Atom<DataType>{}};
    // Column-major swizzled layout for TMA
    constexpr auto swizzled_atom_tile{
        cute::SM90::GMMA::Layout_MN_SW128_Atom<DataType>{}};
    constexpr auto shared_memory_layout{
        cute::tile_to_shape(swizzled_atom_tile, tile_shape)};

    // The cluster size is 1 and there is no multicast.
    // TMA copy saves the pointer to the global memory tensor, global memory
    // layout, shared memory layout, tile shape per block, and multicast size.
    // Because TMA copy already has the pointer to the global memory tensor, and
    // we will perform TMA copy between global memory and shared memory, passing
    // input/output tensors or pointers to the data to the kernel is not useful
    // anymore.
    auto const tma_load{cute::make_tma_copy(cute::SM90_TMA_LOAD{}, tensor_src,
                                            shared_memory_layout, tile_shape,
                                            cute::Int<1>{})};
    auto const tma_store{cute::make_tma_copy(cute::SM90_TMA_STORE{}, tensor_dst,
                                             shared_memory_layout, tile_shape,
                                             cute::Int<1>{})};

    constexpr auto thread_layout{
        cute::make_shape(cute::Int<32>{}, cute::Int<8>{})};
    constexpr auto NUM_THREADS{cute::size(thread_layout)};

    dim3 const thread_dim{NUM_THREADS};
    dim3 const grid_dim{static_cast<unsigned int>(cutlass::ceil_div(
                            cute::get<0>(tensor_shape), TILE_M)),
                        static_cast<unsigned int>(cutlass::ceil_div(
                            cute::get<1>(tensor_shape), TILE_N))};
    dim3 const cluster_dim{1, 1, 1};

    // Configure shared memory
    constexpr int shared_memory_size{
        sizeof(SharedStorageTMA<DataType, decltype(shared_memory_layout)>)};
    // TODO: Check the size of shared memory.

    // Define the kernel function pointer type
    using KernelFunc = void (*)(
        decltype(tma_load), decltype(tma_store), decltype(global_memory_layout),
        decltype(shared_memory_layout), decltype(tile_shape));

    void const* kernel_func = reinterpret_cast<void const*>(
        matrix_copy<DataType, decltype(tma_load), decltype(tma_store),
                    decltype(global_memory_layout),
                    decltype(shared_memory_layout), decltype(tile_shape)>);
    // In practice, the shared memory size for a certain kernel should only be
    // set once to the maximum shared memory size possible at the initialization
    // stage so that there are no complications during the runtime where the
    // same kernel is launched with different configurations.
    cudaError_t smem_set_status{set_smem_size(kernel_func, shared_memory_size)};
    if (smem_set_status != cudaSuccess)
    {
        return smem_set_status;
    }

    cutlass::ClusterLaunchParams launch_params{.grid_dims = grid_dim,
                                               .block_dims = thread_dim,
                                               .cluster_dims = cluster_dim,
                                               .smem_size_in_bytes =
                                                   shared_memory_size,
                                               .cuda_stream = stream};

    cutlass::Status launch_status{cutlass::launch_kernel_on_cluster(
        launch_params, kernel_func, tma_load, tma_store, global_memory_layout,
        shared_memory_layout, tile_shape)};
    if (launch_status != cutlass::Status::kSuccess)
    {
        return cudaErrorLaunchFailure;
    }

    return cudaSuccess;
}

// // Explicit instantiation.
// template cudaError_t launch_vector_copy<float>(float const* input_vector,
//                                                float* output_vector,
//                                                unsigned int size,
//                                                cudaStream_t stream);
// template cudaError_t launch_vector_copy<double>(double const* input_vector,
//                                                 double* output_vector,
//                                                 unsigned int size,
//                                                 cudaStream_t stream);

int main(int argc, char** argv)
{
    using DataType = float;

    unsigned int m = 1024;
    unsigned int n = 1024;

    // Allocate and initialize
    thrust::host_vector<DataType> h_S(m * n); // (M, N)
    thrust::host_vector<DataType> h_D(m * n); // (M, N)

    for (int i = 0; i < h_S.size(); ++i)
    {
        h_S[i] = static_cast<DataType>(i);
    }
    for (int i = 0; i < h_D.size(); ++i)
    {
        h_D[i] = static_cast<DataType>(0);
    }
    thrust::device_vector<DataType> d_S = h_S;
    thrust::device_vector<DataType> d_D = h_D;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaError_t status{
        launch_matrix_copy(d_S.data().get(), d_D.data().get(), m, n, stream)};
    if (status != cudaSuccess)
    {
        std::cerr << "Matrix copy launch failed: " << cudaGetErrorString(status)
                  << std::endl;
        return -1;
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Verify the values are correct
    h_D = d_D;
    for (int i = 0; i < h_D.size(); ++i)
    {
        if (h_D[i] != h_S[i])
        {
            std::cerr << "Matrix copy error at index " << i << ": " << h_D[i]
                      << " != " << h_S[i] << std::endl;
            return -1;
        }
    }
    return 0;
}
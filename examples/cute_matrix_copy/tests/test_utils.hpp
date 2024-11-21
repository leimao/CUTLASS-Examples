#include <iostream>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const* file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Copy a M x N row-major matrix.
template <class T>
void copy(T const* src, T* dst, unsigned int M, unsigned int N)
{
    for (unsigned int i{0}; i < M; ++i)
    {
        for (unsigned int j{0}; j < N; ++j)
        {
            dst[i * N + j] = src[i * N + j];
        }
    }
}

// Initialize a data array.
template <class T>
void initialize(T* data, unsigned int size)
{
    for (unsigned int i{0}; i < size; ++i)
    {
        data[i] = static_cast<T>(i);
    }
}

// Print a data array and a reference array.
template <class T>
void print(T const* data, T const* ref, unsigned int size)
{
    for (unsigned int i{0}; i < size; ++i)
    {
        std::cout << i << " " << data[i] << " " << ref[i] << std::endl;
    }
}

// Compare a data array with a reference array.
template <class T>
bool compare(T const* data, T const* ref, unsigned int size)
{
    bool status{true};
    for (unsigned int i{0}; i < size; ++i)
    {
        if (data[i] != ref[i])
        {
            status = false;
        }
    }

    if (!status)
    {
        print<T>(data, ref, size);
    }

    return status;
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> const& bound_function,
                          cudaStream_t stream, unsigned int num_repeats = 20,
                          unsigned int num_warmups = 10)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (unsigned int i{0U}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (unsigned int i{0U}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

template <class T>
float convert_latency_to_effective_bandwidth(float latency, unsigned int M,
                                             unsigned int N)
{
    size_t const size{M * N * sizeof(T) * 2};
    float const bandwidth{size / (latency / 1.0e3f) / (1 << 30)};
    return bandwidth;
}

template <typename T>
class TestMatrixTranspose
    : public ::testing::TestWithParam<std::tuple<unsigned int, unsigned int>>
{
protected:
    void SetUp() override
    {
        // Create CUDA stream.
        CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));

        // Get paramater.
        m_M = std::get<0>(GetParam());
        m_N = std::get<1>(GetParam());

        // Use thrust to create the host and device vectors.
        m_h_src = thrust::host_vector<T>(m_M * m_N);
        m_h_dst = thrust::host_vector<T>(m_N * m_M);
        m_h_dst_ref = thrust::host_vector<T>(m_N * m_M);

        m_d_src = thrust::device_vector<T>(m_M * m_N);
        m_d_dst = thrust::device_vector<T>(m_N * m_M);

        // Initialize the host vectors.
        initialize(m_h_src.data(), m_h_src.size());
        copy(m_h_src.data(), m_h_dst_ref.data(), m_M, m_N);

        // Copy the host vectors to the device vectors.
        m_d_src = m_h_src;
    }

    void TearDown() override
    {
        // Destroy CUDA stream.
        CHECK_CUDA_ERROR(cudaStreamDestroy(m_stream));
    }

    void RunTest(cudaError_t (*launch_copy)(T const*, T*, unsigned int,
                                                 unsigned int, cudaStream_t))
    {
        // Launch the kernel.
        CHECK_CUDA_ERROR(launch_copy(
            thrust::raw_pointer_cast(m_d_src.data()),
            thrust::raw_pointer_cast(m_d_dst.data()), m_M, m_N, m_stream));

        // Synchronize the stream.
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));

        // Copy the data from device to host.
        m_h_dst = m_d_dst;

        // Compare the data.
        ASSERT_TRUE(
            compare(m_h_dst.data(), m_h_dst_ref.data(), m_h_dst.size()));
    }

    void MeasurePerformance(
        cudaError_t (*launch_copy)(T const*, T*, unsigned int,
                                        unsigned int, cudaStream_t),
        unsigned int num_repeats = 20, unsigned int num_warmups = 20)
    {
        GTEST_COUT << "M: " << m_M << " N: " << m_N << std::endl;

        // Query deive name and peak memory bandwidth.
        int device_id{0};
        CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
        cudaDeviceProp device_prop;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, device_id));
        GTEST_COUT << "Device Name: " << device_prop.name << std::endl;
        float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                                (1 << 30)};
        GTEST_COUT << "Memory Size: " << memory_size << " GB" << std::endl;
        float const peak_bandwidth{
            static_cast<float>(2.0f * device_prop.memoryClockRate *
                               (device_prop.memoryBusWidth / 8) / 1.0e6)};
        GTEST_COUT << "Peak Bandwitdh: " << peak_bandwidth << " GB/s"
                   << std::endl;

        auto const function{std::bind(launch_copy,
                                      thrust::raw_pointer_cast(m_d_src.data()),
                                      thrust::raw_pointer_cast(m_d_dst.data()),
                                      m_M, m_N, std::placeholders::_1)};
        float const latency{measure_performance<T>(function, m_stream)};
        GTEST_COUT << "Latency: " << latency << " ms" << std::endl;
        GTEST_COUT << "Effective Bandwidth: "
                   << convert_latency_to_effective_bandwidth<T>(latency, m_M,
                                                                m_N)
                   << " GB/s" << std::endl;
        GTEST_COUT << "Peak Bandwidth Percentage: "
                   << 100.0f *
                          convert_latency_to_effective_bandwidth<T>(latency,
                                                                    m_M, m_N) /
                          peak_bandwidth
                   << "%" << std::endl;
    }

    unsigned int m_M;
    unsigned int m_N;

    cudaStream_t m_stream;

    thrust::host_vector<T> m_h_src;
    thrust::host_vector<T> m_h_dst;
    thrust::host_vector<T> m_h_dst_ref;

    thrust::device_vector<T> m_d_src;
    thrust::device_vector<T> m_d_dst;
};

class TestMatrixTransposeFloat : public TestMatrixTranspose<float>
{
};

class TestMatrixTransposeDouble : public TestMatrixTranspose<double>
{
};
#ifndef CUTE_GENERAL_MATRIX_MULTIPLICATION_TEST_UTILS_HPP
#define CUTE_GENERAL_MATRIX_MULTIPLICATION_TEST_UTILS_HPP

#include <iostream>
#include <random>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cutlass/half.h>

#include <cublas_v2.h>

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
void checkLast(char const* file, int line)
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

#define CHECK_CUBLASS_ERROR(val) check_cublass((val), #val, __FILE__, __LINE__)
void check_cublass(cublasStatus_t err, char const* const func,
                   char const* const file, int line)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Error at: " << file << ":" << line << std::endl;
        std::cerr << cublasGetStatusString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// The original matrix is column major.
// coordinate (i, j) is from the transposed matrix if there is any.
int coordinate_to_index(int i, int j, int ld, bool trans)
{
    if (trans)
    {
        return j + i * ld;
    }
    else
    {
        return i + j * ld;
    }
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_cpu(char transa, char transb, int m, int n, int k, Alpha alpha,
              TA const* A, int lda, TB const* B, int ldb, Beta beta, TC* C,
              int ldc)
{
    bool const transa_bool{transa == 'T' || transa == 't'};
    bool const transb_bool{transb == 'T' || transb == 't'};

    for (int i{0}; i < m; ++i)
    {
        for (int j{0}; j < n; ++j)
        {
            TC sum{0};
            for (int l{0}; l < k; ++l)
            {
                int const idx_a{
                    coordinate_to_index(i, l, lda, transa_bool)}; // A[i, l]
                int const idx_b{
                    coordinate_to_index(l, j, ldb, transb_bool)}; // B[l, j]
                TA const a{A[idx_a]};
                TB const b{B[idx_b]};
                sum += a * b;
            }
            int const idx_c{coordinate_to_index(i, j, ldc, false)}; // C[i, j]
            TC const c{C[idx_c]};
            C[idx_c] = alpha * sum + beta * c;
        }
    }
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, cutlass::half_t>::value,
                                  bool>::type = true>
constexpr cudaDataType_t cuda_data_type_trait()
{
    if (std::is_same<T, float>::value)
    {
        return CUDA_R_32F;
    }
    else if (std::is_same<T, double>::value)
    {
        return CUDA_R_64F;
    }
    else if (std::is_same<T, cutlass::half_t>::value)
    {
        return CUDA_R_16F;
    }
    else
    {
        throw std::runtime_error("Unsupported data type.");
    }
}

cublasComputeType_t get_compute_type(cudaDataType_t data_type_alpha,
                                     cudaDataType data_type_beta,
                                     cudaDataType data_type_a,
                                     cudaDataType data_type_b,
                                     cudaDataType data_type_c)
{
    if (data_type_alpha == CUDA_R_32F && data_type_beta == CUDA_R_32F &&
        data_type_a == CUDA_R_32F && data_type_b == CUDA_R_32F &&
        data_type_c == CUDA_R_32F)
    {
        return CUBLAS_COMPUTE_32F;
    }
    else if (data_type_alpha == CUDA_R_64F && data_type_beta == CUDA_R_64F &&
             data_type_a == CUDA_R_64F && data_type_b == CUDA_R_64F &&
             data_type_c == CUDA_R_64F)
    {
        return CUBLAS_COMPUTE_64F;
    }
    else if (data_type_alpha == CUDA_R_32F && data_type_beta == CUDA_R_32F &&
             data_type_a == CUDA_R_16F && data_type_b == CUDA_R_16F &&
             data_type_c == CUDA_R_16F)
    {
        return CUBLAS_COMPUTE_32F;
    }
    else if (data_type_alpha == CUDA_R_16F && data_type_beta == CUDA_R_16F &&
             data_type_a == CUDA_R_16F && data_type_b == CUDA_R_16F &&
             data_type_c == CUDA_R_16F)
    {
        return CUBLAS_COMPUTE_16F;
    }
    else
    {
        throw std::runtime_error("Unsupported compute type.");
    }
}

template <class TA, class TB, class TC, class Alpha, class Beta>
cublasStatus_t gemm_cublas(char transa, char transb, int m, int n, int k,
                           Alpha alpha, TA const* A, int lda, TB const* B,
                           int ldb, Beta beta, TC* C, int ldc,
                           cublasHandle_t handle, cudaStream_t stream)
{
    // Set CUDA stream.
    CHECK_CUBLASS_ERROR(cublasSetStream(handle, stream));
    cudaDataType_t const data_type_alpha{cuda_data_type_trait<Alpha>()};
    cudaDataType_t const data_type_beta{cuda_data_type_trait<Beta>()};
    cudaDataType_t const data_type_a{cuda_data_type_trait<TA>()};
    cudaDataType_t const data_type_b{cuda_data_type_trait<TB>()};
    cudaDataType_t const data_type_c{cuda_data_type_trait<TC>()};
    cublasComputeType_t const compute_type{
        get_compute_type(data_type_alpha, data_type_beta, data_type_a,
                         data_type_b, data_type_c)};
    cublasGemmAlgo_t const algo{CUBLAS_GEMM_DEFAULT};
    cublasOperation_t const transa_cublas{
        transa == 'T' || transa == 't' ? CUBLAS_OP_T : CUBLAS_OP_N};
    cublasOperation_t const transb_cublas{
        transb == 'T' || transb == 't' ? CUBLAS_OP_T : CUBLAS_OP_N};
    cublasStatus_t const status{
        cublasGemmEx(handle, transa_cublas, transb_cublas, m, n, k, &alpha, A,
                     data_type_a, lda, B, data_type_b, ldb, &beta, C,
                     data_type_c, ldc, compute_type, algo)};
    return status;
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

template <class T>
void random_initialize(T* data, unsigned int size, T a, T b,
                       std::default_random_engine& random_engine)
{
    std::uniform_int_distribution<int> distribution(static_cast<int>(a),
                                                    static_cast<int>(b));
    for (unsigned int i{0}; i < size; ++i)
    {
        data[i] = static_cast<T>(distribution(random_engine));
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

float convert_latency_to_tops(float latency, int m, int n, int k)
{
    size_t const num_operations{2 * static_cast<size_t>(m) *
                                static_cast<size_t>(n) *
                                static_cast<size_t>(k)};
    float const tops{num_operations / (latency / 1.0e3f) / 1.0e12f};
    return tops;
}

template <class TA, class TB, class TC, class Alpha, class Beta>
class TestGeneralMatrixMultiplication
    : public ::testing::TestWithParam<
          std::tuple<char, char, int, int, int, int, int, int>>
{
protected:
    void SetUp() override
    {
        // Set random seed.
        m_seed = 0;
        m_random_engine = std::default_random_engine{m_seed};

        TA a_lower_bound{static_cast<TA>(-2.0)};
        TA a_upper_bound{static_cast<TA>(2.0)};
        TB b_lower_bound{static_cast<TB>(-2.0)};
        TB b_upper_bound{static_cast<TB>(2.0)};

        // Create CUDA stream.
        CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));
        CHECK_CUBLASS_ERROR(cublasCreate(&m_handle));

        // Get parameter.
        m_transa = std::get<0>(GetParam());
        m_transb = std::get<1>(GetParam());

        // Check if the char is valid.
        ASSERT_TRUE(m_transa == 'N' || m_transa == 'n' || m_transa == 'T' ||
                    m_transa == 't');
        ASSERT_TRUE(m_transb == 'N' || m_transb == 'n' || m_transb == 'T' ||
                    m_transb == 't');

        m_m = std::get<2>(GetParam());
        m_n = std::get<3>(GetParam());
        m_k = std::get<4>(GetParam());

        m_lda = std::get<5>(GetParam());
        m_ldb = std::get<6>(GetParam());
        m_ldc = std::get<7>(GetParam());

        // Compute matrix size.
        // All matrices are column major.
        // op(A) is m x k.
        // op(B) is k x n.
        // C is m x n.
        int const num_ldas{m_transa == 'N' || m_transa == 'n' ? m_k : m_m};
        int const num_ldbs{m_transb == 'N' || m_transb == 'n' ? m_n : m_k};
        int const num_ldcs{m_n};
        int const size_a{num_ldas * m_lda};
        int const size_b{num_ldbs * m_ldb};
        int const size_c{num_ldcs * m_ldc};

        // Use thrust to create the host and device vectors.
        m_h_A = thrust::host_vector<TA>(size_a);
        m_h_B = thrust::host_vector<TB>(size_b);
        m_h_C = thrust::host_vector<TC>(size_c);
        m_h_C_ref = thrust::host_vector<TC>(size_c);

        m_d_A = thrust::device_vector<TA>(size_a);
        m_d_B = thrust::device_vector<TB>(size_b);
        m_d_C = thrust::device_vector<TC>(size_c);

        // Initialize coefficients.
        m_alpha = static_cast<Alpha>(1.0);
        m_beta = static_cast<Beta>(0.0);

        // Initialize the host vectors.
        random_initialize(m_h_A.data(), m_h_A.size(), a_lower_bound,
                          a_upper_bound, m_random_engine);
        random_initialize(m_h_B.data(), m_h_B.size(), b_lower_bound,
                          b_upper_bound, m_random_engine);

        // Copy the host vectors to the device vectors.
        m_d_A = m_h_A;
        m_d_B = m_h_B;

        // Compute the reference result.
        // Very slow on CPU for large problem sizes.
        // gemm_cpu(m_transa, m_transb, m_m, m_n, m_k, m_alpha, m_h_A.data(),
        //          m_lda, m_h_B.data(), m_ldb, m_beta, m_h_C_ref.data(),
        //          m_ldc);

        // Measure cuBLAS latency and compute efficiency.
        CHECK_CUBLASS_ERROR(gemm_cublas(
            m_transa, m_transb, m_m, m_n, m_k, m_alpha,
            thrust::raw_pointer_cast(m_d_A.data()), m_lda,
            thrust::raw_pointer_cast(m_d_B.data()), m_ldb, m_beta,
            thrust::raw_pointer_cast(m_d_C.data()), m_ldc, m_handle, m_stream));
        // Synchronize the stream.
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));
        // Copy the data from device to host.
        m_h_C_ref = m_d_C;

        // Clean up the data on device.
        thrust::fill(m_d_C.begin(), m_d_C.end(), static_cast<TC>(0));
    }

    void TearDown() override
    {
        // Destroy cuBLAS handle.
        CHECK_CUBLASS_ERROR(cublasDestroy(m_handle));
        // Destroy CUDA stream.
        CHECK_CUDA_ERROR(cudaStreamDestroy(m_stream));
    }

    void RunTest(cudaError_t (*launch_gemm)(char, char, int, int, int, Alpha,
                                            TA const*, int, TB const*, int,
                                            Beta, TC*, int, cudaStream_t))
    {
        // Clean up the data on device.
        thrust::fill(m_d_C.begin(), m_d_C.end(), static_cast<TC>(0));

        // Launch the kernel.
        CHECK_CUDA_ERROR(launch_gemm(
            m_transa, m_transb, m_m, m_n, m_k, m_alpha,
            thrust::raw_pointer_cast(m_d_A.data()), m_lda,
            thrust::raw_pointer_cast(m_d_B.data()), m_ldb, m_beta,
            thrust::raw_pointer_cast(m_d_C.data()), m_ldc, m_stream));

        // Synchronize the stream.
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));

        // Copy the data from device to host.
        m_h_C = m_d_C;

        // Compare the data.
        ASSERT_TRUE(compare(m_h_C.data(), m_h_C_ref.data(), m_h_C.size()));
    }

    void MeasurePerformance(
        cudaError_t (*launch_gemm)(char, char, int, int, int, Alpha, TA const*,
                                   int, TB const*, int, Beta, TC*, int,
                                   cudaStream_t),
        unsigned int num_repeats = 10, unsigned int num_warmups = 10)
    {
        GTEST_COUT << "transa: " << m_transa << " transb: " << m_transb
                   << std::endl;
        GTEST_COUT << "m: " << m_m << " n: " << m_n << " k: " << m_k
                   << std::endl;
        GTEST_COUT << "lda: " << m_lda << " ldb: " << m_ldb << " ldc: " << m_ldc
                   << std::endl;

        // Query deive name and peak memory bandwidth.
        int device_id{0};
        CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
        cudaDeviceProp device_prop;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, device_id));
        GTEST_COUT << "Device Name: " << device_prop.name << std::endl;
        float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                                (1 << 30)};

        auto const custom_kernel_function{
            std::bind(launch_gemm, m_transa, m_transb, m_m, m_n, m_k, m_alpha,
                      thrust::raw_pointer_cast(m_d_A.data()), m_lda,
                      thrust::raw_pointer_cast(m_d_B.data()), m_ldb, m_beta,
                      thrust::raw_pointer_cast(m_d_C.data()), m_ldc,
                      std::placeholders::_1)};
        std::function<cudaError_t(cudaStream_t)> bound_custom_kernel_function{
            custom_kernel_function};
        float const custom_kernel_latency{measure_performance(
            bound_custom_kernel_function, m_stream, num_repeats, num_warmups)};

        // Make sure cuBLAS does not produce errors.
        CHECK_CUBLASS_ERROR(gemm_cublas(
            m_transa, m_transb, m_m, m_n, m_k, m_alpha,
            thrust::raw_pointer_cast(m_d_A.data()), m_lda,
            thrust::raw_pointer_cast(m_d_B.data()), m_ldb, m_beta,
            thrust::raw_pointer_cast(m_d_C.data()), m_ldc, m_handle, m_stream));
        // Synchronize the stream.
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));

        auto const cublas_function{std::bind(
            gemm_cublas<TA, TB, TC, Alpha, Beta>, m_transa, m_transb, m_m, m_n,
            m_k, m_alpha, thrust::raw_pointer_cast(m_d_A.data()), m_lda,
            thrust::raw_pointer_cast(m_d_B.data()), m_ldb, m_beta,
            thrust::raw_pointer_cast(m_d_C.data()), m_ldc, m_handle, m_stream)};
        std::function<cublasStatus_t(cudaStream_t)> bound_cublas_function{
            cublas_function};
        float const cublas_latency{measure_performance(
            bound_cublas_function, m_stream, num_repeats, num_warmups)};

        GTEST_COUT << "cuBLAS Latency: " << cublas_latency << " ms"
                   << std::endl;
        GTEST_COUT << "cuBLAS Compute Efficiency: "
                   << convert_latency_to_tops(cublas_latency, m_m, m_n, m_k)
                   << " TOPS" << std::endl;

        GTEST_COUT << "Custom Kernel Latency: " << custom_kernel_latency
                   << " ms" << std::endl;
        GTEST_COUT << "Custom Kernel Compute Efficiency: "
                   << convert_latency_to_tops(custom_kernel_latency, m_m, m_n,
                                              m_k)
                   << " TOPS" << std::endl;
        // Use percentage.
        GTEST_COUT << "Custom Kernel Performance VS cuBLAS Performance: "
                   << cublas_latency / custom_kernel_latency * 100.0f << "%"
                   << std::endl;
    }

    char m_transa;
    char m_transb;

    int m_m;
    int m_n;
    int m_k;

    int m_lda;
    int m_ldb;
    int m_ldc;

    Alpha m_alpha;
    Beta m_beta;

    cudaStream_t m_stream;
    cublasHandle_t m_handle;

    thrust::host_vector<TA> m_h_A;
    thrust::host_vector<TB> m_h_B;
    thrust::host_vector<TC> m_h_C;
    thrust::host_vector<TC> m_h_C_ref;

    thrust::device_vector<TA> m_d_A;
    thrust::device_vector<TB> m_d_B;
    thrust::device_vector<TC> m_d_C;

    // Random seed.
    unsigned int m_seed;
    std::default_random_engine m_random_engine;
};

class TestGeneralMatrixMultiplicationFloat
    : public TestGeneralMatrixMultiplication<float, float, float, float, float>
{
};

class TestGeneralMatrixMultiplicationDouble
    : public TestGeneralMatrixMultiplication<double, double, double, double,
                                             double>
{
};

class TestGeneralMatrixMultiplicationHalfDataFloatCompute
    : public TestGeneralMatrixMultiplication<cutlass::half_t, cutlass::half_t,
                                             cutlass::half_t, float, float>
{
};

class TestGeneralMatrixMultiplicationHalfDataHalfCompute
    : public TestGeneralMatrixMultiplication<cutlass::half_t, cutlass::half_t,
                                             cutlass::half_t, cutlass::half_t,
                                             cutlass::half_t>
{
};

#endif // CUTE_GENERAL_MATRIX_MULTIPLICATION_TEST_UTILS_HPP
#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_general_matrix_multiplication.hpp"

static auto const LAUNCH_GENERAL_MATRIX_MULTIPLICATION_FLOAT{
    launch_gemm_naive<float, float, float, float, float>};
static auto const LAUNCH_GENERAL_MATRIX_MULTIPLICATION_DOUBLE{
    launch_gemm_naive<double, double, double, double, double>};
static auto const LAUNCH_GENERAL_MATRIX_MULTIPLICATION_HALF_DATA_FLOAT_COMPUTE{
    launch_gemm_naive<cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
                      float>};
static auto const LAUNCH_GENERAL_MATRIX_MULTIPLICATION_HALF_DATA_HALF_COMPUTE{
    launch_gemm_naive<cutlass::half_t, cutlass::half_t, cutlass::half_t,
                      cutlass::half_t, cutlass::half_t>};

static auto const TRANS_A_VALUES{::testing::Values('T', 'N')};
static auto const TRANS_B_VALUES{::testing::Values('T', 'N')};
static auto const M_VALUES{::testing::Values(509, 512)};
static auto const N_VALUES{::testing::Values(509, 512)};
static auto const K_VALUES{::testing::Values(509, 512)};
static auto const LDA_VALUES{::testing::Values(512, 517)};
static auto const LDB_VALUES{::testing::Values(512, 517)};
static auto const LDC_VALUES{::testing::Values(512, 517)};

TEST_P(TestGeneralMatrixMultiplicationFloat,
       TestGeneralMatrixMultiplicationFloat)
{
    RunTest(LAUNCH_GENERAL_MATRIX_MULTIPLICATION_FLOAT);
}

TEST_P(TestGeneralMatrixMultiplicationDouble,
       TestGeneralMatrixMultiplicationDouble)
{
    RunTest(LAUNCH_GENERAL_MATRIX_MULTIPLICATION_DOUBLE);
}

TEST_P(TestGeneralMatrixMultiplicationHalfDataFloatCompute,
       TestGeneralMatrixMultiplicationHalfDataFloatCompute)
{
    RunTest(LAUNCH_GENERAL_MATRIX_MULTIPLICATION_HALF_DATA_FLOAT_COMPUTE);
}

TEST_P(TestGeneralMatrixMultiplicationHalfDataHalfCompute,
       TestGeneralMatrixMultiplicationHalfDataHalfCompute)
{
    RunTest(LAUNCH_GENERAL_MATRIX_MULTIPLICATION_HALF_DATA_HALF_COMPUTE);
}

INSTANTIATE_TEST_SUITE_P(TestGeneralMatrixMultiplicationLimited,
                         TestGeneralMatrixMultiplicationFloat,
                         ::testing::Combine(TRANS_A_VALUES, TRANS_B_VALUES,
                                            M_VALUES, N_VALUES, K_VALUES,
                                            LDA_VALUES, LDB_VALUES,
                                            LDC_VALUES));
INSTANTIATE_TEST_SUITE_P(TestGeneralMatrixMultiplicationLimited,
                         TestGeneralMatrixMultiplicationDouble,
                         ::testing::Combine(TRANS_A_VALUES, TRANS_B_VALUES,
                                            M_VALUES, N_VALUES, K_VALUES,
                                            LDA_VALUES, LDB_VALUES,
                                            LDC_VALUES));
INSTANTIATE_TEST_SUITE_P(TestGeneralMatrixMultiplicationLimited,
                         TestGeneralMatrixMultiplicationHalfDataFloatCompute,
                         ::testing::Combine(TRANS_A_VALUES, TRANS_B_VALUES,
                                            M_VALUES, N_VALUES, K_VALUES,
                                            LDA_VALUES, LDB_VALUES,
                                            LDC_VALUES));
INSTANTIATE_TEST_SUITE_P(TestGeneralMatrixMultiplicationLimited,
                         TestGeneralMatrixMultiplicationHalfDataHalfCompute,
                         ::testing::Combine(TRANS_A_VALUES, TRANS_B_VALUES,
                                            M_VALUES, N_VALUES, K_VALUES,
                                            LDA_VALUES, LDB_VALUES,
                                            LDC_VALUES));
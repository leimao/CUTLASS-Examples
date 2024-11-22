#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_matrix_transpose.hpp"

static auto const LAUNCH_TRANSPOSE_FLOAT{
    launch_transpose_shared_memory_bank_conflict_read_vectorized<float>};
static auto const LAUNCH_TRANSPOSE_DOUBLE{
    launch_transpose_shared_memory_bank_conflict_read_vectorized<double>};

static auto const M_POWER_OF_TWO_VALUES{::testing::Values(128)};
static auto const N_POWER_OF_TWO_VALUES{::testing::Values(128)};

TEST_P(TestMatrixTransposeFloat, TestMatrixTransposeFloat)
{
    RunTest(LAUNCH_TRANSPOSE_FLOAT);
}

TEST_P(TestMatrixTransposeDouble, TestMatrixTransposeDouble)
{
    RunTest(LAUNCH_TRANSPOSE_DOUBLE);
}

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeFloat,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeDouble,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));

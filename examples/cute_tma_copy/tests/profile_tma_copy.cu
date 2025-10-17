#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_tma_copy.hpp"

static auto const LAUNCH_MATRIX_COPY_FLOAT{launch_matrix_copy<float>};
static auto const LAUNCH_MATRIX_COPY_HALF{launch_matrix_copy<cutlass::half_t>};

// Test with large matrices for performance measurement
static auto const LARGE_MATRIX_VALUES{
    ::testing::Values(std::make_tuple(1024, 1024), std::make_tuple(2048, 2048),
                      std::make_tuple(4096, 4096))};

TEST_P(TestTmaCopyFloat, TestTmaCopyFloat)
{
    MeasurePerformance(LAUNCH_MATRIX_COPY_FLOAT);
}

TEST_P(TestTmaCopyHalf, TestTmaCopyHalf)
{
    MeasurePerformance(LAUNCH_MATRIX_COPY_HALF);
}

INSTANTIATE_TEST_SUITE_P(TestTmaCopyLarge, TestTmaCopyFloat,
                         LARGE_MATRIX_VALUES);
INSTANTIATE_TEST_SUITE_P(TestTmaCopyLarge, TestTmaCopyHalf,
                         LARGE_MATRIX_VALUES);
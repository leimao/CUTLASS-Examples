#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_tma_copy.hpp"

static auto const LAUNCH_MATRIX_COPY_FLOAT{launch_matrix_copy<float>};
static auto const LAUNCH_MATRIX_COPY_DOUBLE{launch_matrix_copy<double>};
static auto const LAUNCH_MATRIX_COPY_HALF{launch_matrix_copy<cute::half_t>};

// Test with small matrix sizes - using prime numbers for dimensions
static auto const SMALL_MATRIX_VALUES{::testing::Values(
    std::make_tuple(2, 2), std::make_tuple(17, 23), std::make_tuple(32, 64),
    std::make_tuple(64, 32), std::make_tuple(83, 97))};

// Test with power-of-2 matrix sizes
static auto const POWER_OF_TWO_MATRIX_VALUES{::testing::Values(
    std::make_tuple(16, 16), std::make_tuple(64, 64), std::make_tuple(128, 128),
    std::make_tuple(256, 256), std::make_tuple(512, 512))};

// Test with rectangular matrices
static auto const RECTANGULAR_MATRIX_VALUES{
    ::testing::Values(std::make_tuple(128, 256), std::make_tuple(256, 128),
                      std::make_tuple(64, 512), std::make_tuple(512, 64),
                      std::make_tuple(1024, 256))};

TEST_P(TestTmaCopyFloat, TestTmaCopyFloat)
{
    RunTest(LAUNCH_MATRIX_COPY_FLOAT);
}

TEST_P(TestTmaCopyDouble, TestTmaCopyDouble)
{
    RunTest(LAUNCH_MATRIX_COPY_DOUBLE);
}

TEST_P(TestTmaCopyHalf, TestTmaCopyHalf) { RunTest(LAUNCH_MATRIX_COPY_HALF); }

// Instantiate tests for small matrices
INSTANTIATE_TEST_SUITE_P(TestTmaCopySmall, TestTmaCopyFloat,
                         SMALL_MATRIX_VALUES);
INSTANTIATE_TEST_SUITE_P(TestTmaCopySmall, TestTmaCopyDouble,
                         SMALL_MATRIX_VALUES);
INSTANTIATE_TEST_SUITE_P(TestTmaCopySmall, TestTmaCopyHalf,
                         SMALL_MATRIX_VALUES);

// Instantiate tests for power-of-2 matrices
INSTANTIATE_TEST_SUITE_P(TestTmaCopyPowerOfTwo, TestTmaCopyFloat,
                         POWER_OF_TWO_MATRIX_VALUES);
INSTANTIATE_TEST_SUITE_P(TestTmaCopyPowerOfTwo, TestTmaCopyDouble,
                         POWER_OF_TWO_MATRIX_VALUES);
INSTANTIATE_TEST_SUITE_P(TestTmaCopyPowerOfTwo, TestTmaCopyHalf,
                         POWER_OF_TWO_MATRIX_VALUES);

// Instantiate tests for rectangular matrices
INSTANTIATE_TEST_SUITE_P(TestTmaCopyRectangular, TestTmaCopyFloat,
                         RECTANGULAR_MATRIX_VALUES);
INSTANTIATE_TEST_SUITE_P(TestTmaCopyRectangular, TestTmaCopyDouble,
                         RECTANGULAR_MATRIX_VALUES);
INSTANTIATE_TEST_SUITE_P(TestTmaCopyRectangular, TestTmaCopyHalf,
                         RECTANGULAR_MATRIX_VALUES);
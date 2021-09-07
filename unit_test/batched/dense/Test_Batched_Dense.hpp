#ifndef TEST_BATCHED_DENSE_HPP
#define TEST_BATCHED_DENSE_HPP

// Serial kernels
#include "Test_Batched_SerialAxpy.hpp"
#include "Test_Batched_SerialAxpy_Real.hpp"
#include "Test_Batched_SerialAxpy_Complex.hpp"

// Team Kernels
#include "Test_Batched_TeamAxpy.hpp"
#include "Test_Batched_TeamAxpy_Real.hpp"
#include "Test_Batched_TeamAxpy_Complex.hpp"

// TeamVector Kernels
#include "Test_Batched_TeamVectorAxpy.hpp"
#include "Test_Batched_TeamVectorAxpy_Real.hpp"
#include "Test_Batched_TeamVectorAxpy_Complex.hpp"

#endif // TEST_BATCHED_DENSE_HPP

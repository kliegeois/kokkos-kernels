#ifndef TEST_BATCHED_HPP
#define TEST_BATCHED_HPP

// Serial kernels
#include "Test_Batched_SerialEigendecomposition.hpp"
#include "Test_Batched_SerialEigendecomposition_Real.hpp"
#include "Test_Batched_SerialGemm.hpp"
#include "Test_Batched_SerialGemm_Real.hpp"
#include "Test_Batched_SerialGemm_Complex.hpp"
#include "Test_Batched_BatchedGemm.hpp"
#include "Test_Batched_BatchedGemm_Real.hpp"
#include "Test_Batched_BatchedGemm_Complex.hpp"
#include "Test_Batched_SerialGemv.hpp"
#include "Test_Batched_SerialGemv_Real.hpp"
#include "Test_Batched_SerialGemv_Complex.hpp"
#include "Test_Batched_SerialInverseLU.hpp"
#include "Test_Batched_SerialInverseLU_Real.hpp"
#include "Test_Batched_SerialInverseLU_Complex.hpp"
#include "Test_Batched_SerialLU.hpp"
#include "Test_Batched_SerialLU_Real.hpp"
#include "Test_Batched_SerialLU_Complex.hpp"
#include "Test_Batched_SerialMatUtil.hpp"
#include "Test_Batched_SerialMatUtil_Real.hpp"
#include "Test_Batched_SerialMatUtil_Complex.hpp"
#include "Test_Batched_SerialSolveLU.hpp"
#include "Test_Batched_SerialSolveLU_Real.hpp"
#include "Test_Batched_SerialSolveLU_Complex.hpp"
#include "Test_Batched_SerialSpmv.hpp"
#include "Test_Batched_SerialSpmv_Real.hpp"
#include "Test_Batched_SerialTrmm.hpp"
#include "Test_Batched_SerialTrmm_Real.hpp"
#include "Test_Batched_SerialTrmm_Complex.hpp"
#include "Test_Batched_SerialTrsm.hpp"
#include "Test_Batched_SerialTrsm_Real.hpp"
#include "Test_Batched_SerialTrsm_Complex.hpp"
#include "Test_Batched_SerialTrsv.hpp"
#include "Test_Batched_SerialTrsv_Real.hpp"
#include "Test_Batched_SerialTrsv_Complex.hpp"
#include "Test_Batched_SerialTrtri.hpp"
#include "Test_Batched_SerialTrtri_Real.hpp"
#include "Test_Batched_SerialTrtri_Complex.hpp"

// Team Kernels
#include "Test_Batched_TeamGemm.hpp"
#include "Test_Batched_TeamGemm_Real.hpp"
#include "Test_Batched_TeamGemm_Complex.hpp"
#include "Test_Batched_TeamGemv.hpp"
#include "Test_Batched_TeamGemv_Real.hpp"
#include "Test_Batched_TeamGemv_Complex.hpp"
#include "Test_Batched_TeamInverseLU.hpp"
#include "Test_Batched_TeamInverseLU_Real.hpp"
#include "Test_Batched_TeamInverseLU_Complex.hpp"
#include "Test_Batched_TeamLU.hpp"
#include "Test_Batched_TeamLU_Real.hpp"
#include "Test_Batched_TeamLU_Complex.hpp"
#include "Test_Batched_TeamMatUtil.hpp"
#include "Test_Batched_TeamMatUtil_Real.hpp"
#include "Test_Batched_TeamMatUtil_Complex.hpp"
#include "Test_Batched_TeamSolveLU.hpp"
#include "Test_Batched_TeamSolveLU_Real.hpp"
#include "Test_Batched_TeamSolveLU_Complex.hpp"
#include "Test_Batched_TeamSpmv.hpp"
#include "Test_Batched_TeamSpmv_Real.hpp"
#include "Test_Batched_TeamTrsm.hpp"
#include "Test_Batched_TeamTrsm_Real.hpp"
#include "Test_Batched_TeamTrsm_Complex.hpp"
#include "Test_Batched_TeamTrsv.hpp"
#include "Test_Batched_TeamTrsv_Real.hpp"
#include "Test_Batched_TeamTrsv_Complex.hpp"

// TeamVector Kernels
#include "Test_Batched_TeamVectorEigendecomposition.hpp"
#include "Test_Batched_TeamVectorEigendecomposition_Real.hpp"
#include "Test_Batched_TeamVectorGemm.hpp"
#include "Test_Batched_TeamVectorGemm_Real.hpp"
#include "Test_Batched_TeamVectorGemm_Complex.hpp"
#include "Test_Batched_TeamVectorQR.hpp"
#include "Test_Batched_TeamVectorQR_Real.hpp"
#include "Test_Batched_TeamVectorQR_WithColumnPivoting.hpp"
#include "Test_Batched_TeamVectorQR_WithColumnPivoting_Real.hpp"
#include "Test_Batched_TeamVectorSolveUTV.hpp"
#include "Test_Batched_TeamVectorSolveUTV_Real.hpp"
#include "Test_Batched_TeamVectorSolveUTV2.hpp"
#include "Test_Batched_TeamVectorSolveUTV2_Real.hpp"
#include "Test_Batched_TeamVectorUTV.hpp"
#include "Test_Batched_TeamVectorUTV_Real.hpp"


// Vector Kernels
#include "Test_Batched_VectorArithmatic.hpp"
#include "Test_Batched_VectorLogical.hpp"
#include "Test_Batched_VectorMath.hpp"
#include "Test_Batched_VectorMisc.hpp"
#include "Test_Batched_VectorRelation.hpp"
#include "Test_Batched_VectorView.hpp"

#endif // TEST_BATCHED_HPP

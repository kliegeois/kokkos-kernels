#ifndef __KOKKOSBATCHED_SPMV_TEAM_INTERNAL_HPP__
#define __KOKKOSBATCHED_SPMV_TEAM_INTERNAL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// Team Internal Impl
  /// ==================== 
  template<typename ArgAlgo>
  struct TeamSpmvInternal {
    template <typename OrdinalType,
              typename layout>
    KOKKOS_INLINE_FUNCTION
    static void getIndices(const OrdinalType iTemp,
                    const OrdinalType n_rows,
                    const OrdinalType n_matrices,
                    OrdinalType &iRow,
                    OrdinalType &iMatrix);

    template <typename MemberType,
              typename ScalarType,
              typename ValueType,
              typename OrdinalType,
              typename layout,
              int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member,
           const OrdinalType m, const OrdinalType nrows, 
           const ScalarType *__restrict__ alpha, const OrdinalType alphas0,
           const ValueType *__restrict__ D, const OrdinalType ds0, const OrdinalType ds1,
           const OrdinalType *__restrict__ r, const OrdinalType rs0,
           const OrdinalType *__restrict__ c, const OrdinalType cs0,
           const ValueType *__restrict__ X, const OrdinalType xs0, const OrdinalType xs1, 
           const ScalarType *__restrict__ beta, const OrdinalType betas0,
           /**/  ValueType *__restrict__ Y, const OrdinalType ys0, const OrdinalType ys1);
  };


  template<>
  template <typename OrdinalType,
            typename layout>
  KOKKOS_INLINE_FUNCTION
  void
  TeamSpmvInternal<Algo::Spmv::Unblocked>:: 
  getIndices(const OrdinalType iTemp,
             const OrdinalType n_rows,
             const OrdinalType n_matrices,
             OrdinalType &iRow,
             OrdinalType &iMatrix) {
    if (std::is_same<layout, Kokkos::LayoutLeft>::value) {
      iRow    = iTemp / n_matrices;
      iMatrix = iTemp % n_matrices;
    }
    else {
      iRow    = iTemp % n_rows;
      iMatrix = iTemp / n_rows;
    }
  }

  template<>
  template <typename MemberType,
            typename ScalarType,
            typename ValueType,
            typename OrdinalType,
            typename layout,
            int dobeta>
  KOKKOS_INLINE_FUNCTION
  int
  TeamSpmvInternal<Algo::Spmv::Unblocked>::
  invoke(const MemberType &member,
         const OrdinalType m, const OrdinalType nrows, 
         const ScalarType *__restrict__ alpha, const OrdinalType alphas0,
         const ValueType *__restrict__ D, const OrdinalType ds0, const OrdinalType ds1,
         const OrdinalType *__restrict__ r, const OrdinalType rs0,
         const OrdinalType *__restrict__ c, const OrdinalType cs0,
         const ValueType *__restrict__ X, const OrdinalType xs0, const OrdinalType xs1,
         const ScalarType *__restrict__ beta, const OrdinalType betas0,
         /**/  ValueType *__restrict__ Y, const OrdinalType ys0, const OrdinalType ys1) {


    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, 0, m * nrows),
        [&](const OrdinalType& iTemp) {
          OrdinalType iRow, iMatrix;
          getIndices<OrdinalType,layout>(iTemp, nrows, m, iRow, iMatrix);

          const OrdinalType row_length =
              r[(iRow+1)*rs0] - r[iRow*rs0];
          ValueType sum = 0;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
          for (OrdinalType iEntry = 0; iEntry < row_length; ++iEntry) {
            sum += D[iMatrix*ds0+(r[iRow*rs0]+iEntry)*ds1]
                    * X[iMatrix*xs0+c[(r[iRow*rs0]+iEntry)*cs0]*xs1];
          }

          sum *= alpha[iMatrix*alphas0];

          if (dobeta == 0) {
            Y[iMatrix*ys0+iRow*ys1] = sum;
          } else {
            Y[iMatrix*ys0+iRow*ys1] = 
                beta[iMatrix*betas0] * Y[iMatrix*ys0+iRow*ys1] + sum;
          }
      });
      
    return 0;  
  }

}


#endif

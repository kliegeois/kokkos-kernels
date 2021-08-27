#ifndef __KOKKOSBATCHED_SPMV_INTERNAL_HPP__
#define __KOKKOSBATCHED_SPMV_INTERNAL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// Serial Internal Impl
  /// ====================   
  struct SerialSpmvInternal {      
    template <typename ScalarType,
            typename ValueType,
            typename OrdinalType,
            typename layout,
            int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const OrdinalType m, const OrdinalType nrows, 
           const ScalarType *alpha,
           const ScalarType *beta, 
           /* */ ValueType *__restrict__ X, const OrdinalType xs0, const OrdinalType xs1,
           /* */ ValueType *__restrict__ Y, const OrdinalType ys0, const OrdinalType ys1,
           /* */ ValueType *__restrict__ D, const OrdinalType ds0, const OrdinalType ds1,
           /* */ OrdinalType *__restrict__ r, const OrdinalType rs0,
           /* */ OrdinalType *__restrict__ c, const OrdinalType cs0) {

      for (OrdinalType iMatrix = 0; iMatrix < m; ++iMatrix) {
        for (OrdinalType iRow = 0; iRow < nrows; ++iRow) {
            const OrdinalType row_length =
                r[(iRow+1)*rs0] - r[iRow*rs0];
            ValueType sum = 0;
#pragma unroll
            for (OrdinalType iEntry = 0; iEntry < row_length; ++iEntry) {
              sum += D[(r[iRow*rs0]+iEntry)*ds0+iMatrix*ds1]
                     * X[c[(r[iRow*rs0]+iEntry)*cs0]*xs0+iMatrix*xs1];
            }

            sum *= alpha[iMatrix];

            if (dobeta == 0) {
              Y[iRow*ys0+iMatrix*ys1] = sum;
              m_y(iRow, iGlobalMatrix) = sum;
            } else {
              Y[iRow*ys0+iMatrix*ys1] = 
                  beta[iMatrix] * Y[iRow*ys0+iMatrix*ys1] + sum;
            }
        }
      }
        
      return 0;
    }
  };

  ///
  /// TeamVector Internal Impl
  /// ==================== 
  struct TeamVectorSpmvInternal {
    template <typename OrdinalType,
              typename layout>
    KOKKOS_INLINE_FUNCTION
    void getIndices(const OrdinalType iTemp,
                    const OrdinalType n_rows,
                    const OrdinalType n_matrices,
                    OrdinalType &iRow,
                    OrdinalType &iMatrix) const {
      if (std::is_same<layout, Kokkos::LayoutRight>::value) {
        iRow    = iTemp / n_matrices;
        iMatrix = iTemp % n_matrices;
      }
      else {
        iRow    = iTemp % n_rows;
        iMatrix = iTemp / n_rows;
      }
    }
      
    template <typename ScalarType,
            typename ValueType,
            typename OrdinalType,
            typename layout,
            int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member,
           const OrdinalType m, const OrdinalType nrows, 
           const ScalarType *alpha,
           const ScalarType *beta, 
           /* */ ValueType *__restrict__ X, const OrdinalType xs0, const OrdinalType xs1,
           /* */ ValueType *__restrict__ Y, const OrdinalType ys0, const OrdinalType ys1,
           /* */ ValueType *__restrict__ D, const OrdinalType ds0, const OrdinalType ds1,
           /* */ OrdinalType *__restrict__ r, const OrdinalType rs0,
           /* */ OrdinalType *__restrict__ c, const OrdinalType cs0) {

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, 0, m * nrows),
          [&](const OrdinalType& iTemp) {
            OrdinalType iRow, iMatrix;
            this->getIndices<OrdinalType,layout>(iTemp, nrows, m, iRow, iMatrix);

            const OrdinalType row_length =
                r[(iRow+1)*rs0] - r[iRow*rs0];
            ValueType sum = 0;
#pragma unroll
            for (OrdinalType iEntry = 0; iEntry < row_length; ++iEntry) {
              sum += D[(r[iRow*rs0]+iEntry)*ds0+iMatrix*ds1]
                     * X[c[(r[iRow*rs0]+iEntry)*cs0]*xs0+iMatrix*xs1];
            }

            sum *= alpha[iMatrix];

            if (dobeta == 0) {
              Y[iRow*ys0+iMatrix*ys1] = sum;
              m_y(iRow, iGlobalMatrix) = sum;
            } else {
              Y[iRow*ys0+iMatrix*ys1] = 
                  beta[iMatrix] * Y[iRow*ys0+iMatrix*ys1] + sum;
            }
        }
      }
        
      return 0;
    }
  };

}


#endif

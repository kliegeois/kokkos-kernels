#ifndef __KOKKOSBATCHED_SPMV_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_SPMV_SERIAL_INTERNAL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// Serial Internal Impl
  /// ==================== 
  template<typename ArgAlgo>
  struct SerialSpmvInternal {
    template <typename ScalarType,
              typename ValueType,
              typename OrdinalType,
              typename layout,
              int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const OrdinalType m, const OrdinalType nrows, 
           const ScalarType *__restrict__ alpha, const OrdinalType alphas0,
           const ValueType *__restrict__ D, const OrdinalType ds0, const OrdinalType ds1,
           const OrdinalType *__restrict__ r, const OrdinalType rs0,
           const OrdinalType *__restrict__ c, const OrdinalType cs0,
           const ValueType *__restrict__ X, const OrdinalType xs0, const OrdinalType xs1, 
           const ScalarType *__restrict__ beta, const OrdinalType betas0,
           /**/  ValueType *__restrict__ Y, const OrdinalType ys0, const OrdinalType ys1) {

      for (OrdinalType iMatrix = 0; iMatrix < m; ++iMatrix) {
        for (OrdinalType iRow = 0; iRow < nrows; ++iRow) {
            const OrdinalType row_length =
                r[(iRow+1)*rs0] - r[iRow*rs0];
            ValueType sum = 0;
  #if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
  #pragma unroll
  #endif
            for (OrdinalType iEntry = 0; iEntry < row_length; ++iEntry) {
              sum += D[(r[iRow*rs0]+iEntry)*ds0+iMatrix*ds1]
                      * X[c[(r[iRow*rs0]+iEntry)*cs0]*xs0+iMatrix*xs1];
            }

            sum *= alpha[iMatrix*alphas0];

            if (dobeta == 0) {
              Y[iRow*ys0+iMatrix*ys1] = sum;
            } else {
              Y[iRow*ys0+iMatrix*ys1] = 
                  beta[iMatrix*betas0] * Y[iRow*ys0+iMatrix*ys1] + sum;
            }
        }
      }
        
      return 0;  
    }
  };

}

#endif

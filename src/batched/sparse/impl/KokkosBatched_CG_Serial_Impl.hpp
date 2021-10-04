//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.4
//       Copyright (2021) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
#ifndef __KOKKOSBATCHED_CG_SERIAL_IMPL_HPP__
#define __KOKKOSBATCHED_CG_SERIAL_IMPL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_Dot.hpp"
#include "KokkosBatched_Axpy.hpp"

namespace KokkosBatched {

  ///
  /// Serial CG
  ///   No nested parallel_for is used inside of the function.
  ///

  struct SerialCG {
    template<typename ValuesViewType,
             typename IntView,
             typename VectorViewType,
             typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const ValuesViewType &values,
           const IntView &row_ptr,
           const IntView &colIndices,
           const VectorViewType &B,
           const VectorViewType &X,
           const size_t maximum_iteration = 200,
           const typename Kokkos::Details::ArithTraits<ScalarType>::mag_type tolerance = Kokkos::Details::ArithTraits<ScalarType>::epsilon()) {
            typedef typename IntView::non_const_value_type OrdinalType;
            typedef typename Kokkos::Details::ArithTraits<ScalarType>::mag_type MagnitudeType;
            typedef Kokkos::View<MagnitudeType*,Kokkos::LayoutLeft,typename ValuesViewType::device_type> NormViewType;

            const OrdinalType numMatrices = X.extent(0);
            const OrdinalType numRows = X.extent(1);

            VectorViewType P("directions", numMatrices, numRows);
            VectorViewType R("residuals", numMatrices, numRows);
            VectorViewType Q("tmp", numMatrices, numRows);

            NormViewType sqr_norm_0("squared norm 0", numMatrices);
            NormViewType sqr_norm_j("squared norm j", numMatrices);

            NormViewType alpha("alpha", numMatrices);
            NormViewType beta("beta", numMatrices);
            NormViewType tmp("tmp", numMatrices);

            Kokkos::deep_copy(alpha, MagnitudeType(-1.0));
            Kokkos::deep_copy(beta, MagnitudeType(1.0));

            // Deep copy of b into r_0:
            Kokkos::deep_copy(R, B);

            // r_0 := b - A x_0
            SerialSpmv<Trans::NoTranspose>::template invoke<ValuesViewType, IntView, VectorViewType, VectorViewType, NormViewType, NormViewType, 1>(alpha, values, row_ptr, colIndices, X, beta, R);

            // Deep copy of r_0 into p_0:
            Kokkos::deep_copy(P, R);

            SerialDot::template invoke<VectorViewType, NormViewType>(R, R, sqr_norm_0);
            Kokkos::deep_copy(sqr_norm_j, sqr_norm_0);

            int status = 1;
            bool verbose_print = true;
            bool all_converged;

            for(size_t j = 0; j < maximum_iteration; ++j) {
              // q := A p_j (alpha has no influence as "NormViewType, 0>" )
              SerialSpmv<Trans::NoTranspose>::template invoke<ValuesViewType, IntView, VectorViewType, VectorViewType, NormViewType, NormViewType, 0>(beta, values, row_ptr, colIndices, P, alpha, Q);

              SerialDot::template invoke<VectorViewType, NormViewType>(Q, P, tmp);

              for(size_t i = 0; i < numMatrices; ++i) {
                alpha(i) = sqr_norm_j(i) / tmp(i);
              }

              // x_{j+1} := alpha p_j + x_j 
              SerialAxpy::template invoke<VectorViewType, NormViewType>(alpha, P, X);

              // r_{j+1} := - alpha q + r_j 
              for(size_t i = 0; i < numMatrices; ++i) {
                alpha(i) = -alpha(i);
              }

              SerialAxpy::template invoke<VectorViewType, NormViewType>(alpha, Q, R);

              SerialDot::template invoke<VectorViewType, NormViewType>(R, R, tmp);

              for(size_t i = 0; i < numMatrices; ++i) {
                beta(i) = tmp(i) / sqr_norm_j(i);
              }

              Kokkos::deep_copy(sqr_norm_j, tmp);

              // Relative convergence check:
              all_converged = true;
              for(size_t i = 0; i < numMatrices; ++i) {
                MagnitudeType norm_0 = sqrt(sqr_norm_0(i));
                MagnitudeType norm_j = sqrt(sqr_norm_j(i));
                if(verbose_print) {
                  printf("CG iteration %d, system %d: norm of the initial residual %f, norm of the curent residual %f, relative norm %f", j, i, norm_0, norm_j, norm_j/norm_0);
                }
                if(norm_j/norm_0 > tolerance)
                  all_converged = false;
              }
              if(all_converged) {
                status = 0;
                break;
              }

              // p_{j+1} := beta p_j + r_{j+1}
              Kokkos::deep_copy(Q, R);
              SerialAxpy::template invoke<VectorViewType, NormViewType>(beta, P, Q);
              Kokkos::deep_copy(P, Q);
            }

            return status;
          }
  };

}


#endif

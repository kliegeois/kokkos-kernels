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
#ifndef __KOKKOSBATCHED_CG_TEAMVECTOR_IMPL_HPP__
#define __KOKKOSBATCHED_CG_TEAMVECTOR_IMPL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// TeamVector CG
  ///   Two nested parallel_for with both TeamVectorRange and ThreadVectorRange 
  ///   (or one with TeamVectorRange) are used inside.  
  ///

  template<typename MemberType>
  struct TeamVectorCG {
    template<typename ValuesViewType,
             typename IntView,
             typename VectorViewType,
             typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const ValuesViewType &values,
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
            member.team_barrier();
            TeamVectorSpmv<MemberType,Trans::NoTranspose>::template invoke<ValuesViewType, IntView, VectorViewType, VectorViewType, NormViewType, NormViewType, 1>(member, alpha, values, row_ptr, colIndices, X, beta, R);
            member.team_barrier();

            // Deep copy of r_0 into p_0:
            Kokkos::deep_copy(P, R);

            TeamVectorDot<MemberType>::template invoke<VectorViewType, NormViewType>(member, R, R, sqr_norm_0);
            member.team_barrier();

            Kokkos::deep_copy(sqr_norm_j, sqr_norm_0);

            int status = 1;
            bool verbose_print = true;
            int number_not_converged = 0;

            for(size_t j = 0; j < maximum_iteration; ++j) {
              // q := A p_j (alpha has no influence as "NormViewType, 0>" )
              TeamVectorSpmv<MemberType,Trans::NoTranspose>::template invoke<ValuesViewType, IntView, VectorViewType, VectorViewType, NormViewType, NormViewType, 0>(member, beta, values, row_ptr, colIndices, P, alpha, Q);
              member.team_barrier();

              TeamVectorDot<MemberType>::template invoke<VectorViewType, NormViewType>(member, Q, P, tmp);
              member.team_barrier();

              Kokkos::parallel_for(
                Kokkos::TeamVectorRange(member, 0, numMatrices),
                [&](const OrdinalType& i) {
                  alpha(i) = sqr_norm_j(i) / tmp(i);
              });
              member.team_barrier();

              // x_{j+1} := alpha p_j + x_j 
              TeamVectorAxpy<MemberType>::template invoke<VectorViewType, NormViewType>(member, alpha, P, X);
              member.team_barrier();

              // r_{j+1} := - alpha q + r_j 
              Kokkos::parallel_for(
                Kokkos::TeamVectorRange(member, 0, numMatrices),
                [&](const OrdinalType& i) {
                  alpha(i) = -alpha(i);
              });
              member.team_barrier();

              TeamVectorAxpy<MemberType>::template invoke<VectorViewType, NormViewType>(member, alpha, Q, R);
              member.team_barrier();

              TeamVectorDot<MemberType>::template invoke<VectorViewType, NormViewType>(member, R, R, tmp);
              member.team_barrier();

              Kokkos::parallel_for(
                Kokkos::TeamVectorRange(member, 0, numMatrices),
                [&](const OrdinalType& i) {
                  beta(i) = tmp(i) / sqr_norm_j(i);
              });

              Kokkos::deep_copy(sqr_norm_j, tmp);

              // Relative convergence check:
              number_not_converged = 0;
              Kokkos::parallel_reduce(
                Kokkos::TeamVectorRange(member, 0, numMatrices),
                [&](const OrdinalType& i, int& lnumber_not_converged) {
                if(sqr_norm_j(i)/sqr_norm_0(i) > tolerance*tolerance)
                  ++lnumber_not_converged;
              }, number_not_converged);

              member.team_barrier();

              if(number_not_converged == 0) {
                status = 0;
                break;
              }

              // p_{j+1} := beta p_j + r_{j+1}
              Kokkos::deep_copy(Q, R);
              member.team_barrier();
              TeamVectorAxpy<MemberType>::template invoke<VectorViewType, NormViewType>(member, beta, P, Q);
              member.team_barrier();
              Kokkos::deep_copy(P, Q);
              member.team_barrier();
            }

            return status;
          }
  };
}

#endif

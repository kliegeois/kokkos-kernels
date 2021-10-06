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
#ifndef __KOKKOSBATCHED_CG_TEAM_IMPL_HPP__
#define __KOKKOSBATCHED_CG_TEAM_IMPL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_Axpy.hpp"
#include "KokkosBatched_Copy_Decl.hpp"
#include "KokkosBatched_Dot.hpp"
#include "KokkosBatched_Spmv.hpp"

namespace KokkosBatched {

template <class XType>
void write1DArrayTofile(const XType x, std::string name) {
  std::ofstream myfile;
  myfile.open(name);

  typename XType::HostMirror x_h = Kokkos::create_mirror_view(x);

  Kokkos::deep_copy(x_h, x);

  for (int i = 0; i < x_h.extent(0); ++i) {
    myfile << x_h(i) << " ";
  }

  myfile.close();
}

template <class XType>
void write2DArrayTofile(const XType x, std::string name) {
  std::ofstream myfile;
  myfile.open(name);

  typename XType::HostMirror x_h = Kokkos::create_mirror_view(x);

  Kokkos::deep_copy(x_h, x);

  for (int i = 0; i < x_h.extent(0); ++i) {
    for (int j = 0; j < x_h.extent(1); ++j) {
      myfile << x_h(i, j) << " ";
    }
    myfile << std::endl;
  }

  myfile.close();
}

  ///
  /// Team CG
  ///   A nested parallel_for with TeamThreadRange is used.
  ///

  template<typename MemberType>
  struct TeamCG {
    template<typename ValuesViewType,
             typename IntView,
             typename VectorViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const ValuesViewType &values,
           const IntView &row_ptr,
           const IntView &colIndices,
           const VectorViewType &_B,
           const VectorViewType &_X,
           const size_t maximum_iteration = 200,
           const typename Kokkos::Details::ArithTraits<typename ValuesViewType::non_const_value_type>::mag_type tolerance = Kokkos::Details::ArithTraits<typename ValuesViewType::non_const_value_type>::epsilon()) {
            typedef typename IntView::non_const_value_type OrdinalType;
            typedef typename Kokkos::Details::ArithTraits<typename ValuesViewType::non_const_value_type>::mag_type MagnitudeType;
            typedef Kokkos::View<MagnitudeType*,Kokkos::LayoutLeft,typename ValuesViewType::device_type> NormViewType;

            using ScratchPadNormViewType =
                Kokkos::View<MagnitudeType*,
                            typename VectorViewType::execution_space::scratch_memory_space>;
            using ScratchPadVectorViewType =
                Kokkos::View<typename ValuesViewType::non_const_value_type **, typename VectorViewType::array_layout, typename VectorViewType::execution_space::scratch_memory_space>;                                   

            const OrdinalType numMatrices = _X.extent(0);
            const OrdinalType numRows = _X.extent(1);

            ScratchPadVectorViewType X(member.team_scratch(0), numMatrices, numRows);
            ScratchPadVectorViewType B(member.team_scratch(0), numMatrices, numRows);

            TeamCopy<MemberType, Trans::NoTranspose>::invoke(member, _X, X);
            TeamCopy<MemberType, Trans::NoTranspose>::invoke(member, _B, B);

            ScratchPadVectorViewType P(member.team_scratch(0), numMatrices, numRows);
            ScratchPadVectorViewType R(member.team_scratch(0), numMatrices, numRows);
            ScratchPadVectorViewType Q(member.team_scratch(0), numMatrices, numRows);

            ScratchPadNormViewType sqr_norm_0(member.team_scratch(0), numMatrices);
            ScratchPadNormViewType sqr_norm_j(member.team_scratch(0), numMatrices);

            ScratchPadNormViewType alpha(member.team_scratch(0), numMatrices);
            ScratchPadNormViewType beta(member.team_scratch(0), numMatrices);
            ScratchPadNormViewType tmp(member.team_scratch(0), numMatrices);

            ScratchPadNormViewType one(member.team_scratch(0), numMatrices);
            ScratchPadNormViewType m_one(member.team_scratch(0), numMatrices);

            Kokkos::parallel_for(
              Kokkos::TeamThreadRange(member, 0, numMatrices),
              [&](const OrdinalType& i) {
                one(i) = MagnitudeType(1.0);
                m_one(i) = MagnitudeType(-1.0);
            });

            // Deep copy of b into r_0:
            TeamCopy<MemberType, Trans::NoTranspose>::invoke(member, B, R);

            // r_0 := b - A x_0
            member.team_barrier();
            TeamSpmv<MemberType,Trans::NoTranspose>::template invoke<ValuesViewType, IntView, ScratchPadVectorViewType, ScratchPadVectorViewType, ScratchPadNormViewType, ScratchPadNormViewType, 1>(member, m_one, values, row_ptr, colIndices, X, one, R);
            member.team_barrier();


            TeamCopy<MemberType, Trans::NoTranspose>::invoke(member, R, _X);

            // Deep copy of r_0 into p_0:
            TeamCopy<MemberType, Trans::NoTranspose>::invoke(member, R, P);

            SerialDot::template invoke<ScratchPadVectorViewType, ScratchPadNormViewType>(R, R, sqr_norm_0);
            member.team_barrier();

            Kokkos::parallel_for(
              Kokkos::TeamThreadRange(member, 0, numMatrices),
              [&](const OrdinalType& i) {
                sqr_norm_j(i) = sqr_norm_0(i);
            });

            int status = 1;
            int number_not_converged = 0;

            for(size_t j = 0; j < maximum_iteration; ++j) {
              // q := A p_j (m_one has no influence as "NormViewType, 0>" )
              TeamSpmv<MemberType,Trans::NoTranspose>::template invoke<ValuesViewType, IntView, ScratchPadVectorViewType, ScratchPadVectorViewType, ScratchPadNormViewType, ScratchPadNormViewType, 0>(member, one, values, row_ptr, colIndices, P, m_one, Q);
              member.team_barrier();

              SerialDot::template invoke<ScratchPadVectorViewType, ScratchPadNormViewType>(P, Q, tmp);
              member.team_barrier();

              Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, 0, numMatrices),
                [&](const OrdinalType& i) {
                  alpha(i) = sqr_norm_j(i) / tmp(i);

                  printf("CG iteration %d, system %d: alpha %f, q.dot(r) %f \n", (int) j, (int) i, alpha(i), tmp(i));
              });
              member.team_barrier();

              // x_{j+1} := alpha p_j + x_j 
              TeamAxpy<MemberType>::template invoke<ScratchPadVectorViewType, ScratchPadNormViewType>(member, alpha, P, X);
              member.team_barrier();

              // r_{j+1} := - alpha q + r_j 
              Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, 0, numMatrices),
                [&](const OrdinalType& i) {
                  alpha(i) = -alpha(i);
              });
              member.team_barrier();

              TeamAxpy<MemberType>::template invoke<ScratchPadVectorViewType, ScratchPadNormViewType>(member, alpha, Q, R);
              member.team_barrier();

              SerialDot::template invoke<ScratchPadVectorViewType, ScratchPadNormViewType>(R, R, tmp);
              member.team_barrier();

              Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, 0, numMatrices),
                [&](const OrdinalType& i) {
                  beta(i) = tmp(i) / sqr_norm_j(i);

                  printf("CG iteration %d, system %d: beta %f\n", (int) j, (int) i, beta(i));

              });

              Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, 0, numMatrices),
                [&](const OrdinalType& i) {
                  sqr_norm_j(i) = tmp(i);


                  printf("CG iteration %d, system %d: sqr norm of the initial residual %f, sqr norm of the curent residual %f\n", (int) j, (int) i, sqr_norm_0(i), sqr_norm_j(i));
              });

              // Relative convergence check:
              number_not_converged = 0;
              Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member, 0, numMatrices),
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
              TeamCopy<MemberType, Trans::NoTranspose>::invoke(member, R, Q);
              member.team_barrier();
              TeamAxpy<MemberType>::template invoke<ScratchPadVectorViewType, ScratchPadNormViewType>(member, beta, P, Q);
              member.team_barrier();
              TeamCopy<MemberType, Trans::NoTranspose>::invoke(member, Q, P);
              member.team_barrier();
            }

            return status;
          }
  };
}

#endif

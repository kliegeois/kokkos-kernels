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
#ifndef __KOKKOSBATCHED_GMRES_TEAMVECTOR_IMPL_HPP__
#define __KOKKOSBATCHED_GMRES_TEAMVECTOR_IMPL_HPP__

/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_Axpy.hpp"
#include "KokkosBatched_Copy_Decl.hpp"
#include "KokkosBatched_Dot.hpp"
#include "KokkosBatched_Spmv.hpp"
#include "KokkosBatched_Xpay.hpp"
#include "KokkosBatched_Givens_Serial_Internal.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Identity.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"

namespace KokkosBatched {

///
/// TeamVector GMRES
///   Two nested parallel_for with both TeamVectorRange and ThreadVectorRange
///   (or one with TeamVectorRange) are used inside.
///

template <typename MemberType>
struct TeamVectorGMRES {
  template <typename OperatorType, typename VectorViewType,
            typename PrecOperatorType, typename KrylovHandleType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType& member, const OperatorType& A, const VectorViewType& _B,
      const VectorViewType& _X,
      const PrecOperatorType& P,
      const KrylovHandleType& handle) {
    typedef int OrdinalType;
    typedef typename Kokkos::Details::ArithTraits<
        typename VectorViewType::non_const_value_type>::mag_type MagnitudeType;
    typedef Kokkos::Details::ArithTraits<MagnitudeType> ATM;

    using ScratchPadNormViewType = Kokkos::View<
        MagnitudeType*,
        typename VectorViewType::execution_space::scratch_memory_space>;
    using ScratchPadVectorViewType = Kokkos::View<
        typename VectorViewType::non_const_value_type**,
        typename VectorViewType::array_layout,
        typename VectorViewType::execution_space::scratch_memory_space>;
    using ScratchPadMultiVectorViewType = Kokkos::View<
        typename VectorViewType::non_const_value_type***,
        typename VectorViewType::array_layout,
        typename VectorViewType::execution_space::scratch_memory_space>;
    using TeamVectorCopy1D = TeamVectorCopy<MemberType, Trans::NoTranspose, 1>;

    const OrdinalType numMatrices = _X.extent(0);
    const OrdinalType numRows     = _X.extent(1);

    size_t maximum_iteration = handle.get_max_iteration() < numRows
                                   ? handle.get_max_iteration()
                                   : numRows;
    const MagnitudeType tolerance     = handle.get_tolerance();
    const MagnitudeType max_tolerance = 0.;

    int n_V = numRows;
    int n_H = maximum_iteration + 1;
    int n_Givens = 2;

    int offset_V = 0;
    int offset_H = offset_V + n_V;
    int offset_Givens = offset_H + n_H;

    ScratchPadMultiVectorViewType tmp_3D(member.team_scratch(handle.get_Arnoldi_level()), numMatrices, maximum_iteration, n_V+n_H+n_Givens);
    auto V = Kokkos::subview(tmp_3D, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(offset_V, offset_V + n_V));
    auto H = Kokkos::subview(tmp_3D, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(offset_H, offset_H + n_H));
    auto Givens = Kokkos::subview(tmp_3D, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(offset_Givens, offset_Givens + n_Givens));

    int n_G = maximum_iteration + 1;
    int n_W = numRows;
    int n_X = numRows;
    int n_mask = 1;
    int n_tmp = 1;

    int offset_G = 0;
    int offset_W = offset_G + n_G;
    int offset_X = offset_W + n_W;
    int offset_mask = offset_X + n_X;
    int offset_tmp = offset_mask + n_mask;
  
    ScratchPadVectorViewType tmp_2D(member.team_scratch(handle.get_other_level()), numMatrices, n_G+n_W+n_X+n_mask+n_tmp);

    auto G = Kokkos::subview(tmp_2D, Kokkos::ALL, Kokkos::make_pair(offset_G, offset_G + n_G));
    auto W = Kokkos::subview(tmp_2D, Kokkos::ALL, Kokkos::make_pair(offset_W, offset_W + n_W));
    auto X = Kokkos::subview(tmp_2D, Kokkos::ALL, Kokkos::make_pair(offset_X, offset_X + n_X));
    auto mask = Kokkos::subview(tmp_2D, Kokkos::ALL, offset_mask);
    auto tmp = Kokkos::subview(tmp_2D, Kokkos::ALL, offset_tmp);

    TeamVectorCopy<MemberType>::invoke(member, _X, X);
    // Deep copy of b into r_0:
    TeamVectorCopy<MemberType>::invoke(member, _B, W);

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, 0, numMatrices),
                         [&](const OrdinalType& i) { mask(i) = 1.; });

    // r_0 := b - A x_0
    member.team_barrier();
    A.template apply<Trans::NoTranspose, Mode::TeamVector>(member, X, W, -1, 1);
    member.team_barrier();

    P.template apply<Trans::NoTranspose, Mode::TeamVector, 1>(member, W, W);
    member.team_barrier();

    TeamVectorDot<MemberType>::invoke(member, W, W, tmp);
    member.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, 0, numMatrices),
                         [&](const OrdinalType& i) {
                           tmp(i) = ATM::sqrt(tmp(i));
                           G(i, 0) = tmp(i) > max_tolerance ? tmp(i) : 0.;
                           handle.set_norm(member.league_rank(), i, 0, tmp(i));
                           tmp(i) = tmp(i) > max_tolerance ? 1. / tmp(i) : 0.;
                         });

    member.team_barrier();  // Finish writing to tmp

    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, 0, numMatrices * numRows),
        [&](const OrdinalType& iTemp) {
          OrdinalType iRow, iMatrix;
          getIndices<OrdinalType, typename VectorViewType::array_layout>(
              iTemp, numRows, numMatrices, iRow, iMatrix);
          V(iMatrix, 0, iRow) = W(iMatrix, iRow) * tmp(iMatrix);
        });

    int status = 1;
    // int number_not_converged = 0;

    for (size_t j = 0; j < maximum_iteration; ++j) {
      member.team_barrier();  // Finish writing to V
      // q := A p_j
      auto V_j = Kokkos::subview(V, Kokkos::ALL, j, Kokkos::ALL);

      A.template apply<Trans::NoTranspose, Mode::TeamVector>(member, V_j, W);
      member.team_barrier();
      P.template apply<Trans::NoTranspose, Mode::TeamVector, 1>(member, W, W);
      member.team_barrier();

      if (handle.get_ortho_strategy()==0) {
        auto V_old = Kokkos::subview(V, Kokkos::ALL, Kokkos::make_pair(0, (int) j+1), Kokkos::ALL);
        auto H_old = Kokkos::subview(H, Kokkos::ALL, j, Kokkos::make_pair(0, (int) j+1));
        member.team_barrier();
        // Inner products
        TeamVectorGemv<MemberType, Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(member, 1, 
        V_old, W, 0, H_old);
        member.team_barrier();
 
        // Update
        TeamVectorGemv<MemberType, Trans::Transpose, Algo::Gemv::Unblocked>::invoke(member, -1, 
        V_old, H_old, 1, W);
        member.team_barrier();
      }
      if (handle.get_ortho_strategy()==1) {
        for (size_t i = 0; i < j + 1; ++i) {
          auto V_i = Kokkos::subview(V, Kokkos::ALL, i, Kokkos::ALL);
          TeamVectorDot<MemberType>::invoke(member, W, V_i, tmp);
          member.team_barrier();
          TeamVectorCopy1D::invoke(member, tmp,
                                  Kokkos::subview(H, Kokkos::ALL, j, i));
          member.team_barrier();
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(member, 0, numMatrices),
              [&](const OrdinalType& ii) { tmp(ii) = -tmp(ii); });

          member.team_barrier();  // Finish writing to tmp

          TeamVectorAxpy<MemberType>::invoke(member, tmp, V_i, W);
          member.team_barrier();  // Finish writing to W
        }
      }

      TeamVectorDot<MemberType>::invoke(member, W, W, tmp);
      member.team_barrier();
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, 0, numMatrices),
          [&](const OrdinalType& i) {
            H(i, j, j + 1) = ATM::sqrt(tmp(i));
            tmp(i) = H(i, j, j + 1) > max_tolerance ? 1. / H(i, j, j + 1) : 0.;
          });
      member.team_barrier();
      if (j + 1 < maximum_iteration) {
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, 0, numMatrices * numRows),
            [&](const OrdinalType& iTemp) {
              OrdinalType iRow, iMatrix;
              getIndices<OrdinalType, typename VectorViewType::array_layout>(
                  iTemp, numRows, numMatrices, iRow, iMatrix);
              V(iMatrix, j + 1, iRow) = W(iMatrix, iRow) * tmp(iMatrix);
            });
        member.team_barrier();
      }

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, 0, numMatrices),
          [&](const OrdinalType& l) {
            // Apply the previous Givens rotations:
            auto H_j = Kokkos::subview(H, l, j, Kokkos::ALL);

            if (mask(l) == 1.) {
              for (size_t i = 0; i < j; ++i) {
                auto tmp1 =
                    Givens(l, i, 0) * H_j(i) + Givens(l, i, 1) * H_j(i + 1);
                auto tmp2 =
                    -Givens(l, i, 1) * H_j(i) + Givens(l, i, 0) * H_j(i + 1);
                H_j(i)     = tmp1;
                H_j(i + 1) = tmp2;
              }

              // Compute the new Givens rotation:
              Kokkos::pair<typename VectorViewType::non_const_value_type,
                           typename VectorViewType::non_const_value_type>
                  G_new(1, 0);
              typename VectorViewType::non_const_value_type alpha = 0;
              SerialGivensInternal::invoke(H_j(j), H_j(j + 1), &G_new, &alpha);

              Givens(l, j, 0) = G_new.first;
              Givens(l, j, 1) = G_new.second;

              // Apply the new Givens rotation:
              auto tmp1 =
                  Givens(l, j, 0) * H_j(j) + Givens(l, j, 1) * H_j(j + 1);
              auto tmp2 =
                  -Givens(l, j, 1) * H_j(j) + Givens(l, j, 0) * H_j(j + 1);
              H_j(j)     = tmp1;
              H_j(j + 1) = tmp2;

              G(l, j + 1) = -Givens(l, j, 1) * G(l, j);
              G(l, j) *= Givens(l, j, 0);
            } else {
              H_j(j)      = 1.;
              G(l, j + 1) = 0.;
            }

            auto res_norm = std::abs(G(l, j + 1)) / G(l, 0);

            handle.set_norm(member.league_rank(), l, j+1, res_norm);

            if (mask(l) == 1. && res_norm < tolerance) {
              mask(l)     = 0.;
              G(l, j + 1) = 0.;
              handle.set_iteration(member.league_rank(), l, j);
            }
          });
      member.team_barrier();
      bool all_converged = true;
      for (OrdinalType l = 0; l < numMatrices; ++l)
        all_converged = (all_converged && mask(l) == 0.);
      if (all_converged) {
        maximum_iteration = j;
        break;
      }
    }

    member.team_barrier();  // Finish writing to G

    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, 0, numMatrices),
        [&](const OrdinalType& l) {
          if(member.league_rank() == 0 && l == 0) {
            for (size_t i = 0; i < H.extent(1); ++i)
              for (size_t j = 0; j < H.extent(2); ++j)
                printf(" H(%d, %d) = %f\n", (int) i, (int) j, H(l, i, j));
            for (size_t i = 0; i < G.extent(1); ++i)
              printf("before  G(%d) = %f\n", (int) i, G(l, i));
          }
          SerialTrsm<Side::Left, Uplo::Upper, Trans::Transpose, Diag::NonUnit,
                     Algo::Trsm::Unblocked>::template invoke(1,
                                                             Kokkos::subview(
                                                                 H, l,
                                                                 Kokkos::make_pair(0, (int) maximum_iteration),
                                                                 Kokkos::make_pair(0, (int) maximum_iteration)),
                                                             Kokkos::subview(
                                                                 G, l,
                                                                 Kokkos::make_pair(0, (int) maximum_iteration)));
          if(member.league_rank() == 0 && l == 0) {
            for (size_t i = 0; i < G.extent(1); ++i)
              printf("after  G(%d) = %f\n", (int) i, G(l, i));
          }
        });

    member.team_barrier();  // Finish writing to G

    if (handle.get_ortho_strategy()==0) {
      TeamVectorGemv<MemberType, Trans::Transpose, Algo::Gemv::Unblocked>::invoke(member, 1, 
      Kokkos::subview(V, Kokkos::ALL, Kokkos::make_pair(0, (int) maximum_iteration), Kokkos::ALL), 
      Kokkos::subview(G, Kokkos::ALL, Kokkos::make_pair(0, (int) maximum_iteration)), 
      1, 
      X);
    }
    if (handle.get_ortho_strategy()==1) {
      for (size_t j = 0; j < maximum_iteration; ++j)
        TeamVectorAxpy<MemberType>::invoke(
            member, Kokkos::subview(G, Kokkos::ALL, j),
            Kokkos::subview(V, Kokkos::ALL, j, Kokkos::ALL), X);
    }

    member.team_barrier();  // Finish writing to X

    TeamVectorCopy<MemberType>::invoke(member, X, _X);

    member.team_barrier();

    if (handle.get_compute_last_residual()) {
      TeamVectorCopy<MemberType>::invoke(member, _B, W);
      member.team_barrier();
      A.template apply<Trans::NoTranspose, Mode::TeamVector>(member, X, W, -1, 1);
      member.team_barrier();
      P.template apply<Trans::NoTranspose, Mode::TeamVector, 1>(member, W, W);
      member.team_barrier();
      TeamVectorDot<MemberType>::invoke(member, W, W, tmp);
      member.team_barrier();

      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, 0, numMatrices),
                          [&](const OrdinalType& i) {
                            tmp(i) = ATM::sqrt(tmp(i));
                            handle.set_last_norm(member.league_rank(), i, tmp(i));
                          });
    }
    return status;
  }

  template <typename OperatorType, typename VectorViewType, typename KrylovHandleType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType& member, const OperatorType& A, const VectorViewType& _B,
      const VectorViewType& _X,
      const KrylovHandleType& handle) {
    Identity P;
    return invoke<OperatorType, VectorViewType, Identity>(member, A, _B, _X,
                                                          P, handle);
  }
};
}  // namespace KokkosBatched

#endif

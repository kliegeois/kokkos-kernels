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

#include <fstream>

#define KOKKOSKERNELS_DEBUG_LEVEL 0

#include "Kokkos_Core.hpp"
#include "Kokkos_Timer.hpp"
#include "Kokkos_Random.hpp"
#include "Kokkos_UnorderedMap.hpp"
#include "Kokkos_Sort.hpp"

/// KokkosKernels headers
#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

#include <Kokkos_ArithTraits.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Vector.hpp>
#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>
#include <KokkosBatched_AddRadial_Decl.hpp>
#include <KokkosBatched_AddRadial_Impl.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>
#include <KokkosBatched_Gemm_Team_Impl.hpp>
#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Gemv_Serial_Impl.hpp>
#include <KokkosBatched_Gemv_Team_Impl.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <KokkosBatched_Trsm_Serial_Impl.hpp>
#include <KokkosBatched_Trsm_Team_Impl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <KokkosBatched_Trsv_Serial_Impl.hpp>
#include <KokkosBatched_Trsv_Team_Impl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>
#include <KokkosBatched_LU_Team_Impl.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Team_Impl.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Team_Impl.hpp"

#include "examples_helper.hpp"

#include "KokkosBatched_Spmv.hpp"
#include "KokkosBatched_CrsMatrix.hpp"
#include "KokkosBatched_Krylov_Handle.hpp"
#include "KokkosBatched_GMRES.hpp"
#include "KokkosBatched_JacobiPrec.hpp"
#include "KokkosBatched_Dot.hpp"
#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Dot_Internal.hpp"
#include "KokkosBatched_Spmv_Serial_Impl.hpp"
#include "KokkosBatched_Copy_Decl.hpp"

typedef Kokkos::DefaultExecutionSpace exec_space;

template <typename DeviceType, typename AViewType, typename XYViewType, typename IntViewType>
struct Functor_TestStaticPivoting {
  const AViewType _A;
  const AViewType _PDAD;
  const AViewType _tmp;
  const XYViewType _D1;
  const XYViewType _D2;
  const IntViewType _P;
  const XYViewType _X;
  const XYViewType _DX;
  const XYViewType _Y;
  const XYViewType _PDY;
  const int _N_team;

  KOKKOS_INLINE_FUNCTION
  Functor_TestStaticPivoting(const AViewType &A, const AViewType &PDAD, const AViewType &tmp,
                                  const XYViewType &D1, const XYViewType &D2,
                                  const IntViewType &P, const XYViewType &X, const XYViewType &DX, const XYViewType &Y, const XYViewType &PDY, const int N_team)
      : _A(A), _PDAD(PDAD), _tmp(tmp), _D1(D1), _D2(D2), _P(P), _X(X), _DX(DX), _Y(Y), _PDY(PDY), _N_team(N_team) {
  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType &member) const {
    const int first_matrix = static_cast<int>(member.league_rank()) * _N_team;
    const int N            = _A.extent(0);
    const int n            = _A.extent(1);
    const int last_matrix =
        (static_cast<int>(member.league_rank() + 1) * _N_team < N
             ? static_cast<int>(member.league_rank() + 1) * _N_team
             : N);
    using TeamVectorCopy1D = KokkosBatched::TeamVectorCopy<MemberType, KokkosBatched::Trans::NoTranspose, 1>;

    auto a = Kokkos::subview(_A, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL,
                             Kokkos::ALL);
    auto pdad = Kokkos::subview(_PDAD, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL,
                             Kokkos::ALL);
    auto tmp = Kokkos::subview(_tmp, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL,
                             Kokkos::ALL);
    auto d1 = Kokkos::subview(_D1, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);
    auto d2 = Kokkos::subview(_D2, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);
    auto p = Kokkos::subview(_P, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);
    auto x = Kokkos::subview(_X, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);                             
    auto dx = Kokkos::subview(_DX, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL); 
    auto y = Kokkos::subview(_Y, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);                             
    auto pdy = Kokkos::subview(_PDY, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL); 

    member.team_barrier();
    computePDD(member, a, p, d1, d2, tmp);
    member.team_barrier();
    applyPDD(member, a, p, d1, d2, pdad);
    member.team_barrier();
    applyPD(member, y, p, d1, pdy);
    member.team_barrier();

    for (int i_matrix = first_matrix; i_matrix < last_matrix; ++i_matrix) {
      auto pdad = Kokkos::subview(_PDAD, i_matrix, Kokkos::ALL, Kokkos::ALL);
      auto tmp = Kokkos::subview(_tmp, i_matrix, Kokkos::make_pair(0, n), Kokkos::make_pair(0, n));
      auto d2 = Kokkos::subview(_D2, i_matrix, Kokkos::ALL);
      auto x = Kokkos::subview(_X, i_matrix, Kokkos::ALL);
      auto pdy = Kokkos::subview(_PDY, i_matrix, Kokkos::ALL);

      KokkosBatched::TeamVectorCopy<MemberType, KokkosBatched::Trans::NoTranspose>::invoke(member, pdad, tmp);
      TeamVectorCopy1D::invoke(member, pdy, x);

      member.team_barrier();
      KokkosBatched::TeamLU<MemberType, KokkosBatched::Algo::Level3::Unblocked>::invoke(member, tmp);
      member.team_barrier();
      KokkosBatched::TeamTrsm<MemberType, KokkosBatched::Side::Left, KokkosBatched::Uplo::Lower,
                KokkosBatched::Trans::NoTranspose, KokkosBatched::Diag::Unit,
                KokkosBatched::Algo::Level3::Unblocked>::invoke(member, 1.0, tmp, x);
      member.team_barrier();
      
      KokkosBatched::TeamTrsm<MemberType, KokkosBatched::Side::Left, KokkosBatched::Uplo::Upper,
                KokkosBatched::Trans::NoTranspose, KokkosBatched::Diag::NonUnit,
                KokkosBatched::Algo::Level3::Unblocked>::invoke(member, 1.0, tmp, x);
      member.team_barrier();
    }
    applyD(member, x, d2, dx);
    member.team_barrier();    
  }

  inline void run() {
    std::string name("KokkosBatched::Test::StaticPivoting");
    Kokkos::TeamPolicy<DeviceType> policy(ceil(1.*_A.extent(0) / _N_team), Kokkos::AUTO(), Kokkos::AUTO());
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

int main(int argc, char *argv[]) {
  Kokkos::initialize();
  {
    using layout = Kokkos::LayoutLeft;

    using AViewType  = Kokkos::View<double ***, layout, exec_space>;
    using XYViewType = Kokkos::View<double **, layout, exec_space>;
    using IntViewType = Kokkos::View<int **, layout, exec_space>;

    int N = 1;
    int n = 10;

    AViewType A("A", N, n, n);
    AViewType PDAD("PDAD", N, n, n);
    AViewType tmp("tmp", N, n, n+2);
    XYViewType X("X", N, n);
    XYViewType Y("Y", N, n);
    XYViewType PDY("PDY", N, n);
    XYViewType DX("DX", N, n);

    IntViewType P("P", N, n);
    XYViewType D1("D1", N, n);
    XYViewType D2("D2", N, n);

    create_saddle_point_matrices(A, Y);

    const int N_team = 1;

    Functor_TestStaticPivoting<exec_space, AViewType, XYViewType, IntViewType>(A, PDAD, tmp, D1, D2, P, X, DX, Y, PDY, N_team).run();

    write3DArrayToMM("A.mm", A);
    write3DArrayToMM("PDAD.mm", PDAD);
    write3DArrayToMM("tmp.mm", tmp);
    write2DArrayToMM("rhs.mm", Y);
    write2DArrayToMM("PDY.mm", PDY);
    write2DArrayToMM("solution.mm", X);
    write2DArrayToMM("DX.mm", DX);
    write2DArrayToMM("D1.mm", D1);
    write2DArrayToMM("D2.mm", D2);
    write2DArrayToMM("P.mm", P);
  }
  Kokkos::finalize();
}

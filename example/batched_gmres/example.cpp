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

template <typename DeviceType, typename ValuesViewType, typename IntView,
          typename VectorViewType, typename KrylovHandleType, bool UsePrec>
struct Functor_TestBatchedTeamVectorGMRES {
  const ValuesViewType _D;
  const ValuesViewType _diag;
  const IntView _r;
  const IntView _c;
  const VectorViewType _X;
  const VectorViewType _B;
  const int _N_team, _team_size, _vector_length;
  const int _N_iteration;
  const double _tol;
  const int _ortho_strategy;
  const int _scratch_pad_level;
  KrylovHandleType _handle;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorGMRES(const ValuesViewType &D, const IntView &r,
                                  const IntView &c, const VectorViewType &X,
                                  const VectorViewType &B, const int N_team,
                                  const int team_size, const int vector_length,
                                  const int N_iteration, const double tol,
                                  const int ortho_strategy,
                                  const int scratch_pad_level, KrylovHandleType &handle)
      : _D(D), _r(r), _c(c), _X(X), _B(B), _N_team(N_team), _team_size(team_size), _vector_length(vector_length),
      _N_iteration(N_iteration), _tol(tol), _ortho_strategy(ortho_strategy), _scratch_pad_level(scratch_pad_level),
      _handle(handle) {
  }

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorGMRES(const ValuesViewType &D, const ValuesViewType &diag, const IntView &r,
                                  const IntView &c, const VectorViewType &X,
                                  const VectorViewType &B, const int N_team,
                                  const int team_size, const int vector_length,
                                  const int N_iteration, const double tol,
                                  int ortho_strategy,
                                  const int scratch_pad_level, KrylovHandleType &handle)
      : _D(D), _diag(diag), _r(r), _c(c), _X(X), _B(B), _N_team(N_team), _team_size(team_size), _vector_length(vector_length),
      _N_iteration(N_iteration), _tol(tol), _ortho_strategy(ortho_strategy), _scratch_pad_level(scratch_pad_level),
      _handle(handle) {
  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType &member) const {
    const int first_matrix = static_cast<int>(member.league_rank()) * _N_team;
    const int N            = _D.extent(0);
    const int last_matrix =
        (static_cast<int>(member.league_rank() + 1) * _N_team < N
             ? static_cast<int>(member.league_rank() + 1) * _N_team
             : N);
    using TeamVectorCopy1D = KokkosBatched::TeamVectorCopy<MemberType, KokkosBatched::Trans::NoTranspose, 1>;

    auto d = Kokkos::subview(_D, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);
    auto x = Kokkos::subview(_X, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);
    auto b = Kokkos::subview(_B, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);

    using ScratchPadIntViewType = Kokkos::View<
        typename IntView::non_const_value_type*,
        typename IntView::array_layout,
        typename IntView::execution_space::scratch_memory_space>;
    using ScratchPadValuesViewType = Kokkos::View<
        typename ValuesViewType::non_const_value_type**,
        typename ValuesViewType::array_layout,
        typename ValuesViewType::execution_space::scratch_memory_space>;

    using Operator = KokkosBatched::CrsMatrix<ValuesViewType, ScratchPadIntViewType>;

    ScratchPadIntViewType tmp_1D_int(member.team_scratch(0), _r.extent(0) + _c.extent(0));

    auto r = Kokkos::subview(tmp_1D_int, Kokkos::make_pair(0, (int) _r.extent(0)));
    auto c = Kokkos::subview(tmp_1D_int, Kokkos::make_pair((int) _r.extent(0), (int) tmp_1D_int.extent(0)));

    TeamVectorCopy1D::invoke(member, _r, r);
    TeamVectorCopy1D::invoke(member, _c, c);
    Operator A(d, r, c);

    if (UsePrec) {
      ScratchPadValuesViewType diag(member.team_scratch(0), last_matrix-first_matrix, _diag.extent(1));
      using PrecOperator = KokkosBatched::JacobiPrec<ScratchPadValuesViewType>;

      KokkosBatched::TeamVectorCopy<MemberType>::invoke(member, Kokkos::subview(_diag, Kokkos::make_pair(first_matrix, last_matrix), Kokkos::ALL), diag);
      PrecOperator P(diag);
      P.setComputedInverse();

      KokkosBatched::TeamVectorGMRES<MemberType>::template invoke<Operator,
                                                              VectorViewType, PrecOperator, KrylovHandleType>(
          member, A, b, x, P, _handle);
    }
    else {
      KokkosBatched::TeamVectorGMRES<MemberType>::template invoke<Operator,
                                                              VectorViewType>(
          member, A, b, x, _handle);
    }   
  }

  inline double run() {
    typedef typename ValuesViewType::value_type value_type;
    std::string name("KokkosBatched::Test::TeamVectorGMRES");
    Kokkos::Timer timer;
    Kokkos::Profiling::pushRegion(name.c_str());

    Kokkos::TeamPolicy<DeviceType> auto_policy(ceil(1.*_D.extent(0) / _N_team), Kokkos::AUTO(), Kokkos::AUTO());
    Kokkos::TeamPolicy<DeviceType> tuned_policy(ceil(1.*_D.extent(0) / _N_team), _team_size, _vector_length);
    Kokkos::TeamPolicy<DeviceType> policy;

    if (_team_size < 1)
      policy = auto_policy;
    else
      policy = tuned_policy;

    _handle.set_max_iteration(_N_iteration);
    _handle.set_tolerance(_tol);
    _handle.set_ortho_strategy(_ortho_strategy);
    _handle.set_scratch_pad_level(_scratch_pad_level);
    _handle.set_compute_last_residual(true);

    int maximum_iteration = _handle.get_max_iteration();

    using ScalarType = typename ValuesViewType::non_const_value_type;
    using Layout     = typename ValuesViewType::array_layout;
    using EXSP       = typename ValuesViewType::execution_space;

    using MagnitudeType =
          typename Kokkos::Details::ArithTraits<ScalarType>::mag_type;

    using ViewType1D = Kokkos::View<MagnitudeType *, Layout, EXSP>;
    using ViewType2D = Kokkos::View<ScalarType **, Layout, EXSP>;
    using ViewType3D = Kokkos::View<ScalarType ***, Layout, EXSP>;

    size_t bytes_1D = ViewType2D::shmem_size(_N_team, 1);
    size_t bytes_row_ptr = IntView::shmem_size(_r.extent(0));
    size_t bytes_col_idc = IntView::shmem_size(_c.extent(0));
    size_t bytes_2D_1 = ViewType2D::shmem_size(_N_team, _X.extent(1));
    size_t bytes_2D_2 = ViewType2D::shmem_size(_N_team, maximum_iteration+1);
    size_t bytes_3D_1 = ViewType3D::shmem_size(_N_team, _X.extent(1), maximum_iteration);
    size_t bytes_3D_2 = ViewType3D::shmem_size(_N_team, maximum_iteration+1, maximum_iteration);
    size_t bytes_3D_3 = ViewType3D::shmem_size(_N_team, 2, maximum_iteration);


    size_t bytes_int = bytes_row_ptr + bytes_col_idc;
    size_t bytes_diag = bytes_2D_1;
    size_t bytes_tmp = 2 * bytes_2D_1 + 2 * bytes_1D + bytes_2D_2;

    policy.set_scratch_size(0, Kokkos::PerTeam(bytes_tmp + bytes_diag + bytes_int));

    exec_space().fence();
    timer.reset();
    Kokkos::parallel_for(name.c_str(), policy, *this);
    exec_space().fence();
    double sec = timer.seconds();

    return sec;
  }
};

int main(int argc, char *argv[]) {
  Kokkos::initialize();
  {
    using layout = Kokkos::LayoutLeft;

    using IntView          = Kokkos::View<int *, layout, exec_space>;
    using AMatrixValueView = Kokkos::View<double **, layout, exec_space>;
    using XYType           = Kokkos::View<double **, layout, exec_space>;

    bool read_data = false;

    std::string name_A = "A.mm";
    std::string name_B = "B.mm";

    int N, Blk, nnz, ncols;
    if (read_data)
      readSizesFromMM(name_A, Blk, ncols, nnz, N);
    else {
      Blk = 10;
      N = 100;
      nnz = (Blk - 2) * 3 + 2 * 2;
    }

    IntView rowOffsets("rowOffsets", Blk + 1);
    IntView colIndices("colIndices", nnz);
    AMatrixValueView values("values", N, nnz);
    AMatrixValueView diag("diag", N, Blk);
    XYType x("x", N, Blk);
    XYType y("y", N, Blk);

    printf("N = %d, Blk = %d, nnz = %d\n", N, Blk, nnz);
    
    if(read_data) {
      readCRSFromMM(name_A, values, rowOffsets, colIndices);
      readArrayFromMM(name_B, y);
    }
    else{
      create_tridiagonal_batched_matrices(nnz, Blk, N, rowOffsets, colIndices, values, x, y);

      // Replace y by ones:
      Kokkos::deep_copy(y, 1.);

      // Replace x by zeros:
      // Kokkos::deep_copy(x, 0.);
    }
    getInvDiagFromCRS(values, rowOffsets, colIndices, diag);

    using ScalarType = typename AMatrixValueView::non_const_value_type;
    using Layout     = typename AMatrixValueView::array_layout;
    using EXSP       = typename AMatrixValueView::execution_space;

    using MagnitudeType =
        typename Kokkos::Details::ArithTraits<ScalarType>::mag_type;
    using NormViewType = Kokkos::View<MagnitudeType *, Layout, EXSP>;
    
    using Norm2DViewType = Kokkos::View<MagnitudeType **, Layout, EXSP>;
    using Scalar3DViewType = Kokkos::View<ScalarType ***, Layout, EXSP>;
    using IntViewType = Kokkos::View<int*, Layout, EXSP>;

    using KrylovHandleType = KokkosBatched::KrylovHandle<Norm2DViewType, IntViewType, Scalar3DViewType>;

    const int N_team = 10;
    const int n_iterations = 10;

    const int team_size = -1;
    const int vector_length = -1;
    const double tol = 1e-12;
    const int ortho_strategy = 0;

    KrylovHandleType handle(N, N_team, n_iterations, true);
    handle.Arnoldi_view = Scalar3DViewType("", N, n_iterations, Blk+n_iterations+3);

    writeArrayToMM("initial_guess.mm", x);
    writeArrayToMM("rhs.mm", y);

    double time = Functor_TestBatchedTeamVectorGMRES<exec_space, AMatrixValueView, IntView, XYType, KrylovHandleType, true>
                  (values, diag, rowOffsets, colIndices, x, y, N_team, team_size, vector_length, n_iterations, tol, ortho_strategy, 0, handle).run();

    printf("times = %f secondes\n", time);

    for (size_t i = 0; i < N; ++i) {
      if (handle.is_converged_host(i)) {
        std::cout << "System " << i << " converged in " << handle.get_iteration_host(i) << " iterations, the initial absolute norm of the residual was " << handle.get_norm_host(i, 0) << " and is now " << handle.get_last_norm_host(i) << std::endl;
      }
      else {
        std::cout << "System " << i << " did not converge in " << handle.get_max_iteration() << " iterations, the initial absolute norm of the residual was " << handle.get_norm_host(i, 0) << " and is now " << handle.get_last_norm_host(i) << std::endl;
      }
    }
    if (handle.is_converged_host())
      std::cout << "All the systems have converged." << std::endl;
    else
      std::cout << "There is at least one system that did not convegre." << std::endl;

    writeArrayToMM("solution.mm", x);
    writeArrayToMM("convergence.mm", handle.residual_norms);
    writeCRSToMM("newA.mm", values, rowOffsets, colIndices);
  }
  Kokkos::finalize();
}


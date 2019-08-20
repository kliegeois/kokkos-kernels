/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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
*/

#ifndef KOKKOSSPARSE_IMPL_SPTRSV_SOLVE_HPP_
#define KOKKOSSPARSE_IMPL_SPTRSV_SOLVE_HPP_

/// \file KokkosSparse_impl_sptrsv.hpp
/// \brief Implementation(s) of sparse triangular solve.

#include <KokkosKernels_config.h>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_sptrsv_handle.hpp>

#if defined(KOKKOSKERNELS_ENABLE_TPL_CHOLMOD)
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas2_team_gemv.hpp>

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_Trsv_Decl.hpp"
#include "KokkosBatched_Trsv_Serial_Impl.hpp"

#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Gemv_Team_Impl.hpp"
#include "KokkosBatched_Gemv_Serial_Impl.hpp"
#endif


//#define LVL_OUTPUT_INFO

namespace KokkosSparse {
namespace Impl {
namespace Experimental {


struct UnsortedTag {};

template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct LowerTriLvlSchedRPSolverFunctor
{
  typedef typename EntriesType::non_const_value_type lno_t;
  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  LowerTriLvlSchedRPSolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, NGBLType nodes_grouped_by_level_ ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    // Assuming indices are sorted per row, diag entry is final index in the list

    auto soffset = row_map(rowid);
    auto eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);

    for ( auto ptr = soffset; ptr < eoffset; ++ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        lhs(rowid) = rhs_rowid/val;
      }
    } // end for ptr
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    auto soffset = row_map(rowid);
    auto eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);
    auto diag = -1;

    for ( auto ptr = soffset; ptr < eoffset; ++ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        diag = ptr;
      }
    } // end for ptr
    lhs(rowid) = rhs_rowid/values(diag);
  }
};


template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct LowerTriLvlSchedTP1SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;


  LowerTriLvlSchedTP1SolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_team = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

      Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
      }, diff );

        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this; can also use Kokkos::single
        if ( my_team == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at eoffset - 1
          lhs(rowid) = (rhs_rowid+diff)/values(eoffset-1);
        }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_team = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

        auto diag = -1;

        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
          else {
            diag = ptr;
          }
        }, diff );

        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this; can also use Kokkos::single
        if ( my_team == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at eoffset - 1
          lhs(rowid) = (rhs_rowid+diff)/values(diag);
        }
  }
};

// FIXME CUDA: This algorithm not working with all integral type combos
// In any case, this serves as a skeleton for 3-level hierarchical parallelism for alg dev
template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct LowerTriLvlSchedTP2SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;


  LowerTriLvlSchedTP2SolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {

            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at eoffset - 1
            lhs(rowid) = (rhs_rowid+diff)/values(eoffset-1);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {
            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            auto diag = -1;
            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
              else {
                diag = ptr;
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at eoffset - 1
            lhs(rowid) = (rhs_rowid+diff)/values(diag);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }
};

#if defined(KOKKOSKERNELS_ENABLE_TPL_CHOLMOD)
template <class ColptrView, class RowindType, class ValuesType, class LHSType, class NGBLType>
struct LowerTriCholmodFunctor
{
  typedef typename LHSType::execution_space execution_space;
  typedef typename execution_space::memory_space memory_space;

  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename ValuesType::non_const_value_type scalar_t;

  typedef typename Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space, Kokkos::MemoryUnmanaged> supernode_view_t;
  typedef typename Kokkos::View<scalar_t*,                      memory_space, Kokkos::MemoryUnmanaged> vector_view_t;

  typedef Kokkos::View<int*, memory_space>  work_offset_t;
  typedef typename Kokkos::View<scalar_t*,  memory_space> WorkspaceType;

  typedef Kokkos::pair<int,int> range_type;

  const int *supercols;
  ColptrView colptr;
  RowindType rowind;
  ValuesType values;

  LHSType X;

  WorkspaceType work; // needed with gemv for update&scatter
  work_offset_t work_offset;

  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;

  // constructor
  LowerTriCholmodFunctor (// supernode info
                          const int *supercols_,
                          // L in CSR
                          const ColptrView  &colptr_,
                          const RowindType &rowind_,
                          const ValuesType &values_,
                          // right-hand-side (input), solution (output)
                          LHSType &X_,
                          // workspace
                          WorkspaceType work_,
                          work_offset_t &work_offset_,
                          //
                          const NGBLType &nodes_grouped_by_level_,
                          long  node_count_,
                          long  node_groups_ = 0) :
    supercols(supercols_),
    colptr(colptr_), rowind(rowind_), values(values_),
    X(X_), work(work_), work_offset(work_offset_),
    nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */
    const int league_rank = team.league_rank(); // batch id
    const int team_size = team.team_size ();
    const int team_rank = team.team_rank ();
    scalar_t zero = 0.0;
    scalar_t one  = 1.0;

    // raw csr for L
    scalar_t *dataL = const_cast<scalar_t*> (values.data ());

    auto s = nodes_grouped_by_level (node_count + league_rank);

    // extract diagonal and off-diagonal blocks
    int j1 = supercols[s];
    int j2 = supercols[s+1];
    int nscol = j2 - j1 ;      // number of columns in the s-th supernode column

    int i1 = colptr (j1);
    int i2 = colptr (j1+1);
    int nsrow  = i2 - i1;      // "total" number of rows in all the supernodes (diagonal+off-diagonal)

    Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space, Kokkos::MemoryUnmanaged> viewL (&dataL[i1], nsrow, nscol);
    auto Ajj = Kokkos::subview (viewL, range_type (0, nscol), Kokkos::ALL ());
    auto Aij = Kokkos::subview (viewL, range_type (nscol, nsrow), Kokkos::ALL ());

    // extract part of the solution, corresponding to the diagonal block
    scalar_t *dataX = const_cast<scalar_t*> (X.data ());
    Kokkos::View<scalar_t*, memory_space, Kokkos::MemoryUnmanaged> Xj(&dataX[j1], nscol);

    // workspace
    int nsrow2 = nsrow - nscol;  // "total" number of rows in all the off-diagonal supernodes
    int workoffset = work_offset (s);
    scalar_t *dataW = const_cast<scalar_t*> (work.data ());
    Kokkos::View<scalar_t*,  memory_space> Y (&dataW[workoffset],         nscol);  // needed for gemv instead of trmv/trsv
    Kokkos::View<scalar_t*,  memory_space> Z (&dataW[workoffset+nscol], nsrow2);  // needed with gemv for update&scatter

    //if (team_rank == 0) {
    //  printf( " >> LowerTriCholmodFunctor (league=%d, team=%d/%d <<\n",league_rank,team_rank,team_size );
    //  printf( " s = %d, nscol=%d, nsrows=%d (j1=%d, j2=%d), (i1=%d, i2=%d), work_offset=%d\n",s, nscol,nsrow, j1,j2, i1,i2, workoffset );
    //}

    /* TRSM with diagonal block */
    #if defined(CHOLMOD_BATCHED_TRSV)
    if (team_rank == 0) {
      KokkosBatched::SerialTrsv<KokkosBatched::Uplo::Lower,
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Diag::NonUnit,
                                KokkosBatched::Algo::Trsv::Blocked>
        ::invoke(one, Ajj, Xj);
    }
    #else
    if (nscol == 1) {
      if (team_rank == 0) {
        Xj(0) *= Ajj(0, 0);
      }
    } else {
      for (int ii = team_rank; ii < nscol; ii += team_size) {
        Y(ii) = Xj(ii);
      }
      team.team_barrier ();
      KokkosBatched::TeamGemv<member_type,
                              KokkosBatched::Trans::NoTranspose,
                              KokkosBatched::Algo::Gemv::Unblocked>
        ::invoke(team, one, Ajj, Y, zero, Xj);
      /*KokkosBlas::Experimental::
      gemv (team, 'N',
            one,  Ajj,
                  Y,
            zero, Xj);*/
    }
    team.team_barrier ();
    #endif
    //for (int ii=0; ii < nscol; ii++) printf( "%d %e\n",j1+ii,X(j1+ii));
    //printf( "\n" );

    /* GEMM to update with off diagonal blocks */
    if (nsrow2 > 0) {
      #if defined(CHOLMOD_BATCHED_GEMV)
      KokkosBatched::SerialGemv<KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gemv::Blocked>
        ::invoke(-one, Aij, Xj, zero, Z);
      #else
      KokkosBlas::Experimental::
      gemv(team, 'N',
           -one, Aij,
                 Xj,
           zero, Z);
      team.team_barrier();
      #endif

      /* scatter vectors back into X */
      int ps2 = i1 + nscol ;     // offset into rowind 
      Kokkos::View<scalar_t*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic> > Xatomic(X.data(), X.extent(0));
      for (int ii = team_rank; ii < nsrow2; ii += team_size) {
        int i = rowind (ps2 + ii);
        //printf( " X(%d) = %e - %e = %e\n",i,X(i),Z(ii),X(i)-Z(ii) );
        Xatomic (i) += Z (ii);
      }
    }
    team.team_barrier();
  }
};

template <class ColptrType, class RowindType, class ValuesType, class LHSType, class NGBLType>
struct UpperTriCholmodFunctor
{
  typedef typename LHSType::execution_space execution_space;
  typedef typename execution_space::memory_space memory_space;

  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename ValuesType::non_const_value_type scalar_t;

  typedef typename Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space, Kokkos::MemoryUnmanaged> supernode_view_t;
  typedef typename Kokkos::View<scalar_t*,                      memory_space, Kokkos::MemoryUnmanaged> vector_view_t;

  typedef Kokkos::View<int*, memory_space>  supercols_t;
  typedef typename Kokkos::View<scalar_t*, memory_space> WorkspaceType;

  typedef Kokkos::pair<int,int> range_type;

  const int *supercols;
  ColptrType colptr;
  RowindType rowind;
  ValuesType values;

  LHSType X;

  WorkspaceType work; // needed with gemv for update&scatter
  supercols_t work_offset;

  int level;
  supercols_t kernel_type;

  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;

  // constructor
  UpperTriCholmodFunctor (// supernode info
                          const int *supercols_,
                          // L in CSR
                          const ColptrType &colptr_,
                          const RowindType &rowind_,
                          const ValuesType &values_,
                          // right-hand-side (input), solution (output)
                          LHSType &X_,
                          //
                          int level_,
                          supercols_t &kernel_type_,
                          // workspace
                          WorkspaceType &work_,
                          supercols_t &work_offset_,
                          //
                          const NGBLType &nodes_grouped_by_level_,
                          long  node_count_,
                          long  node_groups_ = 0) :
    supercols(supercols_),
    colptr(colptr_), rowind(rowind_), values(values_),
    level(level_), kernel_type(kernel_type_),
    X(X_), work(work_), work_offset(work_offset_),
    nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */
    const int league_rank = team.league_rank(); // batch id
    const int team_size = team.team_size ();
    const int team_rank = team.team_rank ();

    #if !defined(CHOLMOD_BATCHED_TRSV)
    scalar_t zero = 0.0;
    #endif
    scalar_t one  = 1.0;

    auto s = nodes_grouped_by_level (node_count + league_rank);

    //printf( " Upper: node_count=%ld, s=%d\n",node_count,s );
    int j1 = supercols[s];
    int j2 = supercols[s+1];
    int nscol = j2 - j1;      // number of columns in the s-th supernode column

    int i1 = colptr (j1);
    int i2 = colptr (j1+1);
    int nsrow  = i2 - i1 ;    // "total" number of rows in all the supernodes (diagonal+off-diagonal)

    // extract diagonal and off-diagonal blocks of L
    scalar_t *dataL = const_cast<scalar_t*> (values.data ());
    Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space, Kokkos::MemoryUnmanaged> viewL (&dataL[i1], nsrow, nscol);
    auto Ajj = Kokkos::subview (viewL, range_type (0, nscol), Kokkos::ALL ());
    auto Aij = Kokkos::subview (viewL, range_type (nscol, nsrow), Kokkos::ALL ());

    // extract part of solution, corresponding to the diagonal block
    scalar_t *dataX = const_cast<scalar_t*> (X.data ());
    Kokkos::View<scalar_t*, memory_space, Kokkos::MemoryUnmanaged> Xj (&dataX[j1], nscol);

    // workspaces
    int nsrow2 = nsrow - nscol ;  // "total" number of rows in all the off-diagonal supernodes
    int workoffset = work_offset (s);
    scalar_t *dataW = const_cast<scalar_t*> (work.data ());
    Kokkos::View<scalar_t*,  memory_space> Y (&dataW[workoffset],        nscol);  // needed for gemv instead of trmv/trsv
    Kokkos::View<scalar_t*,  memory_space> Z (&dataW[workoffset+nscol], nsrow2);  // needed with gemv for update&scatter

    //if (team_rank == 0) {
    //  printf( " >> UpperTriCholmodFunctor (league=%d, team=%d/%d <<\n",league_rank,team_rank,team_size );
    //  printf( " s = %d, nscol=%d, nsrows=%d (j1=%d, j2=%d), (i1=%d, i2=%d), work_offset=%d\n",s, nscol,nsrow, j1,j2, i1,i2, workoffset );
    //}

    if (nsrow2 > 0) {
      /* gather vector into W */
      int ps2 = i1 + nscol;     // offset into rowind 
      for (int ii = team_rank; ii < nsrow2 ; ii += team_size) {
        int i = rowind (ps2 + ii);
        Z (ii) = X (i);
      }

      /* GEMM to update with off diagonal blocks */
      #if defined(CHOLMOD_BATCHED_GEMV)
      KokkosBatched::SerialGemv<KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Gemv::Blocked>
        ::invoke(-one, Aij, Z, one, Xj);
      #else
      team.team_barrier();
      KokkosBlas::Experimental::
      gemv(team, 'T',
          -one, Aij,
                Z,
           one, Xj);
      team.team_barrier();
      #endif
    }

    //printf( " nscol=%d, nsrows=%d (j1=%d, j2=%d), (i1=%d, i2=%d), psx=%d\n",nscol,nsrow, j1,j2, i1,i2, psx );
    /* TRSM with diagonal block */
    #if defined(CHOLMOD_BATCHED_TRSV)
    if (team_rank == 0) {
     KokkosBatched::SerialTrsv<KokkosBatched::Uplo::Lower,
                               KokkosBatched::Trans::Transpose,
                               KokkosBatched::Diag::NonUnit,
                               KokkosBatched::Algo::Trsv::Blocked>
       ::invoke(one, Ajj, Xj);
    }
    #else
    if (nscol == 1) {
      Xj (0) *= Ajj(0, 0);
    } else {
      for (int ii = team_rank; ii < nscol; ii += team_size) {
        Y (ii) = Xj (ii);
      }
      team.team_barrier();
      if (kernel_type (level) == 0) {
        KokkosBatched::TeamGemv<member_type,
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Gemv::Unblocked>
          ::invoke(team, one, Ajj, Y, zero, Xj);
      } else {
        KokkosBlas::Experimental::
        gemv(team, 'T',
             one,  Ajj,
                   Y,
             zero, Xj);
      }
    }
    #endif
    team.team_barrier();
    //for (int ii=0; ii < nscol; ii++) printf( "%d %e\n",j1+ii,X(j1+ii));
    //printf( "\n" );
  }
};
#endif

template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct UpperTriLvlSchedRPSolverFunctor
{
  typedef typename EntriesType::non_const_value_type lno_t;
  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;


  UpperTriLvlSchedRPSolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_ ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    // Assuming indices are sorted per row, diag entry is final index in the list
    long soffset = row_map(rowid);
    long eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);
    for ( long ptr = eoffset-1; ptr >= soffset; --ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        lhs(rowid) = rhs_rowid/val;
      }
    } // end for ptr
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    long soffset = row_map(rowid);
    long eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);
    auto diag = -1;
    for ( long ptr = eoffset-1; ptr >= soffset; --ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        diag = ptr;
      }
    } // end for ptr
    lhs(rowid) = rhs_rowid/values(diag);
  }

};

template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct UpperTriLvlSchedTP1SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;


  UpperTriLvlSchedTP1SolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0 ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_team = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
        }, diff );

        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this, also can use Kokkos::single
        if ( my_team == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at start offset
          lhs(rowid) = (rhs_rowid+diff)/values(soffset);
        }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_team = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

        auto diag = -1;

        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
          else {
            diag = ptr;
          }
        }, diff );
        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this, also can use Kokkos::single
        if ( my_team == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at start offset
          lhs(rowid) = (rhs_rowid+diff)/values(diag);
        }
  }

};


// FIXME CUDA: This algorithm not working with all integral type combos
// In any case, this serves as a skeleton for 3-level hierarchical parallelism for alg dev
template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct UpperTriLvlSchedTP2SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count;
  long node_groups;


  UpperTriLvlSchedTP2SolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0 ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {

            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at start offset
            lhs(rowid) = (rhs_rowid+diff)/values(soffset);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team ) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {
            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            auto diag = -1;
            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
              else {
                diag = ptr;
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at start offset
            lhs(rowid) = (rhs_rowid+diff)/values(diag);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }

};


template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void lower_tri_solve( TriSolveHandle & thandle, const RowMapType row_map, const EntriesType entries, const ValuesType values, const RHSType & rhs, LHSType &lhs) {

  typedef typename TriSolveHandle::execution_space execution_space;
  typedef typename execution_space::memory_space memory_space;

  typedef typename TriSolveHandle::size_type size_type;
  typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;

  typedef typename ValuesType::non_const_value_type scalar_t;
  typedef typename TriSolveHandle::signed_integral_t signed_integral_t;

  typedef typename TriSolveHandle::supercols_t work_offset_t;

  auto nlevels = thandle.get_num_levels();
  // Keep this a host View, create device version and copy to back to host during scheduling
  auto nodes_per_level = thandle.get_nodes_per_level();
  auto hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
  Kokkos::deep_copy(hnodes_per_level, nodes_per_level);  

  auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

#if defined(KOKKOSKERNELS_ENABLE_TPL_CHOLMOD)
  auto nodes_grouped_by_level_host = Kokkos::create_mirror_view (nodes_grouped_by_level);
  Kokkos::deep_copy (nodes_grouped_by_level_host, nodes_grouped_by_level);

  auto rowmap_host = Kokkos::create_mirror_view (row_map);
  Kokkos::deep_copy (rowmap_host, row_map);

  signed_integral_t lwork = thandle.get_workspace_size ();
  Kokkos::View<scalar_t*, memory_space> work ("work", lwork);

  work_offset_t work_offset = thandle.get_work_offset ();
#endif

  size_type node_count = 0;

  // This must stay serial; would be nice to try out Cuda's graph stuff to reduce kernel launch overhead
  for ( size_type lvl = 0; lvl < nlevels; ++lvl ) {
    size_type lvl_nodes = hnodes_per_level(lvl);

    if ( lvl_nodes != 0 ) {
      if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP ) {
        Kokkos::parallel_for( "parfor_fixed_lvl", Kokkos::RangePolicy<execution_space>( node_count, node_count+lvl_nodes ), LowerTriLvlSchedRPSolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> (row_map, entries, values, lhs, rhs, nodes_grouped_by_level) );
      }
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1 ) {
        typedef Kokkos::TeamPolicy<execution_space> policy_type;
        int team_size = thandle.get_team_size();

        LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
        if ( team_size == -1 )
          Kokkos::parallel_for("parfor_l_team", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
        else
          Kokkos::parallel_for("parfor_l_team", policy_type( lvl_nodes , team_size ), tstf);
      }
      /*
      // TP2 algorithm has issues with some offset-ordinal combo to be addressed
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2 ) {
        typedef Kokkos::TeamPolicy<execution_space> tvt_policy_type;

        int team_size = thandle.get_team_size();
        if ( team_size == -1 ) {
          team_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 128;
        }
        int vector_size = thandle.get_team_size();
        if ( vector_size == -1 ) {
          vector_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 4;
        }

        // This impl: "chunk" lvl_nodes into node_groups; a league_rank is responsible for processing that many nodes
        //       TeamThreadRange over number of node_groups
        //       To avoid masking threads, 1 thread (team) per node in node_group
        //       ThreadVectorRange responsible for the actual solve computation
        const int node_groups = team_size;

        LowerTriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count, node_groups);
        Kokkos::parallel_for("parfor_u_team_vector", tvt_policy_type( (int)std::ceil((float)lvl_nodes/(float)node_groups) , team_size, vector_size ), tstf);
      } // end elseif
      */
#if defined(KOKKOSKERNELS_ENABLE_TPL_CHOLMOD)
      else if (thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::CHOLMOD_NAIVE ||
               thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::CHOLMOD_ETREE) {
        typedef Kokkos::TeamPolicy<execution_space> policy_type;

        const int* supercols = thandle.get_supercols ();
        LowerTriCholmodFunctor<RowMapType, EntriesType, ValuesType, LHSType, NGBLType> 
          tstf (supercols, row_map, entries, values, lhs, work, work_offset, nodes_grouped_by_level, node_count);

//Kokkos::Timer timer;
//timer.reset();
        #if defined(CHOLMOD_BATCHED_KERNEL)
        Kokkos::parallel_for ("parfor_lsolve_cholmot", policy_type(lvl_nodes , 1), tstf);
        #else
        Kokkos::parallel_for ("parfor_lsolve_cholmot", policy_type(lvl_nodes , Kokkos::AUTO), tstf);
        #endif
//Kokkos::fence();
//std::cout << " > CHOLMOD LowerTri: " << lvl << " " << timer.seconds() << std::endl;
      }
#endif
      node_count += lvl_nodes;

    } // end if
  } // end for lvl

} // end lower_tri_solve


template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void upper_tri_solve( TriSolveHandle & thandle, const RowMapType row_map, const EntriesType entries, const ValuesType values, const RHSType & rhs, LHSType &lhs) {

  typedef typename TriSolveHandle::execution_space execution_space;
  typedef typename execution_space::memory_space memory_space;

  typedef typename TriSolveHandle::size_type size_type;
  typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;

  typedef typename ValuesType::non_const_value_type scalar_t;
  typedef typename TriSolveHandle::signed_integral_t signed_integral_t;

  typedef typename TriSolveHandle::supercols_t supercols_t;

  auto nlevels = thandle.get_num_levels();
  // Keep this a host View, create device version and copy to back to host during scheduling
  auto nodes_per_level = thandle.get_nodes_per_level();
  auto hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
  Kokkos::deep_copy(hnodes_per_level, nodes_per_level);

  auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

#if defined(KOKKOSKERNELS_ENABLE_TPL_CHOLMOD)
  auto nodes_grouped_by_level_host = Kokkos::create_mirror_view (nodes_grouped_by_level);
  Kokkos::deep_copy (nodes_grouped_by_level_host, nodes_grouped_by_level);

  auto rowmap_host = Kokkos::create_mirror_view (row_map);
  Kokkos::deep_copy (rowmap_host, row_map);

  signed_integral_t lwork = thandle.get_workspace_size ();
  Kokkos::View<scalar_t*,  memory_space> work ("work", lwork);

  supercols_t kernel_type = thandle.get_kernel_type ();
  supercols_t work_offset = thandle.get_work_offset ();
#endif

  size_type node_count = 0;

  // This must stay serial; would be nice to try out Cuda's graph stuff to reduce kernel launch overhead
  for ( size_type lvl = 0; lvl < nlevels; ++lvl ) {
    size_type lvl_nodes = hnodes_per_level(lvl);

    if ( lvl_nodes != 0 ) {

      if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP ) {
        Kokkos::parallel_for( "parfor_fixed_lvl", Kokkos::RangePolicy<execution_space>( node_count, node_count+lvl_nodes ), UpperTriLvlSchedRPSolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> (row_map, entries, values, lhs, rhs, nodes_grouped_by_level) );
      }
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1 ) {
        typedef Kokkos::TeamPolicy<execution_space> policy_type;

        int team_size = thandle.get_team_size();

        UpperTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
        if ( team_size == -1 )
          Kokkos::parallel_for("parfor_u_team", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
        else
          Kokkos::parallel_for("parfor_u_team", policy_type( lvl_nodes , team_size ), tstf);
      }
      /*
      // TP2 algorithm has issues with some offset-ordinal combo to be addressed
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2 ) {
        typedef Kokkos::TeamPolicy<execution_space> tvt_policy_type;

        int team_size = thandle.get_team_size();
        if ( team_size == -1 ) {
          team_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 128;
        }
        int vector_size = thandle.get_team_size();
        if ( vector_size == -1 ) {
          vector_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 4;
        }

        // This impl: "chunk" lvl_nodes into node_groups; a league_rank is responsible for processing that many nodes
        //       TeamThreadRange over number of node_groups
        //       To avoid masking threads, 1 thread (team) per node in node_group
        //       ThreadVectorRange responsible for the actual solve computation
        const int node_groups = team_size;

        UpperTriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count, node_groups);
        Kokkos::parallel_for("parfor_u_team_vector", tvt_policy_type( (int)std::ceil((float)lvl_nodes/(float)node_groups) , team_size, vector_size ), tstf);
      } // end elseif
      */
#if defined(KOKKOSKERNELS_ENABLE_TPL_CHOLMOD)
      else if (thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::CHOLMOD_NAIVE ||
               thandle.get_algorithm () == KokkosSparse::Experimental::SPTRSVAlgorithm::CHOLMOD_ETREE) {
        typedef Kokkos::TeamPolicy<execution_space> policy_type;

        const int* supercols = thandle.get_supercols ();
        UpperTriCholmodFunctor<RowMapType, EntriesType, ValuesType, LHSType, NGBLType> 
          tstf (supercols, row_map, entries, values, lhs, lvl, kernel_type, work, work_offset, nodes_grouped_by_level, node_count);

//Kokkos::Timer timer;
//timer.reset();
        #if defined(CHOLMOD_BATCHED_KERNEL)
        Kokkos::parallel_for ("parfor_usolve_cholmot", policy_type (lvl_nodes , 1), tstf);
        #else
        Kokkos::parallel_for ("parfor_usolve_cholmot", policy_type (lvl_nodes , Kokkos::AUTO), tstf);
        #endif
//Kokkos::fence();
//std::cout << " > CHOLMOD UpperTri: " << lvl << " " << timer.seconds() << std::endl;
      }
#endif
      node_count += lvl_nodes;

    } // end if
  } // end for lvl

} // end upper_tri_solve


} // namespace Experimental
} // namespace Impl
} // namespace KokkosSparse

#endif

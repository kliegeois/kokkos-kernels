/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
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
*/

#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

#ifndef __KOKKOSBATCHED_KRYLOV_HANDLE_HPP__
#define __KOKKOSBATCHED_KRYLOV_HANDLE_HPP__
//#define VERBOSE

namespace KokkosBatched {

/// \brief KrylovHandle
///
/// \tparam scalar_type: Scalar type of the linear solver

template <class NormViewType, class IntViewType>
class KrylovHandle {
 public:
  using norm_type =
      typename NormViewType::non_const_value_type;


 private:
  NormViewType residual_norms;
  IntViewType iteration_numbers;
  norm_type tolerance;
  int max_iteration;
  int batched_size;
  int N_team;

 public:
  KOKKOS_INLINE_FUNCTION
  KrylovHandle(int _batched_size, int _N_team, int _max_iteration = 200) : 
  max_iteration(_max_iteration), batched_size(_batched_size), N_team(_N_team) {
    tolerance     = Kokkos::Details::ArithTraits<norm_type>::epsilon();
    residual_norms = NormViewType("",batched_size, max_iteration);
    iteration_numbers = IntViewType("",batched_size);
  }

  /// \brief set_tolerance
  ///   Set the tolerance of the batched Krylov solver
  ///
  /// \param _tolerance [in]: New tolerance

  KOKKOS_INLINE_FUNCTION
  void set_tolerance(norm_type _tolerance) { tolerance = _tolerance; }

  /// \brief get_tolerance
  ///   Get the tolerance of the batched Krylov solver

  KOKKOS_INLINE_FUNCTION
  norm_type get_tolerance() const { return tolerance; }

  /// \brief set_max_iteration
  ///   Set the maximum number of iterations of the batched Krylov solver
  ///
  /// \param _max_iteration [in]: New maximum number of iterations

  KOKKOS_INLINE_FUNCTION
  void set_max_iteration(int _max_iteration) {
    max_iteration = _max_iteration;
  }

  /// \brief get_max_iteration
  ///   Get the maximum number of iterations of the batched Krylov solver

  KOKKOS_INLINE_FUNCTION
  int get_max_iteration() const { return max_iteration; }

  KOKKOS_INLINE_FUNCTION
  void set_norm(int batched_id, int iteration_id, norm_type norm_i) {
    residual_norms(batched_id, iteration_id) = norm_i;
  }

  KOKKOS_INLINE_FUNCTION
  void set_norm(int team_id, int batched_id, int iteration_id, norm_type norm_i) {
    residual_norms(team_id * N_team + batched_id, iteration_id) = norm_i;
  }

  KOKKOS_INLINE_FUNCTION
  norm_type get_norm(int batched_id, int iteration_id) {
    return residual_norms(batched_id, iteration_id);
  }

  KOKKOS_INLINE_FUNCTION
  void set_iteration(int batched_id, int iteration_id) {
    iteration_numbers(batched_id) = iteration_id;
  }

  KOKKOS_INLINE_FUNCTION
  void set_iteration(int team_id, int batched_id, int iteration_id) {
    iteration_numbers(team_id * N_team + batched_id) = iteration_id;
  }

  KOKKOS_INLINE_FUNCTION
  int get_iteration(int batched_id) {
    return iteration_numbers(batched_id);
  }
};

}  // namespace KokkosBatched

#endif

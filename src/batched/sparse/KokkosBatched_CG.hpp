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
#ifndef __KOKKOSBATCHED_CG_HPP__
#define __KOKKOSBATCHED_CG_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"


/// Batched CG
///
/// 3 implementations are currently provided:
///  * SerialCG,
///  * TeamCG,
///  * TeamVectorCG.
///
/// The naming of those implementations follows the logic of: 
///   Kim, K. (2019). Solving Many Small Matrix Problems using Kokkos and 
///   KokkosKernels (No. SAND2019-4542PE). Sandia National Lab.(SNL-NM),
///   Albuquerque, NM (United States).
///

#include "KokkosBatched_CG_Serial_Impl.hpp"
#include "KokkosBatched_CG_Team_Impl.hpp"
#include "KokkosBatched_CG_TeamVector_Impl.hpp"

namespace KokkosBatched {

  template<typename MemberType,
           typename ArgMode>
  struct CG {
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
      int r_val = 0;
      if (std::is_same<ArgMode,Mode::Serial>::value) {
        r_val = SerialCG::template invoke<ValuesViewType, IntView, VectorViewType, ScalarType>(values, row_ptr, colIndices, B, X, maximum_iteration, tolerance);
      } else if (std::is_same<ArgMode,Mode::Team>::value) {
        r_val = TeamCG<MemberType>::template invoke<ValuesViewType, IntView, VectorViewType, ScalarType>(member, values, row_ptr, colIndices, B, X, maximum_iteration, tolerance);
      } else if (std::is_same<ArgMode,Mode::TeamVector>::value) {
        r_val = TeamVectorCG<MemberType>::template invoke<ValuesViewType, IntView, VectorViewType, ScalarType>(member, values, row_ptr, colIndices, B, X, maximum_iteration, tolerance);
      } 
      return r_val;
    }      
  };

}
#endif

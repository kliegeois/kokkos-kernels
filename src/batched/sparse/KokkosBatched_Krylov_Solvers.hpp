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

#ifndef __KOKKOSBATCHED_KRYLOV_SOLVERS_HPP__
#define __KOKKOSBATCHED_KRYLOV_SOLVERS_HPP__

namespace KokkosBatched {

struct SerialGMRES {
  template <typename OperatorType, typename VectorViewType,
            typename PrecOperatorType, typename KrylovHandleType>
  KOKKOS_INLINE_FUNCTION static int invoke(const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const PrecOperatorType& P,
                                           const KrylovHandleType& handle,
                                           const int GMRES_id);
  template <typename OperatorType, typename VectorViewType,
            typename KrylovHandleType>
  KOKKOS_INLINE_FUNCTION static int invoke(const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const KrylovHandleType& handle);
};

template <typename MemberType>
struct TeamGMRES {
  template <typename OperatorType, typename VectorViewType,
            typename PrecOperatorType, typename KrylovHandleType,
            typename ArnoldiViewType, typename TMPViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const PrecOperatorType& P,
                                           const KrylovHandleType& handle,
                                           const ArnoldiViewType& _ArnoldiView,
                                           const TMPViewType& _TMPView);
  template <typename OperatorType, typename VectorViewType,
            typename PrecOperatorType, typename KrylovHandleType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const PrecOperatorType& P,
                                           const KrylovHandleType& handle);
  template <typename OperatorType, typename VectorViewType,
            typename KrylovHandleType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const KrylovHandleType& handle);
};

template <typename MemberType>
struct TeamVectorGMRES {
  template <typename OperatorType, typename VectorViewType,
            typename PrecOperatorType, typename KrylovHandleType,
            typename ArnoldiViewType, typename TMPViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const PrecOperatorType& P,
                                           const KrylovHandleType& handle,
                                           const ArnoldiViewType& _ArnoldiView,
                                           const TMPViewType& _TMPView);
  template <typename OperatorType, typename VectorViewType,
            typename PrecOperatorType, typename KrylovHandleType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const PrecOperatorType& P,
                                           const KrylovHandleType& handle);
  template <typename OperatorType, typename VectorViewType,
            typename KrylovHandleType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const KrylovHandleType& handle);
};

template <typename MemberType>
struct TeamCG {
  template <typename OperatorType, typename VectorViewType,
            typename KrylovHandleType, typename TMPViewType, typename TMPNormViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const KrylovHandleType& handle,
                                           const TMPViewType& _TMPView,
                                           const TMPNormViewType& _TMPNormView);
  template <typename OperatorType, typename VectorViewType,
            typename KrylovHandleType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const KrylovHandleType& handle);
};

template <typename MemberType>
struct TeamVectorCG {
  template <typename OperatorType, typename VectorViewType,
            typename KrylovHandleType, typename TMPViewType, typename TMPNormViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const KrylovHandleType& handle,
                                           const TMPViewType& _TMPView,
                                           const TMPNormViewType& _TMPNormView);
  template <typename OperatorType, typename VectorViewType,
            typename KrylovHandleType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const OperatorType& A,
                                           const VectorViewType& _B,
                                           const VectorViewType& _X,
                                           const KrylovHandleType& handle);
};

}  // namespace KokkosBatched

#endif

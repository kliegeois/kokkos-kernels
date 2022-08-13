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
#ifndef KOKKOSBLAS2_SERIAL_GEMV_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS2_SERIAL_GEMV_TPL_SPEC_DECL_HPP_

#include "KokkosBlas_util.hpp"
#include "KokkosBatched_Vector.hpp"

#if defined(KOKKOSKERNELS_ENABLE_TPL_MKL)
#include "mkl_version.h"
#if __INTEL_MKL__ >= 2018
#define __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__ 1
#endif
#endif

#ifdef __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__

#include "mkl_compact.h"

namespace KokkosBlas {
namespace Impl {

// Note: using GEMM because there is no GEMV in MKL compact routines

#define __IMPL_KK_MKL_DGEMM_COMPACT(SCALAR, MKL_ROUTINE)                   \
  void kk_mkl_gemm_compact(                                                \
      MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,       \
      MKL_INT m, MKL_INT n, MKL_INT k, SCALAR alpha, const SCALAR *a,      \
      MKL_INT ldap, const SCALAR *b, MKL_INT ldbp, SCALAR beta, SCALAR *c, \
      MKL_INT ldcp, MKL_COMPACT_PACK format, MKL_INT nm) {                 \
    MKL_ROUTINE(layout, transa, transb, m, n, k, alpha, a, ldap, b, ldbp,  \
                beta, c, ldcp, format, nm);                                \
  }

__IMPL_KK_MKL_DGEMM_COMPACT(double, mkl_dgemm_compact)
__IMPL_KK_MKL_DGEMM_COMPACT(float, mkl_sgemm_compact)
// Note: MKL compact format packs real and imaginary components separately
//       which makes it not directly compatible with our Vector types

#undef __IMPL_KK_MKL_DGEMM_COMPACT

template <typename ScalarType, int VecLen>
MKL_COMPACT_PACK mkl_compact_format() {
  Kokkos::abort("vector size not supported");
}
template <>
MKL_COMPACT_PACK mkl_compact_format<double, 2>() {
  return MKL_COMPACT_SSE;
}
template <>
MKL_COMPACT_PACK mkl_compact_format<float, 4>() {
  return MKL_COMPACT_SSE;
}
template <>
MKL_COMPACT_PACK mkl_compact_format<double, 4>() {
  return MKL_COMPACT_AVX;
}
template <>
MKL_COMPACT_PACK mkl_compact_format<float, 8>() {
  return MKL_COMPACT_AVX;
}
template <>
MKL_COMPACT_PACK mkl_compact_format<double, 8>() {
  return MKL_COMPACT_AVX512;
}
template <>
MKL_COMPACT_PACK mkl_compact_format<float, 16>() {
  return MKL_COMPACT_AVX512;
}

template <typename ScalarType, typename AViewType, typename xViewType,
          typename yViewType>
void kk_mkl_gemv(MKL_TRANSPOSE trans, const ScalarType alpha,
                 const AViewType &A, const xViewType &x, const ScalarType beta,
                 const yViewType &y) {
  typedef typename yViewType::value_type vector_type;

  static_assert(KokkosBatched::is_vector<vector_type>::value,
                "value type is not vector type");
  using value_type = typename vector_type::value_type;
  static_assert(std::is_same<typename AViewType::value_type::value_type,
                             value_type>::value &&
                    std::is_same<typename xViewType::value_type::value_type,
                                 value_type>::value,
                "scalar type mismatch");
  if (A.stride_0() != 1 && A.stride_1() != 1 && x.stride_0() != 1 &&
      y.stride_0() != 1) {
    Kokkos::abort("Strided inputs are not supported in MKL gemv/gemm");
  }

  // Note: not checking 0-sizes as MKL handles it fine
  const bool transposed = trans == MKL_TRANS || trans == MKL_CONJTRANS;
  const int m           = A.extent_int(transposed ? 1 : 0);
  const int n           = 1;
  const int k           = A.extent_int(transposed ? 0 : 1);

  const bool col_major    = A.stride_0() == 1;
  const MKL_LAYOUT layout = col_major ? MKL_COL_MAJOR : MKL_ROW_MAJOR;
  const MKL_INT A_ld = KOKKOSKERNELS_MACRO_MAX(1, A.extent(col_major ? 0 : 1));
  const MKL_COMPACT_PACK format =
      Impl::mkl_compact_format<value_type, vector_type::vector_length>();

  // cast away simd-vector pointers
  auto A_data = reinterpret_cast<const value_type *>(A.data());
  auto x_data = reinterpret_cast<const value_type *>(x.data());
  auto y_data = reinterpret_cast<value_type *>(y.data());

  Impl::kk_mkl_gemm_compact(layout, trans, MKL_NOTRANS, m, n, k,
                            (value_type)alpha, A_data, A_ld, x_data, 1,
                            (value_type)beta, y_data, 1, format,
                            (MKL_INT)vector_type::vector_length);
}

}  // namespace Impl

///
/// NT
///

template <>
template <typename ScalarType, typename AViewType, typename xViewType,
          typename yViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemv<Trans::NoTranspose, Algo::Gemv::CompactMKL>::invoke(
    const ScalarType alpha, const AViewType &A, const xViewType &x,
    const ScalarType beta, const yViewType &y) {
  Impl::kk_mkl_gemv(MKL_NOTRANS, alpha, A, x, beta, y);
  return 0;
}

///
/// T
///

template <>
template <typename ScalarType, typename AViewType, typename xViewType,
          typename yViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemv<Trans::Transpose, Algo::Gemv::CompactMKL>::invoke(
    const ScalarType alpha, const AViewType &A, const xViewType &x,
    const ScalarType beta, const yViewType &y) {
  Impl::kk_mkl_gemv(MKL_TRANS, alpha, A, x, beta, y);
  return 0;
}

///
/// CT
///

template <>
template <typename ScalarType, typename AViewType, typename xViewType,
          typename yViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemv<Trans::ConjTranspose, Algo::Gemv::CompactMKL>::invoke(
    const ScalarType alpha, const AViewType &A, const xViewType &x,
    const ScalarType beta, const yViewType &y) {
  Impl::kk_mkl_gemv(MKL_CONJTRANS, alpha, A, x, beta, y);
  return 0;
}

}  // namespace KokkosBlas

#endif  // __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__
#endif  // KOKKOSBLAS2_SERIAL_GEMV_TPL_SPEC_DECL_HPP_

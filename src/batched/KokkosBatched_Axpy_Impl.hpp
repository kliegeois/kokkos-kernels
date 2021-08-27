#ifndef __KOKKOSBATCHED_AXPY_IMPL_HPP__
#define __KOKKOSBATCHED_AXPY_IMPL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Axpy_Internal.hpp"

namespace KokkosBatched {

  ///
  /// Serial Impl
  /// ===========
  template<typename ScalarType,
           typename AViewType,
           typename alphaViewType>
  KOKKOS_INLINE_FUNCTION
  int
  SerialAxpy::
  invoke(const alphaViewType &alpha,
         const AViewType &X,
         const AViewType &Y) {
    return SerialAxpyInternal::
      invoke(X.extent(0), X.extent(1),
             alpha.data(), alpha.stride_0(),
             X.data(), X.stride_0(), X.stride_1(),
             Y.data(), Y.stride_0(), Y.stride_1());
  }

  ///
  /// Team Impl
  /// =========
    
  template<typename MemberType>
  template<typename ScalarType,
           typename AViewType,
           typename alphaViewType>
  KOKKOS_INLINE_FUNCTION
  int
  TeamAxpy<MemberType>::
  invoke(const MemberType &member, 
         const alphaViewType &alpha,
         const AViewType &X,
         const AViewType &Y) {
    return TeamAxpyInternal::
      invoke(member, 
             X.extent(0), X.extent(1),
             alpha.data(), alpha.stride_0(),
             X.data(), X.stride_0(), X.stride_1(),
             Y.data(), Y.stride_0(), Y.stride_1());
  }

  ///
  /// TeamVector Impl
  /// ===============
    
  template<typename MemberType>
  template<typename ScalarType,
           typename AViewType,
           typename alphaViewType>
  KOKKOS_INLINE_FUNCTION
  int
  TeamVectorAxpy<MemberType>::
  invoke(const MemberType &member, 
         const alphaViewType &alpha,
         const AViewType &X,
         const AViewType &Y) {
    return TeamVectorAxpyInternal::
      invoke<MemberType,
             ScalarType,
             typename AViewType::non_const_value_type,
             typename AViewType::array_layout>
             (member, 
             X.extent(0), X.extent(1),
             alpha.data(), alpha.stride_0(),
             X.data(), X.stride_0(), X.stride_1(),
             Y.data(), Y.stride_0(), Y.stride_1());
  }
  
}


#endif

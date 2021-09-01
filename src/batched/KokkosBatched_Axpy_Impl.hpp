#ifndef __KOKKOSBATCHED_AXPY_IMPL_HPP__
#define __KOKKOSBATCHED_AXPY_IMPL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Axpy_Internal.hpp"

namespace KokkosBatched {

  ///
  /// Serial Impl
  /// ===========
  template<typename ViewType,
           typename alphaViewType>
  KOKKOS_INLINE_FUNCTION
  int
  SerialAxpy::
  invoke(const alphaViewType &alpha,
         const ViewType &X,
         const ViewType &Y) {
    return SerialAxpyInternal::template
      invoke<typename alphaViewType::non_const_value_type,
             typename ViewType::non_const_value_type>
             (X.extent(0), X.extent(1),
              alpha.data(), alpha.stride_0(),
              X.data(), X.stride_0(), X.stride_1(),
              Y.data(), Y.stride_0(), Y.stride_1());
  }

  ///
  /// Team Impl
  /// =========
    
  template<typename MemberType>
  template<typename ViewType,
           typename alphaViewType>
  KOKKOS_INLINE_FUNCTION
  int
  TeamAxpy<MemberType>::
  invoke(const MemberType &member, 
         const alphaViewType &alpha,
         const ViewType &X,
         const ViewType &Y) {
    return TeamAxpyInternal::template
      invoke<MemberType,
             typename alphaViewType::non_const_value_type,
             typename ViewType::non_const_value_type>
             (member, 
              X.extent(0), X.extent(1),
              alpha.data(), alpha.stride_0(),
              X.data(), X.stride_0(), X.stride_1(),
              Y.data(), Y.stride_0(), Y.stride_1());
  }

  ///
  /// TeamVector Impl
  /// ===============
    
  template<typename MemberType>
  template<typename ViewType,
           typename alphaViewType>
  KOKKOS_INLINE_FUNCTION
  int
  TeamVectorAxpy<MemberType>::
  invoke(const MemberType &member, 
         const alphaViewType &alpha,
         const ViewType &X,
         const ViewType &Y) {
    return TeamVectorAxpyInternal::
      invoke<MemberType,
             typename alphaViewType::non_const_value_type,
             typename ViewType::non_const_value_type,
             typename ViewType::array_layout>
             (member, 
             X.extent(0), X.extent(1),
             alpha.data(), alpha.stride_0(),
             X.data(), X.stride_0(), X.stride_1(),
             Y.data(), Y.stride_0(), Y.stride_1());
  }
  
}


#endif

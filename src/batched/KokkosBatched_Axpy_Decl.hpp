#ifndef __KOKKOSBATCHED_AXPY_DECL_HPP__
#define __KOKKOSBATCHED_AXPY_DECL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

namespace KokkosBatched {

  ///
  /// Serial AXPY
  ///
  ///
  /// y <- alpha * x + y
  ///
  ///

  struct SerialAxpy {
    template<typename ScalarType,
             typename AViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const ScalarType *alpha,
           const AViewType &X,
           const AViewType &Y);
  };

  ///
  /// Team AXPY
  ///

  template<typename MemberType>
  struct TeamAxpy {
    template<typename ScalarType,
             typename AViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const ScalarType *alpha,
           const AViewType &X,
           const AViewType &Y);
  };

  ///
  /// TeamVector AXPY
  ///

  template<typename MemberType>
  struct TeamVectorAxpy {
    template<typename ScalarType,
             typename AViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const ScalarType *alpha,
           const AViewType &X,
           const AViewType &Y);
  };

}

#include "KokkosBatched_Axpy_Impl.hpp"

#endif

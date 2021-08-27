#ifndef __KOKKOSBATCHED_SPMV_DECL_HPP__
#define __KOKKOSBATCHED_SPMV_DECL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

namespace KokkosBatched {

  ///
  /// Serial SPMV
  ///
  ///
  /// y <- alpha * A * x + beta * y
  ///
  ///

  struct SerialSpmv {
    template<typename ScalarType,
             typename AViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const ScalarType alpha,
           const AViewType &X,
           const AViewType &Y);
  };

  ///
  /// Team SPMV
  ///

  template<typename MemberType>
  struct TeamSpmv {
    template<typename ScalarType,
             typename AViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const ScalarType alpha,
           const AViewType &X,
           const AViewType &Y);
  };

  ///
  /// TeamVector SPMV
  ///

  template<typename MemberType>
  struct TeamVectorSpmv {
    template<typename ScalarType,
             typename AViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const ScalarType alpha,
           const AViewType &X,
           const AViewType &Y);
  };

}

#include "KokkosBatched_Spmv_Impl.hpp"

#endif

#ifndef __KOKKOSBATCHED_CG_SERIAL_IMPL_HPP__
#define __KOKKOSBATCHED_CG_SERIAL_IMPL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_CG_Serial_Internal.hpp"

namespace KokkosBatched {

  template<typename MemberType, typename ArgAlgo>
  struct SerialCG<MemberType,Trans::NoTranspose,ArgAlgo> {
          
    template<typename ScalarType,
             typename DViewType,
             typename IntView,
             typename xViewType,
             typename yViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const DViewType &D,
           const IntView &r,
           const IntView &c,
           const xViewType &B,
           const yViewType &X) {
      return SerialCGInternal<ArgAlgo>::
        invoke(member, 
               X.extent(0), X.extent(1),
               D.data(), D.stride_0(), D.stride_1(),
               r.data(), r.stride_0(),
               c.data(), c.stride_0(),
               B.data(), B.stride_0(), B.stride_1(),
               X.data(), X.stride_0(), X.stride_1());
    }
  };

}


#endif

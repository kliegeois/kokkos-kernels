#ifndef __KOKKOSBATCHED_SPMV_SERIAL_IMPL_HPP__
#define __KOKKOSBATCHED_SPMV_SERIAL_IMPL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Spmv_Serial_Internal.hpp"

namespace KokkosBatched {

  template<typename ArgAlgo>
  struct SerialSpmv<Trans::NoTranspose,ArgAlgo> {
          
    template<typename DViewType,
             typename IntView,
             typename xViewType,
             typename yViewType,
             typename alphaViewType,
             typename betaViewType,
             int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const alphaViewType &alpha,
           const DViewType &D,
           const IntView &r,
           const IntView &c,
           const xViewType &X,
           const betaViewType &beta,
           const yViewType &Y) {
      return SerialSpmvInternal<ArgAlgo>::template
        invoke<typename alphaViewType::non_const_value_type, 
               typename DViewType::non_const_value_type, 
               typename IntView::non_const_value_type, 
               typename DViewType::array_layout, 
               dobeta>
               (X.extent(0), X.extent(1),
                alpha.data(), alpha.stride_0(),
                D.data(), D.stride_0(), D.stride_1(),
                r.data(), r.stride_0(),
                c.data(), c.stride_0(),
                X.data(), X.stride_0(), X.stride_1(),
                beta.data(), beta.stride_0(),
                Y.data(), Y.stride_0(), Y.stride_1());
    }
  };

}


#endif

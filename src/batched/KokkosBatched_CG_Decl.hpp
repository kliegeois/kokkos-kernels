#ifndef __KOKKOSBATCHED_CG_DECL_HPP__
#define __KOKKOSBATCHED_CG_DECL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

namespace KokkosBatched {

  ///
  /// Serial CG
  ///
  ///
  /// y <- alpha * A * x + beta * y
  ///
  ///

  template<typename ArgTrans,
           typename ArgAlgo>
  struct SerialCG {
    template<typename DViewType,
             typename IntView,
             typename xViewType,
             typename yViewType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const DViewType &D,
           const IntView &r,
           const IntView &c,
           const xViewType &B,
           const yViewType &X);
  };

  ///
  /// Team CG
  ///

  template<typename MemberType,
           typename ArgTrans,
           typename ArgAlgo>
  struct TeamCG {
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
           const yViewType &X);
  };

  ///
  /// TeamVector CG
  ///

  template<typename MemberType,
           typename ArgTrans,
           typename ArgAlgo>
  struct TeamVectorCG {
    template<typename DViewType,
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
           const yViewType &X);
  };

  ///
  /// Selective Interface
  ///
  template<typename MemberType,
           typename ArgTrans,
           typename ArgMode, typename ArgAlgo>
  struct CG {
    template<typename ScalarType,
             typename DViewType,
             typename IntView,
             typename xViewType,
             typename yViewType>
    KOKKOS_FORCEINLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const DViewType &D,
           const IntView &r,
           const IntView &c,
           const xViewType &B,
           const yViewType &X) {
      int r_val = 0;
      if (std::is_same<ArgMode,Mode::Serial>::value) {
        r_val = SerialCG<ArgTrans,ArgAlgo>::invoke(D, r, c, B, X);
      } else if (std::is_same<ArgMode,Mode::Team>::value) {
        r_val = TeamCG<MemberType,ArgTrans,ArgAlgo>::invoke(member, D, r, c, B, X);
      } else if (std::is_same<ArgMode,Mode::TeamVector>::value) {
        r_val = TeamVectorCG<MemberType,ArgTrans,ArgAlgo>::invoke(member, D, r, c, B, X);
      } 
      return r_val;
    }      
  };

}

#include "KokkosBatched_CG_Serial_Impl.hpp"
#include "KokkosBatched_CG_Team_Impl.hpp"
#include "KokkosBatched_CG_TeamVector_Impl.hpp"
#endif

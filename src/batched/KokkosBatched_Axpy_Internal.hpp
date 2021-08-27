#ifndef __KOKKOSBATCHED_AXPY_INTERNAL_HPP__
#define __KOKKOSBATCHED_AXPY_INTERNAL_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// Serial Internal Impl
  /// ==================== 
  struct SerialAxpyInternal {
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, 
           const ScalarType alpha, 
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (int i=0;i<m;++i)
        Y[i*ys0] += alpha*X[i*xs0];
        
      return 0;
    }

    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, 
           const ScalarType *__restrict__ alpha, 
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (int i=0;i<m;++i)
        Y[i*ys0] += alpha[i]*X[i*xs0];
        
      return 0;
    }
      
    template<typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, const int n, 
           const ScalarType *__restrict__ alpha, 
           /* */ ValueType *__restrict__ X, const int xs0, const int xs1,
           /* */ ValueType *__restrict__ Y, const int ys0, const int ys1) {

      if (xs0 > xs1)
        for (int i=0;i<m;++i)
          invoke(n, alpha[i], X+i*xs0, xs1, Y+i*ys0, ys1);
      else
        for (int j=0;j<n;++j)
          invoke(m, alpha, X+j*xs1, xs0, Y+j*ys1, ys0);
        
      return 0;
    }
  };

  ///
  /// Team Internal Impl
  /// ==================== 
  struct TeamAxpyInternal {
    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, 
           const ScalarType alpha, 
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

      Kokkos::parallel_for
        (Kokkos::TeamThreadRange(member,m),
         [&](const int &i) {
          Y[i*ys0] += alpha*X[i*xs0];
        });
      //member.team_barrier();
      return 0;
    }

    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, 
           const ScalarType *__restrict__ alpha, 
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

      Kokkos::parallel_for
        (Kokkos::TeamThreadRange(member,m),
         [&](const int &i) {
          Y[i*ys0] += alpha[i]*X[i*xs0];
        });
      //member.team_barrier();
      return 0;
    }
      
    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, const int n, 
           const ScalarType alpha, 
           /* */ ValueType *__restrict__ X, const int xs0, const int xs1,
           /* */ ValueType *__restrict__ Y, const int ys0, const int ys1) {
      if (m > n) {
        Kokkos::parallel_for
          (Kokkos::TeamThreadRange(member,m),
           [&](const int &i) {
            SerialAxpyInternal::invoke(n, alpha[i], X+i*xs0, xs1, Y+i*ys0, ys1);
          });
      } else {
        Kokkos::parallel_for
          (Kokkos::TeamThreadRange(member,n),
           [&](const int &j) {
            SerialAxpyInternal::invoke(m, alpha, X+j*xs1, xs0, Y+j*ys1, ys0);
          });
      }
      //member.team_barrier();
      return 0;
    }
  };

  ///
  /// TeamVector Internal Impl
  /// ======================== 
  struct TeamVectorAxpyInternal {
    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, 
           const ScalarType alpha, 
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

      Kokkos::parallel_for
        (Kokkos::TeamVectorRange(member,m),
         [&](const int &i) {
          Y[i*ys0] += alpha*X[i*xs0];
        });
      //member.team_barrier();
      return 0;
    }

    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, 
           const ScalarType *__restrict__ alpha, 
           /* */ ValueType *__restrict__ X, const int xs0,
           /* */ ValueType *__restrict__ Y, const int ys0) {

      Kokkos::parallel_for
        (Kokkos::TeamVectorRange(member,m),
         [&](const int &i) {
          Y[i*ys0] += alpha[i]*X[i*xs0];
        });
      //member.team_barrier();
      return 0;
    }
      
    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, const int n, 
           const ScalarType alpha, 
           /* */ ValueType *__restrict__ X, const int xs0, const int xs1,
           /* */ ValueType *__restrict__ Y, const int ys0, const int ys1) {
      if (xs0 > xs1) {
        Kokkos::parallel_for
          (Kokkos::TeamThreadRange(member,m),
           [&](const int &i) {
            Kokkos::parallel_for
              (Kokkos::ThreadVectorRange(member,n),
               [&](const int &j) {
                Y[i*ys0+j*ys1] += alpha[i] * X[i*xs0+j*xs1];
              });
          });
      } else {
        Kokkos::parallel_for
          (Kokkos::ThreadVectorRange(member,m),
           [&](const int &i) {
            Kokkos::parallel_for
              (Kokkos::TeamThreadRange(member,n),
               [&](const int &j) {
                Y[i*ys0+j*ys1] += alpha[i] * X[i*xs0+j*xs1];
              });
          });
      }
      //member.team_barrier();
      return 0;
    }
  };

}


#endif

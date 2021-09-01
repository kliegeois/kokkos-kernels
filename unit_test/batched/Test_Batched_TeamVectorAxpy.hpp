/// \author Kim Liegeois (knliege@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

//#include "KokkosBatched_Vector.hpp"

#include "KokkosBatched_Axpy_Decl.hpp"
#include "KokkosBatched_Axpy_Impl.hpp"

#include "KokkosKernels_TestUtils.hpp"

using namespace KokkosBatched;

namespace Test {
namespace TeamVectorAxpy {
 
  template<typename DeviceType,
           typename ViewType,
           typename alphaViewType>
  struct Functor_TestBatchedTeamVectorAxpy {
    const alphaViewType _alpha;
    const ViewType _X;
    const ViewType _Y;
    const int _N_team;
    
    KOKKOS_INLINE_FUNCTION
    Functor_TestBatchedTeamVectorAxpy(const alphaViewType &alpha,
      const ViewType &X,
      const ViewType &Y,
      const int N_team)
    : _alpha(alpha), _X(X), _Y(Y), _N_team(N_team) {}
    
    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &member) const {
      const int first_matrix =
          static_cast<int>(member.league_rank()) * _N_team;
      const int N = _X.extent(0);
      const int last_matrix = (static_cast<int>(member.league_rank() + 1) * _N_team < N ? static_cast<int>(member.league_rank() + 1) * _N_team : N );

      auto alpha = Kokkos::subview(_alpha,Kokkos::make_pair(first_matrix,last_matrix));
      auto x = Kokkos::subview(_X,Kokkos::make_pair(first_matrix,last_matrix),Kokkos::ALL);
      auto y = Kokkos::subview(_Y,Kokkos::make_pair(first_matrix,last_matrix),Kokkos::ALL);
      
      KokkosBatched::TeamVectorAxpy<MemberType>::template invoke<ViewType, alphaViewType>
          (member, alpha, x, y);
    }
    
    inline
    void run() {
      typedef typename ViewType::value_type value_type;
      std::string name_region("KokkosBatched::Test::TeamVectorAxpy");
      std::string name_value_type = ( std::is_same<value_type,float>::value ? "::Float" : 
                                      std::is_same<value_type,double>::value ? "::Double" :
                                      std::is_same<value_type,Kokkos::complex<float> >::value ? "::ComplexFloat" :
                                      std::is_same<value_type,Kokkos::complex<double> >::value ? "::ComplexDouble" : "::UnknownValueType" );                               
      std::string name = name_region + name_value_type;
      Kokkos::Profiling::pushRegion(name.c_str() );
      Kokkos::TeamPolicy<DeviceType> policy(_X.extent(0)/_N_team, Kokkos::AUTO(), Kokkos::AUTO());
      Kokkos::parallel_for(name.c_str(), policy, *this);
      Kokkos::Profiling::popRegion();
    }
  };
    
  template<typename DeviceType,
           typename ViewType,
           typename alphaViewType>
  void impl_test_batched_axpy(const int N, const int BlkSize, const int N_team) {
    typedef typename ViewType::value_type value_type;
    typedef Kokkos::Details::ArithTraits<value_type> ats;

    ViewType  X0("x0", N, BlkSize), X1("x1", N, BlkSize), Y0("y0", N, BlkSize), Y1("y1", N, BlkSize);

    alphaViewType  alpha("alpha", N);

    Kokkos::Random_XorShift64_Pool<typename DeviceType::execution_space> random(13718);
    Kokkos::fill_random(X0, random, value_type(1.0));
    Kokkos::fill_random(Y0, random, value_type(1.0));
    Kokkos::fill_random(alpha, random, value_type(1.0));

    Kokkos::fence();

    Kokkos::deep_copy(X1, X0);
    Kokkos::deep_copy(Y1, Y0);

    /// test body
    auto alpha_host = Kokkos::create_mirror_view(alpha);
    auto X0_host = Kokkos::create_mirror_view(X0);
    auto Y0_host = Kokkos::create_mirror_view(Y0);

    Kokkos::deep_copy(alpha_host, alpha);
    Kokkos::deep_copy(X0_host, X0);
    Kokkos::deep_copy(Y0_host, Y0);

    for (int l=0;l<N;++l) 
      for (int i=0;i<BlkSize;++i)
        Y0_host(l,i) += alpha_host(l)*X0_host(l,i);

    Functor_TestBatchedTeamVectorAxpy<DeviceType,ViewType,alphaViewType>
    (alpha, X1, Y1, N_team).run();

    Kokkos::fence();

    /// for comparison send it to host
    auto Y1_host = Kokkos::create_mirror_view(Y1);

    Kokkos::deep_copy(Y1_host, Y1);

    /// check c0 = c1 ; this eps is about 10^-14
    typedef typename ats::mag_type mag_type;
    mag_type sum(1), diff(0);
    const mag_type eps = 1.0e3 * ats::epsilon();

    for (int l=0;l<N;++l) 
      for (int i=0;i<BlkSize;++i) {
        sum  += ats::abs(Y0_host(l,i));
        diff += ats::abs(Y0_host(l,i)-Y1_host(l,i));
      }
    EXPECT_NEAR_KK( diff/sum, 0, eps);
  }
}
}

template<typename DeviceType, 
         typename ValueType, 
         typename ScalarType>
int test_batched_teamvector_axpy() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) 
  {
    typedef Kokkos::View<ValueType**,Kokkos::LayoutLeft,DeviceType> ViewType;
    typedef Kokkos::View<ValueType*,Kokkos::LayoutLeft,DeviceType> alphaViewType;
    
    Test::TeamVectorAxpy::impl_test_batched_axpy<DeviceType,ViewType,alphaViewType>( 0, 10, 2);
    for (int i=3;i<10;++i) {                                                                                        
      Test::TeamVectorAxpy::impl_test_batched_axpy<DeviceType,ViewType,alphaViewType>(1024,  i, 2);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) 
  {
    typedef Kokkos::View<ValueType**,Kokkos::LayoutRight,DeviceType> ViewType;
    typedef Kokkos::View<ValueType*,Kokkos::LayoutRight,DeviceType> alphaViewType;

    Test::TeamVectorAxpy::impl_test_batched_axpy<DeviceType,ViewType,alphaViewType>( 0, 10, 2);
    for (int i=3;i<10;++i) {                                                                                        
      Test::TeamVectorAxpy::impl_test_batched_axpy<DeviceType,ViewType,alphaViewType>(1024,  i, 2);
    }
  }
#endif
  
  return 0;
}


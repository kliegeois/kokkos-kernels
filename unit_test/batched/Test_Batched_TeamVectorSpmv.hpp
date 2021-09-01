/// \author Kim Liegeois (knliege@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

//#include "KokkosBatched_Vector.hpp"

#include "KokkosBatched_Spmv_Decl.hpp"
#include "KokkosBatched_Spmv_TeamVector_Impl.hpp"

#include "KokkosKernels_TestUtils.hpp"

using namespace KokkosBatched;

namespace Test {
namespace TeamVectorSpmv {

  template<typename T>
  struct ParamTag { 
    typedef T trans;
  };
 
  template<typename DeviceType,
           typename ParamTagType, 
           typename AlgoTagType,
           typename DViewType,
           typename IntView,
           typename xViewType,
           typename yViewType,
           typename alphaViewType,
           typename betaViewType,
           int dobeta>
  struct Functor_TestBatchedTeamVectorSpmv {
    const alphaViewType _alpha;
    const DViewType _D;
    const IntView _r;
    const IntView _c;
    const xViewType _X;
    const betaViewType _beta;
    const yViewType _Y;
    const int _N_team;
    
    KOKKOS_INLINE_FUNCTION
    Functor_TestBatchedTeamVectorSpmv(const alphaViewType &alpha,
      const DViewType &D,
      const IntView &r,
      const IntView &c,
      const xViewType &X,
      const betaViewType &beta,
      const yViewType &Y,
      const int N_team)
    : _alpha(alpha), _D(D), _r(r), _c(c), _X(X), _beta(beta), _Y(Y), _N_team(N_team) {}
    
    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const ParamTagType &, const MemberType &member) const {
      const int first_matrix =
          static_cast<int>(member.league_rank()) * _N_team;
      const int N = _D.extent(0);
      const int last_matrix = (static_cast<int>(member.league_rank() + 1) * _N_team < N ? static_cast<int>(member.league_rank() + 1) * _N_team : N );
      
      auto alpha = Kokkos::subview(_alpha,Kokkos::make_pair(first_matrix,last_matrix));
      auto d = Kokkos::subview(_D,Kokkos::make_pair(first_matrix,last_matrix),Kokkos::ALL);
      auto x = Kokkos::subview(_X,Kokkos::make_pair(first_matrix,last_matrix),Kokkos::ALL);
      auto beta = Kokkos::subview(_beta,Kokkos::make_pair(first_matrix,last_matrix));
      auto y = Kokkos::subview(_Y,Kokkos::make_pair(first_matrix,last_matrix),Kokkos::ALL);
      
      KokkosBatched::TeamVectorSpmv<MemberType, typename ParamTagType::trans, AlgoTagType>::template invoke<DViewType, IntView, xViewType, yViewType, alphaViewType, betaViewType, dobeta> (member, alpha, d, _r, _c, x, beta, y);
    }
    
    inline
    void run() {
      typedef typename DViewType::value_type value_type;
      std::string name_region("KokkosBatched::Test::TeamVectorSpmv");
      std::string name_value_type = ( std::is_same<value_type,float>::value ? "::Float" : 
                                      std::is_same<value_type,double>::value ? "::Double" :
                                      std::is_same<value_type,Kokkos::complex<float> >::value ? "::ComplexFloat" :
                                      std::is_same<value_type,Kokkos::complex<double> >::value ? "::ComplexDouble" : "::UnknownValueType" );                               
      std::string name = name_region + name_value_type;
      Kokkos::Profiling::pushRegion(name.c_str() );
      Kokkos::TeamPolicy<DeviceType,ParamTagType> policy(_D.extent(0)/_N_team, Kokkos::AUTO(), Kokkos::AUTO());
      Kokkos::parallel_for(name.c_str(), policy, *this);
      Kokkos::Profiling::popRegion();
    }
  };
    
  template<typename DeviceType,
           typename ParamTagType, 
           typename AlgoTagType,
           typename DViewType,
           typename IntView,
           typename xViewType,
           typename yViewType,
           typename alphaViewType,
           typename betaViewType,
           int dobeta>
  void impl_test_batched_spmv(const int N, const int BlkSize, const int N_team) {
    typedef typename DViewType::value_type value_type;
    typedef Kokkos::Details::ArithTraits<value_type> ats;

    const int nnz = (BlkSize-2) * 3 + 2 * 2;

    xViewType  X0("x0", N, BlkSize), X1("x1", N, BlkSize);
    yViewType  Y0("y0", N, BlkSize), Y1("y1", N, BlkSize);
    DViewType  D("D", N, nnz);
    IntView    r("r", BlkSize+1);
    IntView    c("c", nnz);

    alphaViewType  alpha("alpha", N);
    betaViewType   beta("beta", N);

    Kokkos::deep_copy(alpha, value_type(1.0));
    Kokkos::deep_copy(beta, value_type(1.0));

    Kokkos::Random_XorShift64_Pool<typename DeviceType::execution_space> random(13718);
    Kokkos::fill_random(X0, random, value_type(1.0));
    Kokkos::fill_random(Y0, random, value_type(1.0));

    auto D_host = Kokkos::create_mirror_view(D);
    auto r_host = Kokkos::create_mirror_view(r);
    auto c_host = Kokkos::create_mirror_view(c);

    r_host(0) = 0;

    int current_col = 0;

    for (int i=0;i<BlkSize;++i) {
      r_host(i+1) = r_host(i) + (i==0 || i==(BlkSize-1) ? 2 : 3);
    }
    for (int i=0;i<nnz;++i) {
      if (i%3 == 0) {
        for (int l=0;l<N;++l) {
          D_host(l,i) = value_type(1.0);
        }
        c_host(i) = current_col;
        ++current_col;
      }
      else {
        for (int l=0;l<N;++l) {
          D_host(l,i) = value_type(0.5);
        }
        c_host(i) = current_col;
        if (i%3 == 1)
          --current_col;
        else
          ++current_col;
      }
    }

    Kokkos::fence();

    Kokkos::deep_copy(D, D_host);
    Kokkos::deep_copy(r, r_host);
    Kokkos::deep_copy(c, c_host);

    Kokkos::deep_copy(X1, X0);
    Kokkos::deep_copy(Y1, Y0);

    /// test body
    auto alpha_host = Kokkos::create_mirror_view(alpha);
    auto beta_host = Kokkos::create_mirror_view(beta);
    auto X0_host = Kokkos::create_mirror_view(X0);
    auto Y0_host = Kokkos::create_mirror_view(Y0);

    Kokkos::deep_copy(alpha_host, alpha);
    Kokkos::deep_copy(beta_host, beta);
    Kokkos::deep_copy(X0_host, X0);
    Kokkos::deep_copy(Y0_host, Y0);

    for (int l=0;l<N;++l) 
      for (int i=0;i<BlkSize;++i) {
        if (dobeta == 0)
          Y0_host(l,i) = value_type(0.0);
        else
          Y0_host(l,i) *= beta_host(l);
        if (i != 0 && i != (BlkSize-1))
          Y0_host(l,i) += alpha_host(l)*(X0_host(l,i) + 0.5*X0_host(l,i-1) + 0.5*X0_host(l,i+1));
        else if (i == 0)
          Y0_host(l,i) += alpha_host(l)*(X0_host(l,i) + 0.5*X0_host(l,i+1));
        else
          Y0_host(l,i) += alpha_host(l)*(X0_host(l,i) + 0.5*X0_host(l,i-1));
      }

    Functor_TestBatchedTeamVectorSpmv<DeviceType,ParamTagType,AlgoTagType,DViewType,IntView,xViewType,yViewType,alphaViewType,betaViewType,dobeta>
    (alpha, D, r, c, X1, beta, Y1, N_team).run();

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
         typename ScalarType,
         typename ParamTagType,
         typename AlgoTagType>
int test_batched_teamvector_spmv() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) 
  {
    typedef Kokkos::View<ValueType**,Kokkos::LayoutLeft,DeviceType> ViewType;
    typedef Kokkos::View<int*,Kokkos::LayoutLeft,DeviceType> IntView;
    typedef Kokkos::View<ValueType*,Kokkos::LayoutLeft,DeviceType> alphaViewType;
    
    Test::TeamVectorSpmv::impl_test_batched_spmv<DeviceType,ParamTagType,AlgoTagType,ViewType,IntView,ViewType,ViewType,alphaViewType,alphaViewType,0>( 0, 10, 1);
    for (int i=3;i<10;++i) {                                                                                        
      Test::TeamVectorSpmv::impl_test_batched_spmv<DeviceType,ParamTagType,AlgoTagType,ViewType,IntView,ViewType,ViewType,alphaViewType,alphaViewType,0>(1024,  i, 2);
    }
    Test::TeamVectorSpmv::impl_test_batched_spmv<DeviceType,ParamTagType,AlgoTagType,ViewType,IntView,ViewType,ViewType,alphaViewType,alphaViewType,1>( 0, 10, 1);
    for (int i=3;i<10;++i) {                                                                                        
      Test::TeamVectorSpmv::impl_test_batched_spmv<DeviceType,ParamTagType,AlgoTagType,ViewType,IntView,ViewType,ViewType,alphaViewType,alphaViewType,1>(1024,  i, 2);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) 
  {
    typedef Kokkos::View<ValueType**,Kokkos::LayoutRight,DeviceType> ViewType;
    typedef Kokkos::View<int*,Kokkos::LayoutRight,DeviceType> IntView;
    typedef Kokkos::View<ValueType*,Kokkos::LayoutRight,DeviceType> alphaViewType;

    Test::TeamVectorSpmv::impl_test_batched_spmv<DeviceType,ParamTagType,AlgoTagType,ViewType,IntView,ViewType,ViewType,alphaViewType,alphaViewType,0>( 0, 10, 1);
    for (int i=3;i<10;++i) {                                                                                        
      Test::TeamVectorSpmv::impl_test_batched_spmv<DeviceType,ParamTagType,AlgoTagType,ViewType,IntView,ViewType,ViewType,alphaViewType,alphaViewType,0>(1024,  i, 2);
    }

    Test::TeamVectorSpmv::impl_test_batched_spmv<DeviceType,ParamTagType,AlgoTagType,ViewType,IntView,ViewType,ViewType,alphaViewType,alphaViewType,1>( 0, 10, 1);
    for (int i=3;i<10;++i) {                                                                                         
      Test::TeamVectorSpmv::impl_test_batched_spmv<DeviceType,ParamTagType,AlgoTagType,ViewType,IntView,ViewType,ViewType,alphaViewType,alphaViewType,1>(1024,  i, 2);
    }
  }
#endif
  
  return 0;
}


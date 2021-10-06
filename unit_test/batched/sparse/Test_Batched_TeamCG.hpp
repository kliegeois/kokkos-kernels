/// \author Kim Liegeois (knliege@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

//#include "KokkosBatched_Vector.hpp"

#include "KokkosBatched_CG.hpp"

#include "KokkosKernels_TestUtils.hpp"

#include "KokkosBatched_CG.hpp"

using namespace KokkosBatched;

namespace Test {
namespace CG {
 
  template<typename DeviceType,
           typename ValuesViewType,
           typename IntView,
           typename VectorViewType>
  struct Functor_TestBatchedTeamCG {
    const ValuesViewType _D;
    const IntView _r;
    const IntView _c;
    const VectorViewType _X;
    const VectorViewType _B;
    const int _N_team;
    
    KOKKOS_INLINE_FUNCTION
    Functor_TestBatchedTeamCG(const ValuesViewType &D,
      const IntView &r,
      const IntView &c,
      const VectorViewType &X,
      const VectorViewType &B,
      const int N_team)
    : _D(D), _r(r), _c(c), _X(X), _B(B), _N_team(N_team) {}
    
    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &member) const {
      const int first_matrix =
          static_cast<int>(member.league_rank()) * _N_team;
      const int N = _D.extent(0);
      const int last_matrix = (static_cast<int>(member.league_rank() + 1) * _N_team < N ? static_cast<int>(member.league_rank() + 1) * _N_team : N );
      
      auto d = Kokkos::subview(_D,Kokkos::make_pair(first_matrix,last_matrix),Kokkos::ALL);
      auto x = Kokkos::subview(_X,Kokkos::make_pair(first_matrix,last_matrix),Kokkos::ALL);
      auto b = Kokkos::subview(_B,Kokkos::make_pair(first_matrix,last_matrix),Kokkos::ALL);
      
      KokkosBatched::TeamCG<MemberType>::template invoke<ValuesViewType, IntView, VectorViewType>
          (member, d, _r, _c, x, b);
    }
    
    inline
    void run() {
      typedef typename ValuesViewType::value_type value_type;
      std::string name_region("KokkosBatched::Test::TeamCG");
      std::string name_value_type = ( std::is_same<value_type,float>::value ? "::Float" : 
                                      std::is_same<value_type,double>::value ? "::Double" :
                                      std::is_same<value_type,Kokkos::complex<float> >::value ? "::ComplexFloat" :
                                      std::is_same<value_type,Kokkos::complex<double> >::value ? "::ComplexDouble" : "::UnknownValueType" );                               
      std::string name = name_region + name_value_type;
      Kokkos::Profiling::pushRegion(name.c_str() );
      Kokkos::TeamPolicy<DeviceType> policy(_D.extent(0)/_N_team, Kokkos::AUTO(), Kokkos::AUTO());

      size_t bytes_0 = ValuesViewType::shmem_size(_N_team, _D.extent(1));
      size_t bytes_1 = ValuesViewType::shmem_size(_N_team, 1);
      policy.set_scratch_size(0, Kokkos::PerTeam(5 * bytes_0 + 7 * bytes_1));

      Kokkos::parallel_for(name.c_str(), policy, *this);
      Kokkos::Profiling::popRegion();
    }
  };
    
  template<typename DeviceType,
           typename ValuesViewType,
           typename IntView,
           typename VectorViewType>
  void impl_test_batched_CG(const int N, const int BlkSize, const int N_team) {
    typedef typename ValuesViewType::value_type value_type;
    typedef Kokkos::Details::ArithTraits<value_type> ats;

    const int nnz = (BlkSize-2) * 3 + 2 * 2;

    VectorViewType  X("x0", N, BlkSize);
    VectorViewType  B("b", N, BlkSize);
    ValuesViewType  D("D", N, nnz);
    IntView    r("r", BlkSize+1);
    IntView    c("c", nnz);

    Kokkos::Random_XorShift64_Pool<typename DeviceType::execution_space> random(13718);
    Kokkos::fill_random(X, random, value_type(1.0));
    Kokkos::fill_random(B, random, value_type(1.0));

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

    write1DArrayTofile(r, "r.txt");
    write1DArrayTofile(c, "c.txt");

    write2DArrayTofile(D, "D.txt");
    write2DArrayTofile(X, "X.txt");
    write2DArrayTofile(B, "B.txt");
    
    Kokkos::fence();

    Functor_TestBatchedTeamCG<DeviceType,ValuesViewType,IntView,VectorViewType> (D, r, c, X, B, N_team).run();

    write2DArrayTofile(X, "R_0.txt");

    Kokkos::fence();
  }
}
}

template<typename DeviceType, 
         typename ValueType>
int test_batched_team_CG() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) 
  {
    typedef Kokkos::View<ValueType**,Kokkos::LayoutLeft,DeviceType> ViewType;
    typedef Kokkos::View<int*,Kokkos::LayoutLeft,DeviceType> IntView;
    typedef Kokkos::View<ValueType**,Kokkos::LayoutLeft,DeviceType> VectorViewType;
    
    for (int i=3;i<4;++i) {                                                                                        
      Test::CG::impl_test_batched_CG<DeviceType,ViewType,IntView,VectorViewType>(2, i, 2);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) 
  {
    typedef Kokkos::View<ValueType**,Kokkos::LayoutRight,DeviceType> ViewType;
    typedef Kokkos::View<int*,Kokkos::LayoutRight,DeviceType> IntView;
    typedef Kokkos::View<ValueType**,Kokkos::LayoutRight,DeviceType> VectorViewType;

    for (int i=3;i<4;++i) {                                                                                        
      Test::CG::impl_test_batched_CG<DeviceType,ViewType,IntView,VectorViewType>(2, i, 2);
    }
  }
#endif
  
  return 0;
}


/// \author Kim Liegeois (knliege@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

//#include "KokkosBatched_Vector.hpp"

#include "KokkosBatched_CG.hpp"
#include "KokkosBatched_CG_Serial_Impl.hpp"

#include "KokkosKernels_TestUtils.hpp"

#include "KokkosBatched_CG.hpp"

using namespace KokkosBatched;

namespace Test {
namespace CG {
 
  template<typename DeviceType,
           typename ValuesViewType,
           typename IntView,
           typename VectorViewType>
  struct Functor_TestBatchedSerialCG {
    const ValuesViewType _D;
    const IntView _r;
    const IntView _c;
    const VectorViewType _X;
    const VectorViewType _B;
    
    KOKKOS_INLINE_FUNCTION
    Functor_TestBatchedSerialCG(const ValuesViewType &D,
      const IntView &r,
      const IntView &c,
      const VectorViewType &X,
      const VectorViewType &B)
    : _D(D), _r(r), _c(c), _X(X), _B(B) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int k) const {
      auto d = Kokkos::subview(_D,Kokkos::make_pair(k,k+1),Kokkos::ALL);
      auto x = Kokkos::subview(_X,Kokkos::make_pair(k,k+1),Kokkos::ALL);
      auto b = Kokkos::subview(_B,Kokkos::make_pair(k,k+1),Kokkos::ALL);
      
      KokkosBatched::SerialCG::template invoke<ValuesViewType, IntView, VectorViewType>
          (d, _r, _c, x, b);
    }
    
    inline
    void run() {
      typedef typename ValuesViewType::value_type value_type;
      std::string name_region("KokkosBatched::Test::SerialCG");
      std::string name_value_type = ( std::is_same<value_type,float>::value ? "::Float" : 
                                      std::is_same<value_type,double>::value ? "::Double" :
                                      std::is_same<value_type,Kokkos::complex<float> >::value ? "::ComplexFloat" :
                                      std::is_same<value_type,Kokkos::complex<double> >::value ? "::ComplexDouble" : "::UnknownValueType" );                               
      std::string name = name_region + name_value_type;
      Kokkos::Profiling::pushRegion(name.c_str() );
      Kokkos::RangePolicy<DeviceType> policy(0, _D.extent(0));
      Kokkos::parallel_for(name.c_str(), policy, *this);
      Kokkos::Profiling::popRegion();
    }
  };
    
  template<typename DeviceType,
           typename ValuesViewType,
           typename IntView,
           typename VectorViewType>
  void impl_test_batched_CG(const int N, const int BlkSize) {
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

    Functor_TestBatchedSerialCG<DeviceType,ValuesViewType,IntView,VectorViewType> (D, r, c, X, B).run();

    Kokkos::fence();
  }
}
}

template<typename DeviceType, 
         typename ValueType>
int test_batched_CG() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) 
  {
    typedef Kokkos::View<ValueType**,Kokkos::LayoutLeft,DeviceType> ViewType;
    typedef Kokkos::View<int*,Kokkos::LayoutLeft,DeviceType> IntView;
    typedef Kokkos::View<ValueType**,Kokkos::LayoutLeft,DeviceType> VectorViewType;
    
    for (int i=3;i<10;++i) {                                                                                        
      Test::CG::impl_test_batched_CG<DeviceType,ViewType,IntView,VectorViewType>(1024,  i);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) 
  {
    typedef Kokkos::View<ValueType**,Kokkos::LayoutRight,DeviceType> ViewType;
    typedef Kokkos::View<int*,Kokkos::LayoutRight,DeviceType> IntView;
    typedef Kokkos::View<ValueType**,Kokkos::LayoutRight,DeviceType> VectorViewType;

    for (int i=3;i<10;++i) {                                                                                        
      Test::CG::impl_test_batched_CG<DeviceType,ViewType,IntView,VectorViewType>(1024,  i);
    }
  }
#endif
  
  return 0;
}


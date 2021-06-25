/// Kokkos headers
#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"
#include "Kokkos_Random.hpp"

#if  defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
#if !defined(KOKKOS_ENABLE_CUDA) || (8000 <= CUDA_VERSION)
#if  defined(KOKKOS_ENABLE_CUDA_LAMBDA)
#define KOKKOSBATCHED_TEST_BLOCKCG 
#endif 
#endif
#endif


#if defined(KOKKOSBATCHED_TEST_BLOCKCG)

/// KokkosKernels headers
#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

#include <Kokkos_ArithTraits.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Vector.hpp>
#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>
#include <KokkosBatched_AddRadial_Decl.hpp>
#include <KokkosBatched_AddRadial_Impl.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>
#include <KokkosBatched_Gemm_Team_Impl.hpp>
#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Gemv_Serial_Impl.hpp>
#include <KokkosBatched_Gemv_Team_Impl.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <KokkosBatched_Trsm_Serial_Impl.hpp>
#include <KokkosBatched_Trsm_Team_Impl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#include <KokkosBatched_Trsv_Serial_Impl.hpp>
#include <KokkosBatched_Trsv_Team_Impl.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>
#include <KokkosBatched_LU_Team_Impl.hpp>

#define KOKKOSBATCHED_PROFILE 1
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
#include "cuda_profiler_api.h"
#endif

#define KOKKOSBATCHED_USE_128BIT_MEMORY_INST

typedef Kokkos::DefaultExecutionSpace exec_space;
typedef typename exec_space::memory_space memory_space;
typedef Kokkos::DefaultHostExecutionSpace host_space;

typedef double value_type;

/// 128*128*128/16*5 * (2*8) / 16
///
/// simd typedefs
///
using namespace KokkosBatched;

static constexpr int vector_length = DefaultVectorLength<value_type,memory_space>::value;
#if defined(KOKKOSBATCHED_USE_128BIT_MEMORY_INST)
static constexpr int internal_vector_length = DefaultInternalVectorLength<value_type,memory_space>::value;
#else
static constexpr int internal_vector_length = 1;
#endif

typedef Vector<SIMD<value_type>,vector_length> vector_type;
#if defined(KOKKOSBATCHED_USE_128BIT_MEMORY_INST)
typedef Vector<SIMD<value_type>,internal_vector_length> internal_vector_type;
#else
typedef value_type internal_vector_type;
#endif

template<typename ActiveMemorySpace>
struct FactorizeModeAndAlgo;

template<>
struct FactorizeModeAndAlgo<Kokkos::HostSpace> {
  typedef Mode::Serial mode_type;
  typedef Algo::Level3::Blocked algo_type;   
};

#if defined(KOKKOS_ENABLE_CUDA)
template<>
struct FactorizeModeAndAlgo<Kokkos::CudaSpace> {
  typedef Mode::Team mode_type;
  typedef Algo::Level3::Unblocked algo_type;   
};
#endif

template<typename ActiveMemorySpace>
struct SolveModeAndAlgo;

template<>
struct SolveModeAndAlgo<Kokkos::HostSpace> {
  typedef Mode::Serial mode_type;
  typedef Algo::Level2::Blocked algo_type;   
};

#if defined(KOKKOS_ENABLE_CUDA)
template<>
struct SolveModeAndAlgo<Kokkos::CudaSpace> {
  typedef Mode::Team mode_type;
  typedef Algo::Level2::Unblocked algo_type;   
};
#endif

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
    cudaProfilerStop();
#endif
    Kokkos::print_configuration(std::cout);

    //typedef Kokkos::Details::ArithTraits<value_type> ats;
    Kokkos::Impl::Timer timer;

    ///
    /// input arguments parsing
    ///
    int N = 128*128; /// # of problems (batch size)
    int L = 128;     /// length of block tridiags
    int Blk = 5;     /// block dimension
    int Nvec = 1;
    int S = 0; /// scratch size
    int niter = 1;
    for (int i=1;i<argc;++i) {
      const std::string& token = argv[i];
      if (token == std::string("-N")) N = std::atoi(argv[++i]);
      if (token == std::string("-L")) L = std::atoi(argv[++i]);
      if (token == std::string("-B")) Blk = std::atoi(argv[++i]);
      if (token == std::string("-Nvec")) Nvec = std::atoi(argv[++i]);
      if (token == std::string("-S")) S = std::atoi(argv[++i]);
      if (token == std::string("-Niter")) niter = std::atoi(argv[++i]);
    }

    printf(" :::: Testing (N = %d, L = %d, Blk = %d, vl = %d, vi = %d, niter = %d)\n", 
           N, L, Blk, vector_length, internal_vector_length, niter);
  }
  Kokkos::finalize();

  return 0;
}

#else
int main() {
  return 0;
}
#endif


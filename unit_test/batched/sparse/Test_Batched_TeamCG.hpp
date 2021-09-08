/// \author Kim Liegeois (knliege@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

//#include "KokkosBatched_Vector.hpp"

#include "KokkosBatched_CG.hpp"

#include "KokkosKernels_TestUtils.hpp"

#include "KokkosBatched_CG.hpp"

#include "KokkosBatched_CrsMatrix.hpp"

using namespace KokkosBatched;

namespace Test {
namespace TeamCG {

template <typename DeviceType, typename ValuesViewType, typename IntView,
          typename VectorViewType>
struct Functor_TestBatchedTeamCG {
  const ValuesViewType _D;
  const IntView _r;
  const IntView _c;
  const VectorViewType _X;
  const VectorViewType _B;
  const int _N_team;
  KrylovHandle<typename ValuesViewType::value_type> *handle;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamCG(const ValuesViewType &D, const IntView &r,
                            const IntView &c, const VectorViewType &X,
                            const VectorViewType &B, const int N_team)
      : _D(D), _r(r), _c(c), _X(X), _B(B), _N_team(N_team) {
    handle = new KrylovHandle<typename ValuesViewType::value_type>;
  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType &member) const {
    const int first_matrix = static_cast<int>(member.league_rank()) * _N_team;
    const int N            = _D.extent(0);
    const int last_matrix =
        (static_cast<int>(member.league_rank() + 1) * _N_team < N
             ? static_cast<int>(member.league_rank() + 1) * _N_team
             : N);

    auto d = Kokkos::subview(_D, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);
    auto x = Kokkos::subview(_X, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);
    auto b = Kokkos::subview(_B, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);

    using Operator = KokkosBatched::CrsMatrix<ValuesViewType, IntView>;

    Operator A(d, _r, _c);

    KokkosBatched::TeamCG<MemberType>::template invoke<Operator,
                                                       VectorViewType>(
        member, A, b, x, handle);
  }

  inline void run() {
    typedef typename ValuesViewType::value_type value_type;
    std::string name_region("KokkosBatched::Test::TeamCG");
    std::string name_value_type =
        (std::is_same<value_type, float>::value
             ? "::Float"
             : std::is_same<value_type, double>::value
                   ? "::Double"
                   : std::is_same<value_type, Kokkos::complex<float> >::value
                         ? "::ComplexFloat"
                         : std::is_same<value_type,
                                        Kokkos::complex<double> >::value
                               ? "::ComplexDouble"
                               : "::UnknownValueType");
    std::string name = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::TeamPolicy<DeviceType> policy(_D.extent(0) / _N_team,
                                          Kokkos::AUTO(), Kokkos::AUTO());

    size_t bytes_0 = ValuesViewType::shmem_size(_N_team, _D.extent(1));
    size_t bytes_1 = ValuesViewType::shmem_size(_N_team, 1);
    policy.set_scratch_size(0, Kokkos::PerTeam(4 * bytes_0 + 5 * bytes_1));

    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename ValuesViewType, typename IntView,
          typename VectorViewType>
void impl_test_batched_CG(const int N, const int BlkSize, const int N_team) {
  typedef typename ValuesViewType::value_type value_type;
  typedef Kokkos::Details::ArithTraits<value_type> ats;

  const int nnz = (BlkSize - 2) * 3 + 2 * 2;

  VectorViewType X("x0", N, BlkSize);
  VectorViewType R("r0", N, BlkSize);
  VectorViewType B("b", N, BlkSize);
  ValuesViewType D("D", N, nnz);
  IntView r("r", BlkSize + 1);
  IntView c("c", nnz);

  using ScalarType = typename ValuesViewType::non_const_value_type;
  using Layout     = typename ValuesViewType::array_layout;
  using EXSP       = typename ValuesViewType::execution_space;

  using MagnitudeType =
      typename Kokkos::Details::ArithTraits<ScalarType>::mag_type;
  using NormViewType = Kokkos::View<MagnitudeType *, Layout, EXSP>;

  NormViewType sqr_norm_0("sqr_norm_0", N);
  NormViewType sqr_norm_j("sqr_norm_j", N);

  Kokkos::Random_XorShift64_Pool<typename DeviceType::execution_space> random(
      13718);
  Kokkos::fill_random(X, random, value_type(1.0));
  Kokkos::fill_random(B, random, value_type(1.0));

  auto D_host = Kokkos::create_mirror_view(D);
  auto r_host = Kokkos::create_mirror_view(r);
  auto c_host = Kokkos::create_mirror_view(c);

  r_host(0) = 0;

  int current_col = 0;

  for (int i = 0; i < BlkSize; ++i) {
    r_host(i + 1) = r_host(i) + (i == 0 || i == (BlkSize - 1) ? 2 : 3);
  }
  for (int i = 0; i < nnz; ++i) {
    if (i % 3 == 0) {
      for (int l = 0; l < N; ++l) {
        D_host(l, i) = value_type(2.0);
      }
      c_host(i) = current_col;
      ++current_col;
    } else {
      for (int l = 0; l < N; ++l) {
        D_host(l, i) = value_type(-1.0);
      }
      c_host(i) = current_col;
      if (i % 3 == 1)
        --current_col;
      else
        ++current_col;
    }
  }

  Kokkos::fence();

  Kokkos::deep_copy(D, D_host);
  Kokkos::deep_copy(r, r_host);
  Kokkos::deep_copy(c, c_host);

  Kokkos::fence();

  // Compute initial norm

  Kokkos::deep_copy(R, B);

  auto sqr_norm_0_host = Kokkos::create_mirror_view(sqr_norm_0);
  auto sqr_norm_j_host = Kokkos::create_mirror_view(sqr_norm_j);
  auto R_host          = Kokkos::create_mirror_view(R);
  auto X_host          = Kokkos::create_mirror_view(X);

  Kokkos::deep_copy(R, B);
  Kokkos::deep_copy(R_host, R);
  Kokkos::deep_copy(X_host, X);

  KokkosBatched::SerialSpmv<Trans::NoTranspose>::template invoke<
      typename ValuesViewType::HostMirror, typename IntView::HostMirror,
      typename VectorViewType::HostMirror, typename VectorViewType::HostMirror,
      1>(-1, D_host, r_host, c_host, X_host, 1, R_host);
  KokkosBatched::SerialDot<Trans::NoTranspose>::invoke(
      R_host, R_host, sqr_norm_0_host);
  Functor_TestBatchedTeamCG<DeviceType, ValuesViewType, IntView,
                            VectorViewType>(D, r, c, X, B, N_team)
      .run();

  Kokkos::fence();

  Kokkos::deep_copy(R, B);
  Kokkos::deep_copy(R_host, R);
  Kokkos::deep_copy(X_host, X);

  KokkosBatched::SerialSpmv<Trans::NoTranspose>::template invoke<
      typename ValuesViewType::HostMirror, typename IntView::HostMirror,
      typename VectorViewType::HostMirror, typename VectorViewType::HostMirror,
      1>(-1, D_host, r_host, c_host, X_host, 1, R_host);
  KokkosBatched::SerialDot<Trans::NoTranspose>::invoke(R_host, R_host, sqr_norm_j_host);

  const MagnitudeType eps = 1.0e3 * ats::epsilon();

  for (int l = 0; l < N; ++l)
    EXPECT_NEAR_KK(sqr_norm_j_host(l) / sqr_norm_0_host(l), 0, eps);
}
}  // namespace TeamCG
}  // namespace Test

template <typename DeviceType, typename ValueType>
int test_batched_team_CG() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    typedef Kokkos::View<ValueType **, Kokkos::LayoutLeft, DeviceType> ViewType;
    typedef Kokkos::View<int *, Kokkos::LayoutLeft, DeviceType> IntView;
    typedef Kokkos::View<ValueType **, Kokkos::LayoutLeft, DeviceType>
        VectorViewType;

    for (int i = 3; i < 10; ++i) {
      Test::TeamCG::impl_test_batched_CG<DeviceType, ViewType, IntView,
                                         VectorViewType>(1024, i, 2);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    typedef Kokkos::View<ValueType **, Kokkos::LayoutRight, DeviceType>
        ViewType;
    typedef Kokkos::View<int *, Kokkos::LayoutRight, DeviceType> IntView;
    typedef Kokkos::View<ValueType **, Kokkos::LayoutRight, DeviceType>
        VectorViewType;

    for (int i = 3; i < 10; ++i) {
      Test::TeamCG::impl_test_batched_CG<DeviceType, ViewType, IntView,
                                         VectorViewType>(1024, i, 2);
    }
  }
#endif

  return 0;
}

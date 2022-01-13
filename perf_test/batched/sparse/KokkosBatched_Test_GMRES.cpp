#include <fstream>

/// Kokkos headers
#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"
#include "Kokkos_Random.hpp"
#include "Kokkos_UnorderedMap.hpp"
#include "Kokkos_Sort.hpp"

#define KOKKOSBATCHED_TEST_SPMV

#if defined(KOKKOSBATCHED_TEST_SPMV)

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
#include <KokkosSparse_CrsMatrix.hpp>

#include "KokkosBatched_Test_Sparse_Helper.hpp"

#include "KokkosBatched_Spmv.hpp"
#include "KokkosBatched_CrsMatrix.hpp"
#include "KokkosBatched_Krylov_Handle.hpp"
#include "KokkosBatched_GMRES.hpp"
#include "KokkosBatched_JacobiPrec.hpp"
#include "KokkosBatched_Dot.hpp"
#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Dot_Internal.hpp"
#include "KokkosBatched_Spmv_Serial_Impl.hpp"

typedef Kokkos::DefaultExecutionSpace exec_space;
typedef typename exec_space::memory_space memory_space;
typedef Kokkos::DefaultHostExecutionSpace host_space;
typedef typename Kokkos::Device<exec_space, memory_space> device;

template <typename DeviceType, typename ValuesViewType, typename IntView,
          typename VectorViewType, typename KrylovHandleType, bool UsePrec>
struct Functor_TestBatchedTeamGMRES {
  const ValuesViewType _D;
  const ValuesViewType _diag;
  const IntView _r;
  const IntView _c;
  const VectorViewType _X;
  const VectorViewType _B;
  const int _N_team, _team_size, _vector_length;
  KrylovHandleType _handle;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamGMRES(const ValuesViewType &D, const IntView &r,
                                  const IntView &c, const VectorViewType &X,
                                  const VectorViewType &B, const int N_team, const int team_size, const int vector_length, KrylovHandleType &handle)
      : _D(D), _r(r), _c(c), _X(X), _B(B), _N_team(N_team), _team_size(team_size), _vector_length(vector_length), _handle(handle) {
  }

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamGMRES(const ValuesViewType &D, const ValuesViewType &diag, const IntView &r,
                                  const IntView &c, const VectorViewType &X,
                                  const VectorViewType &B, const int N_team, const int team_size, const int vector_length, KrylovHandleType &handle)
      : _D(D), _diag(diag), _r(r), _c(c), _X(X), _B(B), _N_team(N_team), _team_size(team_size), _vector_length(vector_length), _handle(handle) {
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

    if (UsePrec) {
      auto diag = Kokkos::subview(
          _diag, Kokkos::make_pair(first_matrix, last_matrix), Kokkos::ALL);
      using PrecOperator = KokkosBatched::JacobiPrec<ValuesViewType>;
      PrecOperator P(diag);

      KokkosBatched::TeamGMRES<MemberType>::template invoke<Operator,
                                                              VectorViewType, PrecOperator>(
          member, A, b, x, P, _handle);
    }
    else {
      KokkosBatched::TeamGMRES<MemberType>::template invoke<Operator,
                                                              VectorViewType>(
          member, A, b, x, _handle);
    }
  }

  inline void run() {
    typedef typename ValuesViewType::value_type value_type;
    std::string name("KokkosBatched::Test::TeamGMRES");
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::TeamPolicy<DeviceType> policy(ceil(_D.extent(0) / _N_team), _team_size, _vector_length);

    _handle.set_max_iteration(10);
    _handle.set_tolerance(1e-8);
    int maximum_iteration = _handle.get_max_iteration();

    size_t bytes_0 = ValuesViewType::shmem_size(_N_team, _r.extent(0));
    size_t bytes_1 = ValuesViewType::shmem_size(_N_team, 1);
    policy.set_scratch_size(0, Kokkos::PerTeam(4 * bytes_0 + 3 * bytes_1));
    policy.set_scratch_size(
        1, Kokkos::PerTeam((maximum_iteration + 1) * bytes_0 +
                           ((maximum_iteration + 3) * maximum_iteration) *
                               bytes_1));

    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename ValuesViewType, typename IntView,
          typename VectorViewType, typename KrylovHandleType, bool UsePrec>
struct Functor_TestBatchedTeamVectorGMRES {
  const ValuesViewType _D;
  const ValuesViewType _diag;
  const IntView _r;
  const IntView _c;
  const VectorViewType _X;
  const VectorViewType _B;
  const int _N_team, _team_size, _vector_length;
  KrylovHandleType _handle;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorGMRES(const ValuesViewType &D, const IntView &r,
                                  const IntView &c, const VectorViewType &X,
                                  const VectorViewType &B, const int N_team, const int team_size, const int vector_length, KrylovHandleType &handle)
      : _D(D), _r(r), _c(c), _X(X), _B(B), _N_team(N_team), _team_size(team_size), _vector_length(vector_length), _handle(handle) {
  }

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorGMRES(const ValuesViewType &D, const ValuesViewType &diag, const IntView &r,
                                  const IntView &c, const VectorViewType &X,
                                  const VectorViewType &B, const int N_team, const int team_size, const int vector_length, KrylovHandleType &handle)
      : _D(D), _diag(diag), _r(r), _c(c), _X(X), _B(B), _N_team(N_team), _team_size(team_size), _vector_length(vector_length), _handle(handle) {
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

    if (UsePrec) {
      auto diag = Kokkos::subview(
          _diag, Kokkos::make_pair(first_matrix, last_matrix), Kokkos::ALL);
      using PrecOperator = KokkosBatched::JacobiPrec<ValuesViewType>;
      PrecOperator P(diag);

      KokkosBatched::TeamVectorGMRES<MemberType>::template invoke<Operator,
                                                              VectorViewType, PrecOperator>(
          member, A, b, x, P, _handle);
    }
    else {
      KokkosBatched::TeamVectorGMRES<MemberType>::template invoke<Operator,
                                                              VectorViewType>(
          member, A, b, x, _handle);
    }
  }

  inline void run() {
    typedef typename ValuesViewType::value_type value_type;
    std::string name("KokkosBatched::Test::TeamVectorGMRES");
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::TeamPolicy<DeviceType> policy(ceil(_D.extent(0) / _N_team), _team_size, _vector_length);

    _handle.set_max_iteration(10);
    _handle.set_tolerance(1e-8);
    int maximum_iteration = _handle.get_max_iteration();

    size_t bytes_0 = ValuesViewType::shmem_size(_N_team, _r.extent(0));
    size_t bytes_1 = ValuesViewType::shmem_size(_N_team, 1);
    policy.set_scratch_size(0, Kokkos::PerTeam(4 * bytes_0 + 3 * bytes_1));
    policy.set_scratch_size(
        1, Kokkos::PerTeam((maximum_iteration + 1) * bytes_0 +
                           ((maximum_iteration + 3) * maximum_iteration) *
                               bytes_1));

    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
    cudaProfilerStop();
#endif
    Kokkos::print_configuration(std::cout);

    // typedef Kokkos::Details::ArithTraits<value_type> ats;
    Kokkos::Impl::Timer timer;

    ///
    /// input arguments parsing
    ///
    int n_rep_1         = 10;    // # of repetitions
    int n_rep_2         = 1000;  // # of repetitions
    int rows_per_thread = 1;
    int team_size       = 8;
    int n_impl          = 1;
    bool layout_left    = true;
    bool layout_right   = false;
    bool use_preconditioner = false;
    bool monitor_convergence = false;

    std::string name_A = "A.mm";
    std::string name_B = "B.mm";

    std::string name_timer = "timers";
    std::string name_X     = "X";
    std::string name_conv     = "res";

    std::vector<int> impls;
    for (int i = 1; i < argc; ++i) {
      const std::string &token = argv[i];
      if (token == std::string("-A")) name_A = argv[++i];
      if (token == std::string("-B")) name_B = argv[++i];

      if (token == std::string("-X")) name_X = argv[++i];

      if (token == std::string("-res")) name_conv = argv[++i];

      if (token == std::string("-timers")) name_timer = argv[++i];

      if (token == std::string("-n1")) n_rep_1 = std::atoi(argv[++i]);
      if (token == std::string("-n2")) n_rep_2 = std::atoi(argv[++i]);
      if (token == std::string("-rows_per_thread"))
        rows_per_thread = std::atoi(argv[++i]);
      if (token == std::string("-team_size")) team_size = std::atoi(argv[++i]);
      if (token == std::string("-n_implementations"))
        n_impl = std::atoi(argv[++i]);
      if (token == std::string("-implementation"))
        impls.push_back(std::atoi(argv[++i]));
      if (token == std::string("-l")) {
        layout_left  = true;
        layout_right = false;
      }
      if (token == std::string("-r")) {
        layout_left  = false;
        layout_right = true;
      }
      if (token == std::string("-P"))
        use_preconditioner = true;
      if (token == std::string("-C"))
        monitor_convergence = true;
    }

    int N, Blk, nnz, ncols;

    int vector_length          = 8;

    readSizesFromMM(name_A, Blk, ncols, nnz, N);

    if (impls.size() == 0)
      for (int i = 0; i < n_impl; ++i) impls.push_back(i);

    // V100 L2 cache 6MB per core
    constexpr size_t LLC_CAPACITY = 80 * 6 * 1024 * 1024;
    KokkosBatched::Flush<LLC_CAPACITY, exec_space> flush;

    printf(
        " :::: Testing (N = %d, Blk = %d, nnz = %d, vl = %d, n = %d)\n",
        N, Blk, nnz, vector_length, n_rep_1);

    typedef Kokkos::LayoutRight LR;
    typedef Kokkos::LayoutLeft LL;

    using IntView            = Kokkos::View<int *, LR>;
    using AMatrixValueViewLR = Kokkos::View<double **, LR>;
    using AMatrixValueViewLL = Kokkos::View<double **, LL>;
    using XYTypeLR           = Kokkos::View<double **, LR>;
    using XYTypeLL           = Kokkos::View<double **, LL>;

    using alphaViewType = Kokkos::View<double *>;
    alphaViewType alphaV("alpha", N);
    alphaViewType betaV("alpha", N);

    IntView rowOffsets("values", Blk + 1);
    IntView colIndices("values", nnz);
    AMatrixValueViewLR valuesLR("values", N, nnz);
    AMatrixValueViewLL valuesLL("values", N, nnz);

    AMatrixValueViewLR diagLR("values", N, Blk);
    AMatrixValueViewLL diagLL("values", N, Blk);

    XYTypeLR xLR("values", N, Blk);
    XYTypeLR yLR("values", N, Blk);

    XYTypeLL xLL("values", N, Blk);
    XYTypeLL yLL("values", N, Blk);

    int rows_per_team = launch_parameters<exec_space>(Blk, nnz, rows_per_thread,
                                                      team_size, vector_length);

    if (layout_left)
      printf(
          " :::: Testing left layout (team_size = %d)\n",
          team_size);
    if (layout_right)
      printf(
          " :::: Testing right layout (team_size = %d)\n",
          team_size);

    if (layout_left) {
      readCRSFromMM(name_A, valuesLL, rowOffsets, colIndices);
      readArrayFromMM(name_B, yLL);
      if (use_preconditioner)
        getDiagFromCRS(valuesLL, rowOffsets, colIndices, diagLL);
    }
    if (layout_right) {
      readCRSFromMM(name_A, valuesLR, rowOffsets, colIndices);
      readArrayFromMM(name_B, yLR);
      if (use_preconditioner)
        getDiagFromCRS(valuesLR, rowOffsets, colIndices, diagLR);
    }

    for (auto i_impl : impls) {
      std::vector<double> timers;

      int n_skip = 2;

      int N_team = i_impl > 1 ? vector_length : 1;

      using ScalarType = typename AMatrixValueViewLL::non_const_value_type;
      using Layout     = typename AMatrixValueViewLL::array_layout;
      using EXSP       = typename AMatrixValueViewLL::execution_space;

      using MagnitudeType =
          typename Kokkos::Details::ArithTraits<ScalarType>::mag_type;
      using NormViewType = Kokkos::View<MagnitudeType *, Layout, EXSP>;
      
      using Norm2DViewType = Kokkos::View<MagnitudeType **, Layout, EXSP>;
      using IntViewType = Kokkos::View<int*, Layout, EXSP>;

      using KrylovHandleType = KokkosBatched::KrylovHandle<Norm2DViewType, IntViewType>;
      KrylovHandleType handle(N, N_team);
      
      for (int i_rep = 0; i_rep < n_rep_1 + n_skip; ++i_rep) {
        double t_spmv = 0;
        for (int j_rep = 0; j_rep < n_rep_2; ++j_rep) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
          cudaProfilerStart();
#endif
          exec_space().fence();
          Kokkos::deep_copy(xLL, 0.0);
          Kokkos::deep_copy(xLR, 0.0);
          flush.run();
          exec_space().fence();

          timer.reset();
          exec_space().fence();

          if (i_impl%2 == 0 && layout_left) {
            if (use_preconditioner)
              Functor_TestBatchedTeamGMRES<exec_space, AMatrixValueViewLL, IntView, XYTypeLL, KrylovHandleType, true>(valuesLL, diagLL, rowOffsets, colIndices, xLL, yLL, N_team, team_size, vector_length, handle)
                  .run();
            else 
              Functor_TestBatchedTeamGMRES<exec_space, AMatrixValueViewLL, IntView, XYTypeLL, KrylovHandleType, false>(valuesLL, rowOffsets, colIndices, xLL, yLL, N_team, team_size, vector_length, handle)
                  .run();
          }
          if (i_impl%2 == 1 && layout_left) {
            if (use_preconditioner)
              Functor_TestBatchedTeamVectorGMRES<exec_space, AMatrixValueViewLL, IntView, XYTypeLL, KrylovHandleType, true>(valuesLL, diagLL, rowOffsets, colIndices, xLL, yLL, N_team, team_size, vector_length, handle)
                  .run();
            else
              Functor_TestBatchedTeamVectorGMRES<exec_space, AMatrixValueViewLL, IntView, XYTypeLL, KrylovHandleType, false>(valuesLL, rowOffsets, colIndices, xLL, yLL, N_team, team_size, vector_length, handle)
                  .run();
          }
          if (i_impl%2 == 0 && layout_right) {
            if (use_preconditioner)
              Functor_TestBatchedTeamGMRES<exec_space, AMatrixValueViewLR, IntView, XYTypeLR, KrylovHandleType, true>(valuesLR, diagLR, rowOffsets, colIndices, xLR, yLR, N_team, team_size, vector_length, handle)
                  .run();
            else
              Functor_TestBatchedTeamGMRES<exec_space, AMatrixValueViewLR, IntView, XYTypeLR, KrylovHandleType, false>(valuesLR, rowOffsets, colIndices, xLR, yLR, N_team, team_size, vector_length, handle)
                  .run();
          }
          if (i_impl%2 == 1 && layout_right) {
            if (use_preconditioner)
              Functor_TestBatchedTeamVectorGMRES<exec_space, AMatrixValueViewLR, IntView, XYTypeLR, KrylovHandleType, true>(valuesLR, diagLR, rowOffsets, colIndices, xLR, yLR, N_team, team_size, vector_length, handle)
                  .run();
            else
              Functor_TestBatchedTeamVectorGMRES<exec_space, AMatrixValueViewLR, IntView, XYTypeLR, KrylovHandleType, false>(valuesLR, rowOffsets, colIndices, xLR, yLR, N_team, team_size, vector_length, handle)
                  .run();
          }
          exec_space().fence();

          if(layout_right) {
            NormViewType sqr_norm_0("sqr_norm_0", N);
            NormViewType sqr_norm_j("sqr_norm_j", N);

            auto sqr_norm_0_host = Kokkos::create_mirror_view(sqr_norm_0);
            auto sqr_norm_j_host = Kokkos::create_mirror_view(sqr_norm_j);
            auto R_host          = Kokkos::create_mirror_view(yLR);
            auto X_host          = Kokkos::create_mirror_view(xLR);
            auto D_host          = Kokkos::create_mirror_view(valuesLR);
            auto r_host          = Kokkos::create_mirror_view(rowOffsets);
            auto c_host          = Kokkos::create_mirror_view(colIndices);
            
            Kokkos::deep_copy(R_host, yLR);
            Kokkos::deep_copy(X_host, xLR);
            Kokkos::deep_copy(D_host, valuesLR);
            Kokkos::deep_copy(r_host, rowOffsets);
            Kokkos::deep_copy(c_host, colIndices);

            KokkosBatched::SerialDot<KokkosBatched::Trans::NoTranspose>::invoke(R_host, R_host,
                                                                sqr_norm_0_host);
            KokkosBatched::SerialSpmv<KokkosBatched::Trans::NoTranspose>::template invoke<
                typename AMatrixValueViewLR::HostMirror, typename IntView::HostMirror,
                typename XYTypeLR::HostMirror, typename XYTypeLR::HostMirror,
                1>(-1, D_host, r_host, c_host, X_host, 1, R_host);
            KokkosBatched::SerialDot<KokkosBatched::Trans::NoTranspose>::invoke(R_host, R_host,
                                                                sqr_norm_j_host);

            for (int l = 0; l < N; ++l)
              if (1e-8 < std::sqrt(sqr_norm_j_host(l)) / std::sqrt(sqr_norm_0_host(l)) )
                std::cout << std::setprecision (15) << "Right: System " << l << " relative residual " << std::sqrt(sqr_norm_j_host(l)) / std::sqrt(sqr_norm_0_host(l)) << " norm r_0 " << std::sqrt(sqr_norm_0_host(l)) << std::endl;
          }
          else {
            NormViewType sqr_norm_0("sqr_norm_0", N);
            NormViewType sqr_norm_j("sqr_norm_j", N);

            auto sqr_norm_0_host = Kokkos::create_mirror_view(sqr_norm_0);
            auto sqr_norm_j_host = Kokkos::create_mirror_view(sqr_norm_j);
            auto R_host          = Kokkos::create_mirror_view(yLL);
            auto X_host          = Kokkos::create_mirror_view(xLL);
            auto D_host          = Kokkos::create_mirror_view(valuesLL);
            auto r_host          = Kokkos::create_mirror_view(rowOffsets);
            auto c_host          = Kokkos::create_mirror_view(colIndices);
            
            Kokkos::deep_copy(R_host, yLL);
            Kokkos::deep_copy(X_host, xLL);
            Kokkos::deep_copy(D_host, valuesLL);
            Kokkos::deep_copy(r_host, rowOffsets);
            Kokkos::deep_copy(c_host, colIndices);

            KokkosBatched::SerialDot<KokkosBatched::Trans::NoTranspose>::invoke(R_host, R_host,
                                                                sqr_norm_0_host);
            KokkosBatched::SerialSpmv<KokkosBatched::Trans::NoTranspose>::template invoke<
                typename AMatrixValueViewLL::HostMirror, typename IntView::HostMirror,
                typename XYTypeLL::HostMirror, typename XYTypeLL::HostMirror,
                1>(-1, D_host, r_host, c_host, X_host, 1, R_host);
            KokkosBatched::SerialDot<KokkosBatched::Trans::NoTranspose>::invoke(R_host, R_host,
                                                                sqr_norm_j_host);

            for (int l = 0; l < N; ++l)
              if (1e-8 < std::sqrt(sqr_norm_j_host(l)) / std::sqrt(sqr_norm_0_host(l)) )
                std::cout << std::setprecision (15) << "Left: System " << l << " relative residual " << std::sqrt(sqr_norm_j_host(l)) / std::sqrt(sqr_norm_0_host(l)) << " norm r_0 " << std::sqrt(sqr_norm_0_host(l)) << std::endl;
          }

          t_spmv += timer.seconds();
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
          cudaProfilerStop();
#endif
        }
        if (i_rep > n_skip) timers.push_back(t_spmv / n_rep_2);
      }

      {
        std::ofstream myfile;
        std::string name;
        if (layout_left)
          name = name_timer + "_" + std::to_string(i_impl) + "_left.txt";
        if (layout_right)
          name = name_timer + "_" + std::to_string(i_impl) + "_right.txt";

        myfile.open(name);

        for (size_t i = 0; i < timers.size(); ++i) myfile << timers[i] << " ";

        myfile << std::endl;

        myfile.close();
      }

      double average_time = 0.;

      for (size_t i = 0; i < timers.size(); ++i)
        average_time += timers[i]/timers.size();


      if (layout_left)
        printf(
            "Left layout: Implementation %d: solve time = %f\n",
            i_impl, average_time);
      if (layout_right)
        printf(
            "Right layout: Implementation %d: solve time = %f\n",
            i_impl, average_time);

      if (layout_left) {
        writeArrayToMM(name_X + std::to_string(i_impl) + "_l.mm", xLL);
      }
      if (layout_right) {
        writeArrayToMM(name_X + std::to_string(i_impl) + "_r.mm", xLR);
      }
      if (monitor_convergence) {
        writeArrayToMM(name_conv + std::to_string(i_impl) + ".mm", handle.residual_norms);
      }
    }
  }
  Kokkos::finalize();

  return 0;
}

#else
int main() { return 0; }
#endif

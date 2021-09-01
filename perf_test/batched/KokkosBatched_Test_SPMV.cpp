#include <fstream>

/// Kokkos headers
#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"
#include "Kokkos_Random.hpp"
#include "Kokkos_UnorderedMap.hpp"
#include "Kokkos_Sort.hpp"

#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
#if !defined(KOKKOS_ENABLE_CUDA) || (8000 <= CUDA_VERSION)
#if defined(KOKKOS_ENABLE_CUDA_LAMBDA)
#define KOKKOSBATCHED_TEST_SPMV
#endif
#endif
#endif

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
#include <KokkosBatched_SPMV.hpp>
#include <KokkosBatched_SPMV_View.hpp>

#include "quantiles.hpp"
#include "KokkosBatched_Test_SPMV_Helper.hpp"

#include "KokkosBatched_Axpy_Decl.hpp"
#include "KokkosBatched_Axpy_Impl.hpp"

#include "KokkosBatched_Spmv_Decl.hpp"


template <typename PolicyType,
          typename DViewType,
          typename IntView,
          typename xViewType,
          typename yViewType,
          typename alphaViewType,
          typename betaViewType,
          int dobeta>
struct Functor_TestBatchedTeamVectorSpmv {
  PolicyType _policy;
  const alphaViewType _alpha;
  const DViewType _D;
  const IntView _r;
  const IntView _c;
  const xViewType _X;
  const betaViewType _beta;
  const yViewType _Y;
  int _matrices_per_team;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorSpmv(PolicyType policy,
                                    const alphaViewType &alpha,
                                    const DViewType &D,
                                    const IntView &r,
                                    const IntView &c,
                                    const xViewType &X,
                                    const betaViewType &beta,
                                    const yViewType &Y,
                                    const int matrices_per_team)
    : _policy(policy), _alpha(alpha), _D(D), _r(r), _c(c), _X(X), _beta(beta), _Y(Y), _matrices_per_team(matrices_per_team) {} 

  template<typename MemberType>
  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType &member) const {
    const int first_matrix =
        static_cast<int>(member.league_rank()) * _matrices_per_team;
    const int N = _D.extent(0);
    const int last_matrix = (static_cast<int>(member.league_rank() + 1) * _matrices_per_team < N ? static_cast<int>(member.league_rank() + 1) * _matrices_per_team : N );

    auto alpha_team = Kokkos::subview(_alpha,Kokkos::make_pair(first_matrix,last_matrix));
    auto D_team = Kokkos::subview(_D,Kokkos::make_pair(first_matrix,last_matrix),Kokkos::ALL);
    auto X_team = Kokkos::subview(_X,Kokkos::make_pair(first_matrix,last_matrix),Kokkos::ALL);
    auto beta_team = Kokkos::subview(_beta,Kokkos::make_pair(first_matrix,last_matrix));
    auto Y_team = Kokkos::subview(_Y,Kokkos::make_pair(first_matrix,last_matrix),Kokkos::ALL);

    using ScratchPadIntView =
        Kokkos::View<int*,
                      Kokkos::DefaultExecutionSpace::scratch_memory_space>;

    const int n_rows = _r.extent(0) - 1;
    const int nnz    = _c.extent(0);

    ScratchPadIntView cols(member.team_scratch(0), nnz);
    ScratchPadIntView row_map(member.team_scratch(0), n_rows + 1);

    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, 0, n_rows + 1),
        [&](const int& i) { row_map(i) = _r(i); });

    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, 0, nnz),
        [&](const int& i) { cols(i) = _c(i); });

    member.team_barrier();

    KokkosBatched::TeamVectorSpmv<MemberType,Trans::NoTranspose,Algo::Gemv::Unblocked>::template invoke<DViewType, ScratchPadIntView, xViewType, yViewType, alphaViewType, betaViewType, dobeta> (member, alpha_team, D_team, row_map, cols, X_team, beta_team, Y_team); 
  }

  inline
  void run() {
    Kokkos::parallel_for("KokkosSparse::PerfTest::BSpMV", _policy, *this);
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
    int N               = 128;  /// # of problems (batch size)
    int Blk             = 30;   /// block dimension
    int nnz_per_row     = 5;
    int n_rep_1         = 10;    // # of repetitions
    int n_rep_2         = 1000;  // # of repetitions
    int rows_per_thread = 1;
    int team_size       = 64 / vector_length;
    int n_impl          = 1;
    int offset          = 4;
    int max_offset      = 3;
    bool layout_left    = true;
    bool layout_right   = false;
    bool random         = false;
    std::vector<int> impls;
    for (int i = 1; i < argc; ++i) {
      const std::string &token = argv[i];
      if (token == std::string("-N")) N = std::atoi(argv[++i]);
      if (token == std::string("-B")) Blk = std::atoi(argv[++i]);
      if (token == std::string("-nnz_per_row"))
        nnz_per_row = std::atoi(argv[++i]);
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
      if (token == std::string("-random")) random = true;
      if (token == std::string("-offset")) offset = std::atoi(argv[++i]);
      if (token == std::string("-max_offset"))
        max_offset = std::atoi(argv[++i]);
    }

    if (impls.size() == 0)
      for (int i = 0; i < n_impl; ++i) impls.push_back(i);

    // V100 L2 cache 6MB per core
    constexpr size_t LLC_CAPACITY = 80 * 6 * 1024 * 1024;
    Flush<LLC_CAPACITY, exec_space> flush;

    printf(
        " :::: Testing (N = %d, Blk = %d, vl = %d, vi = %d, nnz_per_row = "
        "%d, n = %d)\n",
        N, Blk, vector_length, internal_vector_length, nnz_per_row, n_rep_1);

    {
      std::ofstream myfile;
      myfile.open("dimensions.txt");

      myfile << Blk << " " << N << std::endl;

      myfile.close();
    }

    int nnz = random ? Blk * nnz_per_row : getNNZ(Blk, max_offset, offset);

    using IntView = typename graph_type::row_map_type::non_const_type;
    using AMatrixValueViewLR = Kokkos::View<double **, LR>;
    using AMatrixValueViewLL = Kokkos::View<double **, LL>;
    using XYTypeLR           = Kokkos::View<double **, LR>;
    using XYTypeLL           = Kokkos::View<double **, LL>;
    using XYVTypeLR          = Kokkos::View<vector_type **, LR>;
    using XYVTypeLL          = Kokkos::View<vector_type **, LL>;

    using alphaViewType      = Kokkos::View<double *>;
    alphaViewType alphaV("alpha", N);
    alphaViewType betaV("alpha", N);

    IntView rowOffsets("values", Blk + 1);
    IntView colIndices("values", nnz);
    AMatrixValueViewLR valuesLR("values", N, nnz);
    AMatrixValueViewLL valuesLL("values", N, nnz);
    matrix_type myMatrices[N];
    vector_matrix_type myVectorMatrices[N / vector_length];

    XYTypeLR xLR("values", N, Blk);
    XYTypeLR yLR("values", N, Blk);
    XYVTypeLR xvLR("values", N / vector_length, Blk);
    XYVTypeLR yvLR("values", N / vector_length, Blk);

    XYTypeLL xLL("values", N, Blk);
    XYTypeLL yLL("values", N, Blk);
    XYVTypeLL xvLL("values", N / vector_length, Blk);
    XYVTypeLL yvLL("values", N / vector_length, Blk);

    double s_a[N], s_b[N];
    vector_type s_av[N / vector_length], s_bv[N / vector_length];

    int rows_per_team = launch_parameters<exec_space>(Blk, nnz, rows_per_thread,
                                                      team_size, vector_length);

    if (layout_left)
      printf(
          " :::: Testing left layout (team_size = %d, rows_per_thread = %d, "
          "rows_per_team = "
          "%d)\n",
          team_size, rows_per_thread, rows_per_team);
    if (layout_right)
      printf(
          " :::: Testing right layout (team_size = %d, rows_per_thread = %d, "
          "rows_per_team = "
          "%d)\n",
          team_size, rows_per_thread, rows_per_team);

    if (layout_right)
      getSPMVInputs(myMatrices, myVectorMatrices, valuesLR, rowOffsets,
                    colIndices, xLR, yLR, xvLR, yvLR, Blk, nnz, random,
                    max_offset, offset, N, s_a, s_b, s_av, s_bv);
    if (layout_left)
      getSPMVInputs(myMatrices, myVectorMatrices, valuesLL, rowOffsets,
                    colIndices, xLL, yLL, xvLL, yvLL, Blk, nnz, random,
                    max_offset, offset, N, s_a, s_b, s_av, s_bv);

    auto alphaV_h = Kokkos::create_mirror_view(alphaV);
    auto betaV_h = Kokkos::create_mirror_view(betaV);

    for (int i = 0; i < N; ++i) {
      s_a[i] = 1.;
      s_b[i] = 0.;
      alphaV_h(i) = s_a[i];
      betaV_h(i) = s_b[i];
    }

    Kokkos::deep_copy(alphaV, alphaV_h);
    Kokkos::deep_copy(betaV, betaV_h);

    using ScratchPadIntView =
        Kokkos::View<int *, exec_space::scratch_memory_space>;

    for (auto i_impl : impls) {
      std::vector<double> timers;

      int n_skip = 2;

      for (int i_rep = 0; i_rep < n_rep_1 + n_skip; ++i_rep) {
        double t_spmv = 0;
        for (int j_rep = 0; j_rep < n_rep_2; ++j_rep) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
          cudaProfilerStart();
#endif
          exec_space().fence();
          flush.run();
          exec_space().fence();

          timer.reset();
          exec_space().fence();

          // BSPMV_Functor<matrix_type, XType, YType, 0> func(
          //    s_a, myMatrices, x, s_b, y, vector_length, N, i_impl);

          int number_of_teams = i_impl == 0 ? N : N / vector_length;
          int N_team = i_impl == 0 ? 1 : vector_length;

          if (layout_left) {
            using policy_type = Kokkos::TeamPolicy<exec_space>;
            using member_type = typename policy_type::member_type;
            policy_type policy(number_of_teams, team_size, vector_length);
            size_t bytes_0 = ScratchPadIntView::shmem_size(Blk + 1);
            size_t bytes_1 = ScratchPadIntView::shmem_size(nnz);
            if (i_impl > 1)
              policy.set_scratch_size(0, Kokkos::PerTeam(bytes_0 + bytes_1));
            // policy.set_scratch_size(1, Kokkos::PerTeam(bytes_1));
            if (i_impl == 3) {
              Functor_TestBatchedTeamVectorSpmv<policy_type, AMatrixValueViewLL, IntView, XYTypeLL,  XYTypeLL, alphaViewType, alphaViewType, 0>
                (policy, alphaV, valuesLL, rowOffsets, colIndices, xLL, betaV, yLL, N_team).run();
            }
            else {
              Kokkos::parallel_for(
                  "KokkosSparse::PerfTest::BSpMV", policy,
                  BSPMV_Functor_View<AMatrixValueViewLL, IntView, XYTypeLL,
                                      XYTypeLL, 0>(s_a, valuesLL, rowOffsets,
                                                  colIndices, xLL, s_b, yLL,
                                                  N_team, N, i_impl));
            }
          }
          if (layout_right) {
            using policy_type = Kokkos::TeamPolicy<exec_space>;
            using member_type = typename policy_type::member_type;
            policy_type policy(number_of_teams, team_size, vector_length);
            size_t bytes_0 = ScratchPadIntView::shmem_size(Blk + 1);
            size_t bytes_1 = ScratchPadIntView::shmem_size(nnz);
            if (i_impl > 1)
              policy.set_scratch_size(0, Kokkos::PerTeam(bytes_0 + bytes_1));
            // policy.set_scratch_size(1, Kokkos::PerTeam(bytes_1));
            if (i_impl == 3) {
              Functor_TestBatchedTeamVectorSpmv<policy_type, AMatrixValueViewLR, IntView, XYTypeLR,  XYTypeLR, alphaViewType, alphaViewType, 0>
                (policy, alphaV, valuesLR, rowOffsets, colIndices, xLR, betaV, yLR, N_team).run();
            }
            else {
              Kokkos::parallel_for(
                  "KokkosSparse::PerfTest::BSpMV", policy,
                  BSPMV_Functor_View<AMatrixValueViewLR, IntView, XYTypeLR,
                                      XYTypeLR, 0>(s_a, valuesLR, rowOffsets,
                                                  colIndices, xLR, s_b, yLR,
                                                  N_team, N, i_impl));
            }
          }
          exec_space().fence();
          t_spmv += timer.seconds();
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
          cudaProfilerStop();
#endif
        }
        if (i_rep > n_skip) timers.push_back(t_spmv / n_rep_2);
      }
      int median_id = 3;
      auto quantiles =
          Quantile<double>(timers, {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99});

      if (layout_left)
        printf(
            "Left layout: Implementation %d: solve time = %f , # of SPMV per "
            "min = %f\n",
            i_impl, quantiles[median_id], 1.0 / quantiles[median_id] * 60 * N);
      if (layout_right)
        printf(
            "Right layout: Implementation %d: solve time = %f , # of SPMV per "
            "min = %f\n",
            i_impl, quantiles[median_id], 1.0 / quantiles[median_id] * 60 * N);
      {
        std::ofstream myfile;
        std::string name;
        if (layout_left) name = "timer_" + std::to_string(i_impl) + "_left.txt";
        if (layout_right)
          name = "timer_" + std::to_string(i_impl) + "_right.txt";

        myfile.open(name);

        for (size_t i = 0; i < quantiles.size(); ++i)
          myfile << quantiles[i] << " ";

        myfile << std::endl;

        myfile.close();
      }

      {
        std::ofstream myfile;
        std::string name = "nnz.txt";

        myfile.open(name);

        myfile << nnz << std::endl;

        myfile.close();
      }

      if (layout_left) {
        writeArrayTofile(yLL, "y_" + std::to_string(i_impl) + "_l.txt");
        writeArrayTofile(xLL, "x_l.txt");
      }
      if (layout_right) {
        writeArrayTofile(yLR, "y_" + std::to_string(i_impl) + "_r.txt");
        writeArrayTofile(xLR, "x_r.txt");
      }
    }
  }
  Kokkos::finalize();

  return 0;
}

#else
int main() { return 0; }
#endif

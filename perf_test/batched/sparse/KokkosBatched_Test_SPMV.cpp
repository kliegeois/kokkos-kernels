#include <fstream>

/// Kokkos headers
#include "Kokkos_Core.hpp"
#include "Kokkos_Timer.hpp"
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
#include <KokkosBatched_SPMV_View.hpp>

#include "KokkosBatched_Test_Sparse_Helper.hpp"

#include "KokkosBatched_Spmv.hpp"

typedef Kokkos::DefaultExecutionSpace exec_space;
typedef typename exec_space::memory_space memory_space;
typedef Kokkos::DefaultHostExecutionSpace host_space;
typedef typename Kokkos::Device<exec_space, memory_space> device;

template <typename PolicyType, typename DViewType, typename IntView,
          typename xViewType, typename yViewType, typename alphaViewType,
          typename betaViewType, int dobeta>
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
  Functor_TestBatchedTeamVectorSpmv(
      PolicyType policy, const alphaViewType &alpha, const DViewType &D,
      const IntView &r, const IntView &c, const xViewType &X,
      const betaViewType &beta, const yViewType &Y, const int matrices_per_team)
      : _policy(policy),
        _alpha(alpha),
        _D(D),
        _r(r),
        _c(c),
        _X(X),
        _beta(beta),
        _Y(Y),
        _matrices_per_team(matrices_per_team) {}

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType &member) const {
    const int first_matrix =
        static_cast<int>(member.league_rank()) * _matrices_per_team;
    const int N = _D.extent(0);
    const int last_matrix =
        (static_cast<int>(member.league_rank() + 1) * _matrices_per_team < N
             ? static_cast<int>(member.league_rank() + 1) * _matrices_per_team
             : N);

    auto alpha_team =
        Kokkos::subview(_alpha, Kokkos::make_pair(first_matrix, last_matrix));
    auto D_team = Kokkos::subview(
        _D, Kokkos::make_pair(first_matrix, last_matrix), Kokkos::ALL);
    auto X_team = Kokkos::subview(
        _X, Kokkos::make_pair(first_matrix, last_matrix), Kokkos::ALL);
    auto beta_team =
        Kokkos::subview(_beta, Kokkos::make_pair(first_matrix, last_matrix));
    auto Y_team = Kokkos::subview(
        _Y, Kokkos::make_pair(first_matrix, last_matrix), Kokkos::ALL);

    using ScratchPadIntView =
        Kokkos::View<int *,
                     Kokkos::DefaultExecutionSpace::scratch_memory_space>;

    const int n_rows = _r.extent(0) - 1;
    const int nnz    = _c.extent(0);

    ScratchPadIntView cols(member.team_scratch(0), nnz);
    ScratchPadIntView row_map(member.team_scratch(0), n_rows + 1);

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, 0, n_rows + 1),
                         [&](const int &i) { row_map(i) = _r(i); });

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, 0, nnz),
                         [&](const int &i) { cols(i) = _c(i); });

    member.team_barrier();

    if (last_matrix != N && _matrices_per_team == 8)
      KokkosBatched::TeamVectorSpmv<
          MemberType, KokkosBatched::Trans::NoTranspose,
          8>::template invoke<DViewType, ScratchPadIntView, xViewType,
                              yViewType, alphaViewType, betaViewType, dobeta>(
          member, alpha_team, D_team, row_map, cols, X_team, beta_team, Y_team);
    else
      KokkosBatched::TeamVectorSpmv<
          MemberType, KokkosBatched::Trans::NoTranspose,
          0>::template invoke<DViewType, ScratchPadIntView, xViewType,
                              yViewType, alphaViewType, betaViewType, dobeta>(
          member, alpha_team, D_team, row_map, cols, X_team, beta_team, Y_team);
  }

  inline void run() {
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
    Kokkos::Timer timer;

    ///
    /// input arguments parsing
    ///
    int n_rep_1          = 10;    // # of repetitions
    int n_rep_2          = 1000;  // # of repetitions
    int team_size        = 8;
    int vector_length    = 8;
    int N_team_potential = 8;
    int n_impl           = 1;
    bool layout_left     = true;
    bool layout_right    = false;

    std::string name_A = "A.mm";
    std::string name_B = "B.mm";

    std::string name_timer = "timers";
    std::string name_X     = "X";

    std::vector<int> impls;
    for (int i = 1; i < argc; ++i) {
      const std::string &token = argv[i];
      if (token == std::string("-A")) name_A = argv[++i];
      if (token == std::string("-B")) name_B = argv[++i];

      if (token == std::string("-X")) name_X = argv[++i];

      if (token == std::string("-timers")) name_timer = argv[++i];

      if (token == std::string("-n1")) n_rep_1 = std::atoi(argv[++i]);
      if (token == std::string("-n2")) n_rep_2 = std::atoi(argv[++i]);
      if (token == std::string("-vector_length"))
        vector_length = std::atoi(argv[++i]);
      if (token == std::string("-N_team"))
        N_team_potential = std::atoi(argv[++i]);
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
    }

    int N, Blk, nnz, ncols;

    int internal_vector_length = 2;

    readSizesFromMM(name_A, Blk, ncols, nnz, N);

    if (impls.size() == 0)
      for (int i = 0; i < n_impl; ++i) impls.push_back(i);

    // V100 L2 cache 6MB per core
    constexpr size_t LLC_CAPACITY = 80 * 6 * 1024 * 1024;
    KokkosBatched::Flush<LLC_CAPACITY, exec_space> flush;

    printf(
        " :::: Testing (N = %d, Blk = %d, nnz = %d, vl = %d, vi = %d, n = "
        "%d, N_team_potential = %d)\n",
        N, Blk, nnz, vector_length, internal_vector_length, n_rep_1, N_team_potential);

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

    XYTypeLR xLR("values", N, Blk);
    XYTypeLR yLR("values", N, Blk);

    XYTypeLL xLL("values", N, Blk);
    XYTypeLL yLL("values", N, Blk);

    double *s_a = new double[N];
    double *s_b = new double[N];

    if (layout_left)
      printf(" :::: Testing left layout (team_size = %d)\n", team_size);
    if (layout_right)
      printf(" :::: Testing right layout (team_size = %d)\n", team_size);

    if (layout_left) {
      readCRSFromMM(name_A, valuesLL, rowOffsets, colIndices);
      readArrayFromMM(name_B, xLL);
    }
    if (layout_right) {
      readCRSFromMM(name_A, valuesLR, rowOffsets, colIndices);
      readArrayFromMM(name_B, xLR);
    }

    auto alphaV_h = Kokkos::create_mirror_view(alphaV);
    auto betaV_h  = Kokkos::create_mirror_view(betaV);

    for (int i = 0; i < N; ++i) {
      s_a[i]      = 1.;
      s_b[i]      = 0.;
      alphaV_h(i) = s_a[i];
      betaV_h(i)  = s_b[i];
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
          if (n_rep_2 != 1) flush.run();
          exec_space().fence();

          timer.reset();
          exec_space().fence();

          int N_team = i_impl == 0 ? 1 : N_team_potential;
          N_team = N_team_potential;
          int number_of_teams =
              i_impl == 0 ? N : ceil(1. * N / N_team);

          if (layout_left) {
            using policy_type = Kokkos::TeamPolicy<exec_space>;
            policy_type auto_policy(number_of_teams, Kokkos::AUTO(), Kokkos::AUTO());
            policy_type tuned_policy(number_of_teams, team_size, vector_length);
            policy_type policy;

            if (team_size < 1)
              policy = auto_policy;
            else
              policy = tuned_policy;

            size_t bytes_0 = ScratchPadIntView::shmem_size(Blk + 1);
            size_t bytes_1 = ScratchPadIntView::shmem_size(nnz);
            if (i_impl > 1)
              policy.set_scratch_size(0, Kokkos::PerTeam(bytes_0 + bytes_1));
            // policy.set_scratch_size(1, Kokkos::PerTeam(bytes_1));
            if (i_impl == 3) {
              Functor_TestBatchedTeamVectorSpmv<
                  policy_type, AMatrixValueViewLL, IntView, XYTypeLL, XYTypeLL,
                  alphaViewType, alphaViewType, 0>(policy, alphaV, valuesLL,
                                                   rowOffsets, colIndices, xLL,
                                                   betaV, yLL, N_team)
                  .run();
            } else {
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
            policy_type auto_policy(number_of_teams, Kokkos::AUTO(), Kokkos::AUTO());
            policy_type tuned_policy(number_of_teams, team_size, vector_length);
            policy_type policy;

            if (team_size < 1)
              policy = auto_policy;
            else
              policy = tuned_policy;

            size_t bytes_0 = ScratchPadIntView::shmem_size(Blk + 1);
            size_t bytes_1 = ScratchPadIntView::shmem_size(nnz);
            if (i_impl > 1)
              policy.set_scratch_size(0, Kokkos::PerTeam(bytes_0 + bytes_1));
            // policy.set_scratch_size(1, Kokkos::PerTeam(bytes_1));
            if (i_impl == 3) {
              Functor_TestBatchedTeamVectorSpmv<
                  policy_type, AMatrixValueViewLR, IntView, XYTypeLR, XYTypeLR,
                  alphaViewType, alphaViewType, 0>(policy, alphaV, valuesLR,
                                                   rowOffsets, colIndices, xLR,
                                                   betaV, yLR, N_team)
                  .run();
            } else {
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
        average_time += timers[i] / timers.size();

      if (layout_left)
        printf(
            "Left layout: Implementation %d: solve time = %f , # of SPMV per "
            "min = %f\n",
            i_impl, average_time, 1.0 / average_time * 60 * N);
      if (layout_right)
        printf(
            "Right layout: Implementation %d: solve time = %f , # of SPMV per "
            "min = %f\n",
            i_impl, average_time, 1.0 / average_time * 60 * N);

      if (layout_left) {
        writeArrayToMM(name_X + std::to_string(i_impl) + "_l.mm", xLL);
      }
      if (layout_right) {
        writeArrayToMM(name_X + std::to_string(i_impl) + "_r.mm", xLR);
      }
    }
  }
  Kokkos::finalize();

  return 0;
}

#else
int main() { return 0; }
#endif

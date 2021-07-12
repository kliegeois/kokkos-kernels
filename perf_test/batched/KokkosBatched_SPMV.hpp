template <class AMatrix, class XVector, class YVector, int dobeta>
struct BSPMV_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef typename AMatrix::staticcrsgraph_type::entries_type entries_type;
  typedef Kokkos::Details::ArithTraits<value_type> ATV;

  const value_type* alpha;
  const AMatrix* m_A;
  XVector m_x;
  const value_type* beta;
  YVector m_y;
  const int N;
  const int implementation;

  const ordinal_type matrices_per_team;
  typename entries_type::non_const_type row_indices;

  BSPMV_Functor(const value_type* alpha_, const AMatrix* m_A_,
                const XVector m_x_, const value_type* beta_, const YVector m_y_,
                const int matrices_per_team_, const int N_,
                const int implementation_ = 0)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        beta(beta_),
        m_y(m_y_),
        matrices_per_team(matrices_per_team_),
        N(N_),
        implementation(implementation_) {
    static_assert(static_cast<int>(XVector::rank) == 2,
                  "XVector must be a rank 2 View.");
    static_assert(static_cast<int>(YVector::rank) == 2,
                  "YVector must be a rank 2 View.");
    if (implementation > 1) {
      Kokkos::resize(row_indices, m_A[0].nnz());
      typename entries_type::HostMirror row_indices_h =
          Kokkos::create_mirror_view(row_indices);
      typename entries_type::HostMirror row_map_tmp_h =
          Kokkos::create_mirror_view(m_A[0].graph.row_map);
      Kokkos::deep_copy(row_map_tmp_h, m_A[0].graph.row_map);
      for (int irow = 0; irow < m_A[0].numRows(); ++irow)
        for (int iEntry = row_map_tmp_h(irow); iEntry < row_map_tmp_h(irow + 1);
             ++iEntry)
          row_indices_h(iEntry) = irow;
      Kokkos::deep_copy(row_indices, row_indices_h);
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const team_member& dev) const {
    if (implementation == 0) {
      for (int i_matrix =
               static_cast<int>(dev.league_rank()) * matrices_per_team;
           i_matrix <
           min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
           ++i_matrix) {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(dev, 0, m_A[i_matrix].numRows()),
            [&](const ordinal_type& iRow) {
              const KokkosSparse::SparseRowViewConst<AMatrix> row =
                  m_A[i_matrix].rowConst(iRow);
              const ordinal_type row_length =
                  static_cast<ordinal_type>(row.length);
              value_type sum = 0;

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(dev, row_length),
                  [&](const ordinal_type& iEntry, value_type& lsum) {
                    const value_type val = row.value(iEntry);
                    lsum += val * m_x(i_matrix, row.colidx(iEntry));
                  },
                  sum);

              Kokkos::single(Kokkos::PerThread(dev), [&]() {
                sum *= alpha[i_matrix];

                if (dobeta == 0) {
                  m_y(i_matrix, iRow) = sum;
                } else {
                  m_y(i_matrix, iRow) =
                      beta[i_matrix] * m_y(i_matrix, iRow) + sum;
                }
              });
            });
      }
    }
    if (implementation == 1) {
      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices = last_matrix - first_matrix;

      Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(dev, n_matrices),
          [&](const ordinal_type& iMatrix) {
            const int iGlobalMatrix = first_matrix + iMatrix;
            const auto A            = m_A[iGlobalMatrix];

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(dev, 0, A.numRows()),
                [&](const ordinal_type& iRow) {
                  const KokkosSparse::SparseRowViewConst<AMatrix> row =
                      A.rowConst(iRow);
                  const ordinal_type row_length =
                      static_cast<ordinal_type>(row.length);
                  value_type sum = 0;

                  for (int iEntry = 0; iEntry < row_length; ++iEntry) {
                    sum += row.value(iEntry) *
                           m_x(iGlobalMatrix, row.colidx(iEntry));
                  }

                  sum *= alpha[iGlobalMatrix];

                  if (dobeta == 0) {
                    m_y(iGlobalMatrix, iRow) = sum;
                  } else {
                    m_y(iGlobalMatrix, iRow) =
                        beta[iGlobalMatrix] * m_y(iGlobalMatrix, iRow) + sum;
                  }
                });
          });
    }
    if (implementation == 2) {
      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices = last_matrix - first_matrix;

      const auto A = m_A[0];

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(dev, 0, A.numRows()),
          [&](const ordinal_type& iRow) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(dev, n_matrices),
                [&](const ordinal_type& iMatrix) {
                  const int iGlobalMatrix = first_matrix + iMatrix;

                  const KokkosSparse::SparseRowViewConst<AMatrix> row =
                      m_A[iGlobalMatrix].rowConst(iRow);
                  const ordinal_type row_length =
                      static_cast<ordinal_type>(row.length);
                  value_type sum = 0;

                  for (int iEntry = 0; iEntry < row_length; ++iEntry) {
                    sum += row.value(iEntry) *
                           m_x(iGlobalMatrix, row.colidx(iEntry));
                  }

                  Kokkos::single(Kokkos::PerThread(dev), [&]() {
                    sum *= alpha[iGlobalMatrix];

                    if (dobeta == 0) {
                      m_y(iGlobalMatrix, iRow) = sum;
                    } else {
                      m_y(iGlobalMatrix, iRow) =
                          beta[iGlobalMatrix] * m_y(iGlobalMatrix, iRow) + sum;
                    }
                  });
                });
          });
    }
    if (implementation == 3) {
      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices = last_matrix - first_matrix;

      using shared_value_array_type =
          Kokkos::View<value_type**,
                       Kokkos::DefaultExecutionSpace::scratch_memory_space>;

      shared_value_array_type sum(dev.team_scratch(0), m_A[0].numRows(),
                                  n_matrices);

      Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(dev, n_matrices),
          [&](const ordinal_type& iMatrix) {
            const int iGlobalMatrix = first_matrix + iMatrix;
            const auto A            = m_A[iGlobalMatrix];

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(dev, 0, A.numRows()),
                [&](const ordinal_type& iRow) { sum(iRow, iMatrix) = 0.0; });

            dev.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(dev, 0, A.nnz()),
                [&](const ordinal_type& iEntry) {
                  int col = A.graph.entries(iEntry);
                  int row = row_indices(iEntry);

                  Kokkos::atomic_fetch_add(
                      &sum(row, iMatrix),
                      A.values(iEntry) * m_x(iGlobalMatrix, col));
                });

            dev.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(dev, 0, A.numRows()),
                [&](const ordinal_type& iRow) {
                  sum(iRow, iMatrix) *= alpha[iGlobalMatrix];

                  if (dobeta == 0) {
                    m_y(iGlobalMatrix, iRow) = sum(iRow, iMatrix);
                  } else {
                    m_y(iGlobalMatrix, iRow) =
                        beta[iGlobalMatrix] * m_y(iGlobalMatrix, iRow) +
                        sum(iRow, iMatrix);
                  }
                });
          });
    }
    if (implementation == 4) {
      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices = last_matrix - first_matrix;

      using ScratchPadView =
          Kokkos::View<value_type**,
                       Kokkos::DefaultExecutionSpace::scratch_memory_space>;

      const auto A      = m_A[0];
      const int numRows = A.numRows();
      const int nnz     = A.nnz();

      ScratchPadView sum(dev.team_scratch(0), numRows, n_matrices);

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(dev, 0, A.numRows()),
          [&](const ordinal_type& iRow) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(dev, n_matrices),
                [&](const ordinal_type& iMatrix) { sum(iRow, iMatrix) = 0.0; });
          });

      dev.team_barrier();

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(dev, 0, nnz),
          [&](const ordinal_type& iEntry) {
            int col = A.graph.entries(iEntry);
            int row = row_indices(iEntry);
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(dev, n_matrices),
                [&](const ordinal_type& iMatrix) {
                  const int iGlobalMatrix = first_matrix + iMatrix;
                  Kokkos::atomic_fetch_add(&sum(row, iMatrix),
                                           m_A[iGlobalMatrix].values(iEntry) *
                                               m_x(iGlobalMatrix, col));
                });
          });

      dev.team_barrier();

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(dev, 0, A.numRows()),
          [&](const ordinal_type& iRow) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(dev, n_matrices),
                [&](const ordinal_type& iMatrix) {
                  const int iGlobalMatrix = first_matrix + iMatrix;
                  sum(iRow, iMatrix) *= alpha[iGlobalMatrix];

                  if (dobeta == 0) {
                    m_y(iGlobalMatrix, iRow) = sum(iRow, iMatrix);
                  } else {
                    m_y(iGlobalMatrix, iRow) =
                        beta[iGlobalMatrix] * m_y(iGlobalMatrix, iRow) +
                        sum(iRow, iMatrix);
                  }
                });
          });
    }
    if (implementation == 5) {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(dev, 0, m_A[0].numRows()),
          [&](const ordinal_type& iRow) {
            for (int i_matrix =
                     static_cast<int>(dev.league_rank()) * matrices_per_team;
                 i_matrix < min(static_cast<int>(dev.league_rank() + 1) *
                                    matrices_per_team,
                                N);
                 ++i_matrix) {
              value_type sum = 0;

              const KokkosSparse::SparseRowViewConst<AMatrix> row =
                  m_A[i_matrix].rowConst(iRow);
              const ordinal_type row_length =
                  static_cast<ordinal_type>(row.length);

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(dev, row_length),
                  [&](const ordinal_type& iEntry, value_type& lsum) {
                    const value_type val = row.value(iEntry);
                    lsum += val * m_x(i_matrix, row.colidx(iEntry));
                  },
                  sum);

              Kokkos::single(Kokkos::PerThread(dev), [&]() {
                sum *= alpha[i_matrix];

                if (dobeta == 0) {
                  m_y(i_matrix, iRow) = sum;
                } else {
                  m_y(i_matrix, iRow) =
                      beta[i_matrix] * m_y(i_matrix, iRow) + sum;
                }
              });
            }
          });
    }
  }
};
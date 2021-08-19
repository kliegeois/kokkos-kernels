template <class AMatrix, class IntView, class XVector, class YVector,
          int dobeta>
struct BSPMV_Functor_View {
  typedef typename AMatrix::execution_space exec_space;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename IntView::non_const_value_type ordinal_type;
  typedef typename Kokkos::TeamPolicy<exec_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef typename AMatrix::non_const_value_type entries_type;
  typedef Kokkos::Details::ArithTraits<value_type> ATV;

  const value_type* alpha;
  const AMatrix m_A_values;
  const IntView m_A_row_ptr;
  const IntView m_A_col_indices;
  XVector m_x;
  const value_type* beta;
  YVector m_y;
  const int N;
  int implementation;

  const ordinal_type matrices_per_team;

  BSPMV_Functor_View(const value_type* alpha_, const AMatrix m_A_values_,
                     const IntView m_A_row_ptr_, const IntView m_A_col_indices_,
                     const XVector m_x_, const value_type* beta_,
                     const YVector m_y_, const int matrices_per_team_,
                     const int N_, const int implementation_ = 0)
      : alpha(alpha_),
        m_A_values(m_A_values_),
        m_A_row_ptr(m_A_row_ptr_),
        m_A_col_indices(m_A_col_indices_),
        m_x(m_x_),
        beta(beta_),
        m_y(m_y_),
        matrices_per_team(matrices_per_team_),
        N(N_),
        implementation(implementation_) {
    static_assert(static_cast<int>(AMatrix::rank) == 2,
                  "AMatrix must be a rank 2 View.");
    static_assert(static_cast<int>(IntView::rank) == 1,
                  "IntView must be a rank 1 View.");
    static_assert(static_cast<int>(XVector::rank) == 2,
                  "XVector must be a rank 2 View.");
    static_assert(static_cast<int>(YVector::rank) == 2,
                  "YVector must be a rank 2 View.");
  }

  KOKKOS_INLINE_FUNCTION void getIndices(const ordinal_type iTemp, const ordinal_type n_rows, const ordinal_type n_matrices, ordinal_type &iRow, ordinal_type &iMatrix) const {
    if (std::is_same<typename AMatrix::array_layout, Kokkos::LayoutLeft>::value) {
      iRow    = iTemp / n_matrices;
      iMatrix = iTemp % n_matrices;
    }
    else {
      iRow    = iTemp % n_rows;
      iMatrix = iTemp / n_rows;
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const team_member& dev) const {
    if (implementation == 0) {
      const ordinal_type n_rows = m_A_row_ptr.extent(0) - 1;
      int i_matrix = static_cast<int>(dev.league_rank());
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(dev, 0, n_rows),
          [&](const ordinal_type& iRow) {
            const ordinal_type row_length =
                m_A_row_ptr(iRow + 1) - m_A_row_ptr(iRow);
            value_type sum = 0;

            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(dev, row_length),
                [&](const ordinal_type& iEntry, value_type& lsum) {
                  const value_type val =
                      m_A_values(m_A_row_ptr(iRow) + iEntry, i_matrix);
                  lsum +=
                      val * m_x(m_A_col_indices(m_A_row_ptr(iRow) + iEntry), i_matrix);
                },
                sum);

            Kokkos::single(Kokkos::PerThread(dev), [&]() {
              sum *= alpha[i_matrix];

              if (dobeta == 0) {
                m_y(iRow, i_matrix) = sum;
              } else {
                m_y(iRow, i_matrix) =
                    beta[i_matrix] * m_y(iRow, i_matrix) + sum;
              }
            });
          });
    }
    if (implementation == 1) {
      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices      = last_matrix - first_matrix;
      const ordinal_type n_rows = m_A_row_ptr.extent(0) - 1;

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(dev, 0, n_rows * n_matrices),
          [&](const ordinal_type& iTemp) {
            ordinal_type iRow, iMatrix;
            this->getIndices(iTemp, n_rows, n_matrices, iRow, iMatrix);
            const int iGlobalMatrix     = first_matrix + iMatrix;

            const ordinal_type row_length =
                m_A_row_ptr(iRow + 1) - m_A_row_ptr(iRow);
            value_type sum = 0;

#pragma unroll
            for (int iEntry = 0; iEntry < row_length; ++iEntry) {
              sum += m_A_values(m_A_row_ptr(iRow) + iEntry, iGlobalMatrix) *
                     m_x(m_A_col_indices(m_A_row_ptr(iRow) + iEntry), iGlobalMatrix);
            }

            sum *= alpha[iGlobalMatrix];

            if (dobeta == 0) {
              m_y(iRow, iGlobalMatrix) = sum;
            } else {
              m_y(iRow, iGlobalMatrix) =
                  beta[iGlobalMatrix] * m_y(iRow, iGlobalMatrix) + sum;
            }
          });
    }
    if (implementation == 2) {
      using ScratchPadIntView =
          Kokkos::View<int*,
                       Kokkos::DefaultExecutionSpace::scratch_memory_space>;

      const ordinal_type n_rows = m_A_row_ptr.extent(0) - 1;
      const ordinal_type nnz    = m_A_col_indices.extent(0);

      ScratchPadIntView cols(dev.team_scratch(0), nnz);
      ScratchPadIntView row_map(dev.team_scratch(0), n_rows + 1);

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(dev, 0, n_rows + 1),
          [&](const ordinal_type& i) { row_map(i) = m_A_row_ptr(i); });

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(dev, 0, nnz),
          [&](const ordinal_type& i) { cols(i) = m_A_col_indices(i); });

      dev.team_barrier();

      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices = last_matrix - first_matrix;

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(dev, 0, n_rows * n_matrices),
          [&](const ordinal_type& iTemp) {
            ordinal_type iRow, iMatrix;
            this->getIndices(iTemp, n_rows, n_matrices, iRow, iMatrix);
            const int iGlobalMatrix     = first_matrix + iMatrix;

            const ordinal_type row_length = row_map(iRow + 1) - row_map(iRow);
            value_type sum                = 0;

#pragma unroll
            for (int iEntry = 0; iEntry < row_length; ++iEntry) {
              sum += m_A_values(row_map(iRow) + iEntry, iGlobalMatrix) *
                     m_x(cols(row_map(iRow) + iEntry), iGlobalMatrix);
            }

            sum *= alpha[iGlobalMatrix];

            if (dobeta == 0) {
              m_y(iRow, iGlobalMatrix) = sum;
            } else {
              m_y(iRow, iGlobalMatrix) =
                  beta[iGlobalMatrix] * m_y(iRow, iGlobalMatrix) + sum;
            }
          });
    }
  }
};
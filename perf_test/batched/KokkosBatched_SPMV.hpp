template <class AMatrix, class XVector, class YVector, int dobeta>
struct BSPMV_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef Kokkos::Details::ArithTraits<value_type> ATV;

  const value_type* alpha;
  const AMatrix* m_A;
  XVector m_x;
  const value_type* beta;
  YVector m_y;
  const int N;
  const int implementation;

  const ordinal_type rows_per_team;

  BSPMV_Functor(const value_type* alpha_, const AMatrix* m_A_,
                const XVector m_x_, const value_type* beta_, const YVector m_y_,
                const int rows_per_team_, const int N_,
                const int implementation_ = 0)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        beta(beta_),
        m_y(m_y_),
        rows_per_team(rows_per_team_),
        N(N_),
        implementation(implementation_) {
    static_assert(static_cast<int>(XVector::rank) == 2,
                  "XVector must be a rank 2 View.");
    static_assert(static_cast<int>(YVector::rank) == 2,
                  "YVector must be a rank 2 View.");
  }

  KOKKOS_INLINE_FUNCTION void operator()(const team_member& dev) const {
    if (implementation == 0) {
      for (int i_matrix = 0; i_matrix < N; ++i_matrix) {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(dev, 0, rows_per_team),
            [&](const ordinal_type& loop) {
              const ordinal_type iRow =
                  static_cast<ordinal_type>(dev.league_rank()) * rows_per_team +
                  loop;
              if (iRow >= m_A[i_matrix].numRows()) {
                return;
              }
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
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(dev, 0, rows_per_team),
          [&](const ordinal_type& loop) {
            const ordinal_type iRow =
                static_cast<ordinal_type>(dev.league_rank()) * rows_per_team +
                loop;
            if (iRow >= m_A[0].numRows()) {
              return;
            }

            for (int i_matrix = 0; i_matrix < N; ++i_matrix) {
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
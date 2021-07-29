template <class AMatrix, class IntView, class XVector, class YVector, int dobeta>
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
  const int implementation;

  const ordinal_type matrices_per_team;

  BSPMV_Functor_View(const value_type* alpha_, const AMatrix m_A_values_, const IntView m_A_row_ptr_, const IntView m_A_col_indices_,
                const XVector m_x_, const value_type* beta_, const YVector m_y_,
                const int matrices_per_team_, const int N_,
                const int implementation_ = 0)
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

  KOKKOS_INLINE_FUNCTION void operator()(const team_member& dev) const {
    if (implementation == 0) {
      const ordinal_type n_rows = m_A_row_ptr.extent(0)-1;
      for (int i_matrix =
               static_cast<int>(dev.league_rank()) * matrices_per_team;
           i_matrix <
           min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
           ++i_matrix) {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(dev, 0, n_rows),
            [&](const ordinal_type& iRow) {
              const ordinal_type row_length = m_A_row_ptr(iRow+1) - m_A_row_ptr(iRow);
              value_type sum = 0;

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(dev, row_length),
                  [&](const ordinal_type& iEntry, value_type& lsum) {
                    const value_type val = m_A_values(i_matrix, m_A_row_ptr(iRow)+iEntry);
                    lsum += val * m_x(i_matrix, m_A_col_indices(m_A_row_ptr(iRow)+iEntry));
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

      const ordinal_type n_rows = m_A_row_ptr.extent(0)-1;

      Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(dev, n_matrices),
          [&](const ordinal_type& iMatrix) {
            const int iGlobalMatrix = first_matrix + iMatrix;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(dev, 0, n_rows),
                [&](const ordinal_type& iRow) {
                  const ordinal_type row_length = m_A_row_ptr(iRow+1) - m_A_row_ptr(iRow);
                  value_type sum = 0;

                  for (int iEntry = 0; iEntry < row_length; ++iEntry) {
                    sum += m_A_values(iGlobalMatrix, m_A_row_ptr(iRow)+iEntry) *
                           m_x(iGlobalMatrix, m_A_col_indices(m_A_row_ptr(iRow)+iEntry));
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

      const ordinal_type n_rows = m_A_row_ptr.extent(0)-1;

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(dev, 0, n_rows),
          [&](const ordinal_type& iRow) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(dev, n_matrices),
                [&](const ordinal_type& iMatrix) {
                  const int iGlobalMatrix = first_matrix + iMatrix;

                  const ordinal_type row_length = m_A_row_ptr(iRow+1) - m_A_row_ptr(iRow);
                  value_type sum = 0;

                  for (int iEntry = 0; iEntry < row_length; ++iEntry) {
                    sum += m_A_values(iGlobalMatrix, m_A_row_ptr(iRow)+iEntry) *
                           m_x(iGlobalMatrix, m_A_col_indices(m_A_row_ptr(iRow)+iEntry));
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
    if (implementation == 3) {
      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices = last_matrix - first_matrix;
      const ordinal_type n_rows = m_A_row_ptr.extent(0)-1;

      //using MDPolicyType_2D = typename Kokkos::Experimental::MDRangePolicy<
      //    Kokkos::Experimental::Rank<2> >;
      //MDPolicyType_2D policyInit({0,0}, {n_rows, n_matrices});

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(dev, 0, n_rows * n_matrices),
          [&](const ordinal_type& iTemp) {
            const ordinal_type& iRow = iTemp/n_matrices;
            const ordinal_type& iMatrix = iTemp%n_matrices;
            const int iGlobalMatrix = first_matrix + iMatrix;

            const ordinal_type row_length = m_A_row_ptr(iRow+1) - m_A_row_ptr(iRow);
            value_type sum = 0;

            for (int iEntry = 0; iEntry < row_length; ++iEntry) {
              sum += m_A_values(iGlobalMatrix, m_A_row_ptr(iRow)+iEntry) *
                     m_x(iGlobalMatrix, m_A_col_indices(m_A_row_ptr(iRow)+iEntry));
            }

            sum *= alpha[iGlobalMatrix];

            if (dobeta == 0) {
              m_y(iGlobalMatrix, iRow) = sum;
            } else {
              m_y(iGlobalMatrix, iRow) =
                  beta[iGlobalMatrix] * m_y(iGlobalMatrix, iRow) + sum;
            }
          });
    }
    if (implementation == 4) {
      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices = last_matrix - first_matrix;
      const ordinal_type n_rows = m_A_row_ptr.extent(0)-1;

      //using MDPolicyType_2D = typename Kokkos::Experimental::MDRangePolicy<
      //    Kokkos::Experimental::Rank<2> >;
      //MDPolicyType_2D policyInit({0,0}, {n_rows, n_matrices});

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(dev, 0, n_rows * n_matrices),
          [&](const ordinal_type& iTemp) {
            const ordinal_type& iRow = iTemp%n_rows;
            const ordinal_type& iMatrix = iTemp/n_rows;
            const int iGlobalMatrix = first_matrix + iMatrix;

            const ordinal_type row_length = m_A_row_ptr(iRow+1) - m_A_row_ptr(iRow);
            value_type sum = 0;

            for (int iEntry = 0; iEntry < row_length; ++iEntry) {
              sum += m_A_values(iGlobalMatrix, m_A_row_ptr(iRow)+iEntry) *
                     m_x(iGlobalMatrix, m_A_col_indices(m_A_row_ptr(iRow)+iEntry));
            }

            sum *= alpha[iGlobalMatrix];

            if (dobeta == 0) {
              m_y(iGlobalMatrix, iRow) = sum;
            } else {
              m_y(iGlobalMatrix, iRow) =
                  beta[iGlobalMatrix] * m_y(iGlobalMatrix, iRow) + sum;
            }
          });
    }
    if (implementation == 5) {
      using ScratchPadIntView =
        Kokkos::View<int *, Kokkos::DefaultExecutionSpace::scratch_memory_space>;
      
      const ordinal_type n_rows = m_A_row_ptr.extent(0)-1;
      const ordinal_type nnz = m_A_col_indices.extent(0);

      ScratchPadIntView cols(dev.team_scratch(1), nnz);
      ScratchPadIntView row_map(dev.team_scratch(0), n_rows+1);
 
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(dev, 0, n_rows+1),
        [&](const ordinal_type& i) {
          row_map(i) = m_A_row_ptr(i);
        });

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(dev, 0, nnz),
        [&](const ordinal_type& i) {
          cols(i) = m_A_col_indices(i);
        });

      dev.team_barrier();

      for (int i_matrix =
               static_cast<int>(dev.league_rank()) * matrices_per_team;
           i_matrix <
           min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
           ++i_matrix) {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(dev, 0, n_rows),
            [&](const ordinal_type& iRow) {
              const ordinal_type row_length = m_A_row_ptr(iRow+1) - m_A_row_ptr(iRow);
              value_type sum = 0;

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(dev, row_length),
                  [&](const ordinal_type& iEntry, value_type& lsum) {
                    const value_type val = m_A_values(i_matrix, row_map(iRow)+iEntry);
                    lsum += val * m_x(i_matrix, cols(row_map(iRow) + iEntry));
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
    if (implementation == 6) {
      using ScratchPadIntView =
        Kokkos::View<int *, Kokkos::DefaultExecutionSpace::scratch_memory_space>;
      
      const ordinal_type n_rows = m_A_row_ptr.extent(0)-1;
      const ordinal_type nnz = m_A_col_indices.extent(0);

      ScratchPadIntView cols(dev.team_scratch(1), nnz);
      ScratchPadIntView row_map(dev.team_scratch(0), n_rows+1);

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(dev, 0, n_rows+1),
        [&](const ordinal_type& i) {
          row_map(i) = m_A_row_ptr(i);
        });

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(dev, 0, nnz),
        [&](const ordinal_type& i) {
          cols(i) = m_A_col_indices(i);
        });

      dev.team_barrier();

      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices = last_matrix - first_matrix;

      Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(dev, n_matrices),
          [&](const ordinal_type& iMatrix) {
            const int iGlobalMatrix = first_matrix + iMatrix;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(dev, 0, n_rows),
                [&](const ordinal_type& iRow) {
                  const ordinal_type row_length = m_A_row_ptr(iRow+1) - m_A_row_ptr(iRow);
                  value_type sum = 0;

                  for (int iEntry = 0; iEntry < row_length; ++iEntry) {
                    sum += m_A_values(iGlobalMatrix, row_map(iRow)+iEntry) *
                           m_x(iGlobalMatrix, cols(row_map(iRow) + iEntry));
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
    if (implementation == 7) {
      using ScratchPadIntView =
        Kokkos::View<int *, Kokkos::DefaultExecutionSpace::scratch_memory_space>;
      
      const ordinal_type n_rows = m_A_row_ptr.extent(0)-1;
      const ordinal_type nnz = m_A_col_indices.extent(0);

      ScratchPadIntView cols(dev.team_scratch(1), nnz);
      ScratchPadIntView row_map(dev.team_scratch(0), n_rows+1);

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(dev, 0, n_rows+1),
        [&](const ordinal_type& i) {
          row_map(i) = m_A_row_ptr(i);
        });

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(dev, 0, nnz),
        [&](const ordinal_type& i) {
          cols(i) = m_A_col_indices(i);
        });

      dev.team_barrier();

      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices = last_matrix - first_matrix;

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(dev, 0, n_rows),
          [&](const ordinal_type& iRow) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(dev, n_matrices),
                [&](const ordinal_type& iMatrix) {
                  const int iGlobalMatrix = first_matrix + iMatrix;

                  const ordinal_type row_length = m_A_row_ptr(iRow+1) - m_A_row_ptr(iRow);
                  value_type sum = 0;

                  for (int iEntry = 0; iEntry < row_length; ++iEntry) {
                    sum += m_A_values(iGlobalMatrix, row_map(iRow)+iEntry) *
                           m_x(iGlobalMatrix, cols(row_map(iRow) + iEntry));
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
    if (implementation == 8) {
      using ScratchPadIntView =
        Kokkos::View<int *, Kokkos::DefaultExecutionSpace::scratch_memory_space>;
      
      const ordinal_type n_rows = m_A_row_ptr.extent(0)-1;
      const ordinal_type nnz = m_A_col_indices.extent(0);

      ScratchPadIntView cols(dev.team_scratch(1), nnz);
      ScratchPadIntView row_map(dev.team_scratch(0), n_rows+1);

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(dev, 0, n_rows+1),
        [&](const ordinal_type& i) {
          row_map(i) = m_A_row_ptr(i);
        });

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(dev, 0, nnz),
        [&](const ordinal_type& i) {
          cols(i) = m_A_col_indices(i);
        });

      dev.team_barrier();

      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices = last_matrix - first_matrix;

      //using MDPolicyType_2D = typename Kokkos::Experimental::MDRangePolicy<
      //    Kokkos::Experimental::Rank<2> >;
      //MDPolicyType_2D policyInit({0,0}, {n_rows, n_matrices});

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(dev, 0, n_rows * n_matrices),
          [&](const ordinal_type& iTemp) {
            const ordinal_type& iRow = iTemp/n_matrices;
            const ordinal_type& iMatrix = iTemp%n_matrices;
            const int iGlobalMatrix = first_matrix + iMatrix;

            const ordinal_type row_length = m_A_row_ptr(iRow+1) - m_A_row_ptr(iRow);
            value_type sum = 0;

            for (int iEntry = 0; iEntry < row_length; ++iEntry) {
              sum += m_A_values(iGlobalMatrix, row_map(iRow)+iEntry) *
                     m_x(iGlobalMatrix, cols(row_map(iRow) + iEntry));
            }

            sum *= alpha[iGlobalMatrix];

            if (dobeta == 0) {
              m_y(iGlobalMatrix, iRow) = sum;
            } else {
              m_y(iGlobalMatrix, iRow) =
                  beta[iGlobalMatrix] * m_y(iGlobalMatrix, iRow) + sum;
            }
          });
    }
    if (implementation == 9) {
      using ScratchPadIntView =
        Kokkos::View<int *, Kokkos::DefaultExecutionSpace::scratch_memory_space>;
      
      const ordinal_type n_rows = m_A_row_ptr.extent(0)-1;
      const ordinal_type nnz = m_A_col_indices.extent(0);

      ScratchPadIntView cols(dev.team_scratch(1), nnz);
      ScratchPadIntView row_map(dev.team_scratch(0), n_rows+1);

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(dev, 0, n_rows+1),
        [&](const ordinal_type& i) {
          row_map(i) = m_A_row_ptr(i);
        });

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(dev, 0, nnz),
        [&](const ordinal_type& i) {
          cols(i) = m_A_col_indices(i);
        });

      dev.team_barrier();

      const int first_matrix =
          static_cast<int>(dev.league_rank()) * matrices_per_team;
      const int last_matrix =
          min(static_cast<int>(dev.league_rank() + 1) * matrices_per_team, N);
      const int n_matrices = last_matrix - first_matrix;

      //using MDPolicyType_2D = typename Kokkos::Experimental::MDRangePolicy<
      //    Kokkos::Experimental::Rank<2> >;
      //MDPolicyType_2D policyInit({0,0}, {n_rows, n_matrices});

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(dev, 0, n_rows * n_matrices),
          [&](const ordinal_type& iTemp) {
            const ordinal_type& iRow = iTemp%n_rows;
            const ordinal_type& iMatrix = iTemp/n_rows;
            const int iGlobalMatrix = first_matrix + iMatrix;

            const ordinal_type row_length = m_A_row_ptr(iRow+1) - m_A_row_ptr(iRow);
            value_type sum = 0;

            for (int iEntry = 0; iEntry < row_length; ++iEntry) {
              sum += m_A_values(iGlobalMatrix, row_map(iRow)+iEntry) *
                     m_x(iGlobalMatrix, cols(row_map(iRow) + iEntry));
            }

            sum *= alpha[iGlobalMatrix];

            if (dobeta == 0) {
              m_y(iGlobalMatrix, iRow) = sum;
            } else {
              m_y(iGlobalMatrix, iRow) =
                  beta[iGlobalMatrix] * m_y(iGlobalMatrix, iRow) + sum;
            }
          });
    }
  }
};
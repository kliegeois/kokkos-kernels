#define KOKKOSBATCHED_PROFILE 1
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSBATCHED_PROFILE)
#include "cuda_profiler_api.h"
#endif

#define KOKKOSBATCHED_USE_128BIT_MEMORY_INST

typedef Kokkos::DefaultExecutionSpace exec_space;
typedef typename exec_space::memory_space memory_space;
typedef Kokkos::DefaultHostExecutionSpace host_space;
typedef typename Kokkos::Device<exec_space, memory_space> device;

typedef double value_type;
typedef int local_ordinal_type;

/// 128*128*128/16*5 * (2*8) / 16
///
/// simd typedefs
///
using namespace KokkosBatched;

static constexpr int vector_length =
    DefaultVectorLength<value_type, memory_space>::value;
#if defined(KOKKOSBATCHED_USE_128BIT_MEMORY_INST)
static constexpr int internal_vector_length =
    DefaultInternalVectorLength<value_type, memory_space>::value;
#else
static constexpr int internal_vector_length = 1;
#endif

typedef Vector<SIMD<value_type>, vector_length> vector_type;
#if defined(KOKKOSBATCHED_USE_128BIT_MEMORY_INST)
typedef Vector<SIMD<value_type>, internal_vector_length> internal_vector_type;
#else
typedef value_type internal_vector_type;
#endif

typedef KokkosSparse::CrsMatrix<value_type, local_ordinal_type, device, void,
                                local_ordinal_type>
    matrix_type;
typedef KokkosSparse::CrsMatrix<vector_type, local_ordinal_type, device, void,
                                local_ordinal_type>
    vector_matrix_type;
typedef typename matrix_type::staticcrsgraph_type graph_type;

template <typename ActiveMemorySpace>
struct FactorizeModeAndAlgo;

template <>
struct FactorizeModeAndAlgo<Kokkos::HostSpace> {
  typedef Mode::Serial mode_type;
  typedef Algo::Level3::Blocked algo_type;
};

#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct FactorizeModeAndAlgo<Kokkos::CudaSpace> {
  typedef Mode::Team mode_type;
  typedef Algo::Level3::Unblocked algo_type;
};
#endif

template <typename ActiveMemorySpace>
struct SolveModeAndAlgo;

template <>
struct SolveModeAndAlgo<Kokkos::HostSpace> {
  typedef Mode::Serial mode_type;
  typedef Algo::Level2::Blocked algo_type;
};

#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct SolveModeAndAlgo<Kokkos::CudaSpace> {
  typedef Mode::Team mode_type;
  typedef Algo::Level2::Unblocked algo_type;
};
#endif

typedef Kokkos::LayoutRight LR;
typedef Kokkos::LayoutLeft LL;

template <typename ScalarType, typename OrdinalType, class Layout>
void SPDSparseRandomMatrices(
    OrdinalType nrows, OrdinalType nnz, OrdinalType N,
    typename graph_type::row_map_type::non_const_type &rowOffsets,
    typename graph_type::entries_type::non_const_type &colIndices,
    Kokkos::View<ScalarType **, Layout> &values) {
  OrdinalType nnz_d     = nrows;
  OrdinalType nnz_off_d = nnz - nnz_d;

  OrdinalType nnz_lower_trig = floor(nnz_off_d * 0.5);
  nnz                        = 2 * nnz_lower_trig + nnz_d;

  using row_map_type = typename graph_type::row_map_type;
  using entries_type = typename graph_type::entries_type;

  typename entries_type::non_const_type row_ind_lower_trig(
      "row indices", nnz_lower_trig + nnz_d);
  typename entries_type::non_const_type col_ind_lower_trig(
      "column indices", nnz_lower_trig + nnz_d);
  Kokkos::View<ScalarType **, Layout> value_lower_trig("values", N,
                                                       nnz_lower_trig + nnz_d);

  srand(1);

  {
    typename entries_type::HostMirror row_ind_lower_trig_h =
        Kokkos::create_mirror_view(row_ind_lower_trig);
    typename entries_type::HostMirror col_ind_lower_trig_h =
        Kokkos::create_mirror_view(col_ind_lower_trig);
    typename Kokkos::View<ScalarType **, Layout>::HostMirror
        value_lower_trig_h = Kokkos::create_mirror_view(value_lower_trig);

    int current_nnz_lower_trig = 0;
    while (current_nnz_lower_trig < nnz_lower_trig) {
      int r1 = rand() % nrows;
      int r2 = rand() % (nrows - 1);

      if (r2 >= r1) ++r2;

      int i1 = std::max(r1, r2);
      int i2 = std::min(r1, r2);

      bool already_generated = false;

      for (int i = 0; i < current_nnz_lower_trig; ++i) {
        if (i1 == row_ind_lower_trig_h(i) && i2 == col_ind_lower_trig_h(i)) {
          already_generated = true;
          break;
        }
      }

      if (already_generated) continue;

      row_ind_lower_trig_h(current_nnz_lower_trig) = i1;
      col_ind_lower_trig_h(current_nnz_lower_trig) = i2;

      for (int i_matrix = 0; i_matrix < N; ++i_matrix)
        value_lower_trig_h(i_matrix, current_nnz_lower_trig) =
            100.0 * rand() / INT_MAX - 50.0;

      ++current_nnz_lower_trig;
    }

    for (int i = 0; i < nnz_d; ++i) {
      row_ind_lower_trig_h(nnz_lower_trig + i) = i;
      col_ind_lower_trig_h(nnz_lower_trig + i) = i;

      for (int i_matrix = 0; i_matrix < N; ++i_matrix)
        value_lower_trig_h(i_matrix, nnz_lower_trig + i) =
            100.0 * rand() / INT_MAX - 50.0;
    }

    Kokkos::deep_copy(row_ind_lower_trig, row_ind_lower_trig_h);
    Kokkos::deep_copy(col_ind_lower_trig, col_ind_lower_trig_h);
    Kokkos::deep_copy(value_lower_trig, value_lower_trig_h);
  }

  using Kokkos::pair;

  Kokkos::UnorderedMap<pair<OrdinalType, OrdinalType>, bool> nodenode(
      nnz_lower_trig + nnz_d);

  typename row_map_type::non_const_type rowCounts("row counts", nrows);

  Kokkos::resize(rowOffsets, nrows + 1);

  Kokkos::parallel_for(
      nnz_lower_trig + nnz_d, KOKKOS_LAMBDA(const OrdinalType ielem) {
        const pair<OrdinalType, OrdinalType> key(row_ind_lower_trig(ielem),
                                                 col_ind_lower_trig(ielem));
        auto result = nodenode.insert(key);

        Kokkos::atomic_fetch_add(&rowCounts(row_ind_lower_trig(ielem)), 1);
        if (row_ind_lower_trig(ielem) != col_ind_lower_trig(ielem))
          Kokkos::atomic_fetch_add(&rowCounts(col_ind_lower_trig(ielem)), 1);
      });

  // Parallel prefix-sum row counts and allocate column index array
  Kokkos::parallel_scan(
      nrows, KOKKOS_LAMBDA(int irow, int &update, bool final) {
        // parallel scan is a multi-pass parallel pattern
        // In the ‘final’ pass ‘update’ has the prefix value
        if (final) rowOffsets(irow) = update;
        update += rowCounts(irow);
        if (final && nrows == irow + 1)
          rowOffsets(irow + 1) = update;  // total non-zeros
      });

  Kokkos::deep_copy(rowCounts, static_cast<OrdinalType>(0));
  Kokkos::resize(colIndices, nnz);

  // Fill column index array with rows in non-deterministic order
  Kokkos::parallel_for(
      nodenode.capacity(), KOKKOS_LAMBDA(int ientry) {
        if (nodenode.valid_at(ientry)) {
          const pair<OrdinalType, OrdinalType> key = nodenode.key_at(ientry);
          const OrdinalType row                    = key.first;
          const OrdinalType col                    = key.second;

          const bool isDiffIndices = (row != col);

          const size_t count = Kokkos::atomic_fetch_add(&rowCounts(row), 1);
          colIndices(rowOffsets(row) + count) = col;
          if (isDiffIndices) {
            const size_t count = Kokkos::atomic_fetch_add(&rowCounts(col), 1);
            colIndices(rowOffsets(col) + count) = row;
          }
        }
      });

  {
    typename row_map_type::HostMirror row_map_h =
        Kokkos::create_mirror_view(rowOffsets);
    Kokkos::deep_copy(row_map_h, rowOffsets);
    // Sort eacch row of column index array
    for (int irow = 0; irow < 1; ++irow)
      Kokkos::sort(colIndices, row_map_h(irow), row_map_h(irow + 1));
  }

  Kokkos::resize(values, N, nnz);

  // Fill the view with the random values
  Kokkos::parallel_for(
      nrows, KOKKOS_LAMBDA(int irow) {
        for (OrdinalType jcol = rowOffsets(irow); jcol < rowOffsets(irow + 1);
             ++jcol) {
          OrdinalType row = irow;
          OrdinalType col = colIndices(jcol);

          if (row > col)
            for (OrdinalType ielem = 0; ielem < nnz_lower_trig; ++ielem) {
              if (row == row_ind_lower_trig(ielem) &&
                  col == col_ind_lower_trig(ielem)) {
                for (int i_matrix = 0; i_matrix < N; ++i_matrix)
                  values(i_matrix, jcol) = value_lower_trig(i_matrix, ielem);
                break;
              }
            }
          else if (row < col)
            for (OrdinalType ielem = 0; ielem < nnz_lower_trig; ++ielem) {
              if (col == row_ind_lower_trig(ielem) &&
                  row == col_ind_lower_trig(ielem)) {
                for (int i_matrix = 0; i_matrix < N; ++i_matrix)
                  values(i_matrix, jcol) = value_lower_trig(i_matrix, ielem);
                break;
              }
            }
          else
            for (int i_matrix = 0; i_matrix < N; ++i_matrix)
              values(i_matrix, jcol) =
                  value_lower_trig(i_matrix, nnz_lower_trig + irow);
        }
      });

  using Kokkos::Experimental::fabs;

  // Make the diagonal dominant (and therefore SPD using the Gershgorin circle
  // theorem)
  Kokkos::parallel_for(
      nrows, KOKKOS_LAMBDA(int irow) {
        for (OrdinalType jcol = rowOffsets(irow); jcol < rowOffsets(irow + 1);
             ++jcol) {
          OrdinalType row = irow;
          OrdinalType col = colIndices(jcol);

          if (row == col) {
            for (int i_matrix = 0; i_matrix < N; ++i_matrix)
              values(i_matrix, jcol) = fabs(values(i_matrix, jcol));
            for (OrdinalType jjcol = rowOffsets(irow);
                 jjcol < rowOffsets(irow + 1); ++jjcol) {
              if (jjcol != jcol)
                for (int i_matrix = 0; i_matrix < N; ++i_matrix)
                  values(i_matrix, col) += fabs(values(i_matrix, jjcol));
            }
          }
        }
      });
  /*
  {
    std::ofstream myfile;
    myfile.open("matrices.txt");
    typename row_map_type::HostMirror row_map_h =
        Kokkos::create_mirror_view(rowOffsets);
    typename entries_type::HostMirror entries_h =
        Kokkos::create_mirror_view(colIndices);
    typename Kokkos::View<ScalarType **, Layout>::HostMirror values_h =
        Kokkos::create_mirror_view(values);
    Kokkos::deep_copy(row_map_h, rowOffsets);
    Kokkos::deep_copy(entries_h, colIndices);
    Kokkos::deep_copy(values_h, values);
    for (OrdinalType i = 0; i < nrows; ++i) {
      for (OrdinalType j = row_map_h(i); j < row_map_h(i + 1); ++j) {
        myfile << i << " " << entries_h(j) << " ";
        for (int i_matrix = 0; i_matrix < N; ++i_matrix) {
          myfile << values_h(i_matrix, j) << " ";
        }
        myfile << std::endl;
      }
    }
    myfile.close();
  }
  */
}

template <typename OrdinalType>
OrdinalType getNNZ(OrdinalType nrows, OrdinalType max_offset,
                   OrdinalType offset) {
  OrdinalType nnz = 0;
  for (int current_row = 0; current_row < nrows; ++current_row) {
    for (int i_offset = 1; i_offset <= max_offset; ++i_offset) {
      int current_col = current_row - i_offset * offset;
      if (current_col >= 0) ++nnz;
    }
  }

  nnz *= 2;
  nnz += nrows;

  return nnz;
}

template <typename ScalarType, typename OrdinalType, class Layout>
void SPDSparseMatrices(
    OrdinalType nrows, OrdinalType max_offset, OrdinalType offset,
    OrdinalType N,
    typename graph_type::row_map_type::non_const_type &rowOffsets,
    typename graph_type::entries_type::non_const_type &colIndices,
    Kokkos::View<ScalarType **, Layout> &values) {
  OrdinalType nnz_d = nrows;
  OrdinalType nnz   = getNNZ(nrows, max_offset, offset);

  OrdinalType nnz_lower_trig = (nnz - nnz_d) * 0.5;

  using row_map_type = typename graph_type::row_map_type;
  using entries_type = typename graph_type::entries_type;

  typename entries_type::non_const_type row_ind_lower_trig(
      "row indices", nnz_lower_trig + nnz_d);
  typename entries_type::non_const_type col_ind_lower_trig(
      "column indices", nnz_lower_trig + nnz_d);
  Kokkos::View<ScalarType **, Layout> value_lower_trig("values", N,
                                                       nnz_lower_trig + nnz_d);

  srand(1);

  {
    typename entries_type::HostMirror row_ind_lower_trig_h =
        Kokkos::create_mirror_view(row_ind_lower_trig);
    typename entries_type::HostMirror col_ind_lower_trig_h =
        Kokkos::create_mirror_view(col_ind_lower_trig);
    typename Kokkos::View<ScalarType **, Layout>::HostMirror
        value_lower_trig_h = Kokkos::create_mirror_view(value_lower_trig);

    int current_nnz_lower_trig = 0;
    for (int current_row = 0; current_row < nrows; ++current_row) {
      for (int i_offset = 1; i_offset <= max_offset; ++i_offset) {
        int current_col = current_row - i_offset * offset;
        if (current_col >= 0) {
          row_ind_lower_trig_h(current_nnz_lower_trig) = current_row;
          col_ind_lower_trig_h(current_nnz_lower_trig) = current_col;

          for (int i_matrix = 0; i_matrix < N; ++i_matrix)
            value_lower_trig_h(i_matrix, current_nnz_lower_trig) =
                100.0 * rand() / INT_MAX - 50.0;

          ++current_nnz_lower_trig;
        }
      }
    }

    for (int i = 0; i < nnz_d; ++i) {
      row_ind_lower_trig_h(nnz_lower_trig + i) = i;
      col_ind_lower_trig_h(nnz_lower_trig + i) = i;

      for (int i_matrix = 0; i_matrix < N; ++i_matrix)
        value_lower_trig_h(i_matrix, nnz_lower_trig + i) =
            100.0 * rand() / INT_MAX - 50.0;
    }

    Kokkos::deep_copy(row_ind_lower_trig, row_ind_lower_trig_h);
    Kokkos::deep_copy(col_ind_lower_trig, col_ind_lower_trig_h);
    Kokkos::deep_copy(value_lower_trig, value_lower_trig_h);
  }

  using Kokkos::pair;

  Kokkos::UnorderedMap<pair<OrdinalType, OrdinalType>, bool> nodenode(
      nnz_lower_trig + nnz_d);

  typename row_map_type::non_const_type rowCounts("row counts", nrows);

  Kokkos::resize(rowOffsets, nrows + 1);

  Kokkos::parallel_for(
      nnz_lower_trig + nnz_d, KOKKOS_LAMBDA(const OrdinalType ielem) {
        const pair<OrdinalType, OrdinalType> key(row_ind_lower_trig(ielem),
                                                 col_ind_lower_trig(ielem));
        auto result = nodenode.insert(key);

        Kokkos::atomic_fetch_add(&rowCounts(row_ind_lower_trig(ielem)), 1);
        if (row_ind_lower_trig(ielem) != col_ind_lower_trig(ielem))
          Kokkos::atomic_fetch_add(&rowCounts(col_ind_lower_trig(ielem)), 1);
      });

  // Parallel prefix-sum row counts and allocate column index array
  Kokkos::parallel_scan(
      nrows, KOKKOS_LAMBDA(int irow, int &update, bool final) {
        // parallel scan is a multi-pass parallel pattern
        // In the ‘final’ pass ‘update’ has the prefix value
        if (final) rowOffsets(irow) = update;
        update += rowCounts(irow);
        if (final && nrows == irow + 1)
          rowOffsets(irow + 1) = update;  // total non-zeros
      });

  Kokkos::deep_copy(rowCounts, static_cast<OrdinalType>(0));
  Kokkos::resize(colIndices, nnz);

  // Fill column index array with rows in non-deterministic order
  Kokkos::parallel_for(
      nodenode.capacity(), KOKKOS_LAMBDA(int ientry) {
        if (nodenode.valid_at(ientry)) {
          const pair<OrdinalType, OrdinalType> key = nodenode.key_at(ientry);
          const OrdinalType row                    = key.first;
          const OrdinalType col                    = key.second;

          const bool isDiffIndices = (row != col);

          const size_t count = Kokkos::atomic_fetch_add(&rowCounts(row), 1);
          colIndices(rowOffsets(row) + count) = col;
          if (isDiffIndices) {
            const size_t count = Kokkos::atomic_fetch_add(&rowCounts(col), 1);
            colIndices(rowOffsets(col) + count) = row;
          }
        }
      });

  {
    typename row_map_type::HostMirror row_map_h =
        Kokkos::create_mirror_view(rowOffsets);
    Kokkos::deep_copy(row_map_h, rowOffsets);
    // Sort eacch row of column index array
    for (int irow = 0; irow < 1; ++irow)
      Kokkos::sort(colIndices, row_map_h(irow), row_map_h(irow + 1));
  }

  Kokkos::resize(values, N, nnz);

  // Fill the view with the random values
  Kokkos::parallel_for(
      nrows, KOKKOS_LAMBDA(int irow) {
        for (OrdinalType jcol = rowOffsets(irow); jcol < rowOffsets(irow + 1);
             ++jcol) {
          OrdinalType row = irow;
          OrdinalType col = colIndices(jcol);

          if (row > col)
            for (OrdinalType ielem = 0; ielem < nnz_lower_trig; ++ielem) {
              if (row == row_ind_lower_trig(ielem) &&
                  col == col_ind_lower_trig(ielem)) {
                for (int i_matrix = 0; i_matrix < N; ++i_matrix)
                  values(i_matrix, jcol) = value_lower_trig(i_matrix, ielem);
                break;
              }
            }
          else if (row < col)
            for (OrdinalType ielem = 0; ielem < nnz_lower_trig; ++ielem) {
              if (col == row_ind_lower_trig(ielem) &&
                  row == col_ind_lower_trig(ielem)) {
                for (int i_matrix = 0; i_matrix < N; ++i_matrix)
                  values(i_matrix, jcol) = value_lower_trig(i_matrix, ielem);
                break;
              }
            }
          else
            for (int i_matrix = 0; i_matrix < N; ++i_matrix)
              values(i_matrix, jcol) =
                  value_lower_trig(i_matrix, nnz_lower_trig + irow);
        }
      });

  using Kokkos::Experimental::fabs;

  // Make the diagonal dominant (and therefore SPD using the Gershgorin circle
  // theorem)
  Kokkos::parallel_for(
      nrows, KOKKOS_LAMBDA(int irow) {
        for (OrdinalType jcol = rowOffsets(irow); jcol < rowOffsets(irow + 1);
             ++jcol) {
          OrdinalType row = irow;
          OrdinalType col = colIndices(jcol);

          if (row == col) {
            for (int i_matrix = 0; i_matrix < N; ++i_matrix)
              values(i_matrix, jcol) = fabs(values(i_matrix, jcol));
            for (OrdinalType jjcol = rowOffsets(irow);
                 jjcol < rowOffsets(irow + 1); ++jjcol) {
              if (jjcol != jcol)
                for (int i_matrix = 0; i_matrix < N; ++i_matrix)
                  values(i_matrix, col) += fabs(values(i_matrix, jjcol));
            }
          }
        }
      });
  /*
  {
    std::ofstream myfile;
    myfile.open("matrices.txt");
    typename row_map_type::HostMirror row_map_h =
        Kokkos::create_mirror_view(rowOffsets);
    typename entries_type::HostMirror entries_h =
        Kokkos::create_mirror_view(colIndices);
    typename Kokkos::View<ScalarType **, Layout>::HostMirror values_h =
        Kokkos::create_mirror_view(values);
    Kokkos::deep_copy(row_map_h, rowOffsets);
    Kokkos::deep_copy(entries_h, colIndices);
    Kokkos::deep_copy(values_h, values);
    for (OrdinalType i = 0; i < nrows; ++i) {
      for (OrdinalType j = row_map_h(i); j < row_map_h(i + 1); ++j) {
        myfile << i << " " << entries_h(j) << " ";
        for (int i_matrix = 0; i_matrix < N; ++i_matrix) {
          myfile << values_h(i_matrix, j) << " ";
        }
        myfile << std::endl;
      }
    }
    myfile.close();
  }
  */
}

template <class XType>
void writeArrayTofile(const XType x, std::string name) {
  std::ofstream myfile;
  myfile.open(name);

  typename XType::HostMirror x_h = Kokkos::create_mirror_view(x);

  Kokkos::deep_copy(x_h, x);

  for (int i = 0; i < x_h.extent(0); ++i) {
    for (int j = 0; j < x_h.extent(1); ++j) {
      myfile << x_h(i, j) << " ";
    }
    myfile << std::endl;
  }

  myfile.close();
}

template <class execution_space>
int launch_parameters(int numRows, int nnz, int rows_per_thread, int &team_size,
                      int vector_length) {
  int rows_per_team;
  int nnz_per_row = nnz / numRows;
  if (nnz_per_row < 1) nnz_per_row = 1;

  // Determine rows per thread
  if (rows_per_thread < 1) {
#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same<Kokkos::Cuda, execution_space>::value)
      rows_per_thread = 1;
    else
#endif
    {
      if (nnz_per_row < 20 && nnz > 5000000) {
        rows_per_thread = 256;
      } else
        rows_per_thread = 64;
    }
  }

#ifdef KOKKOS_ENABLE_CUDA
  if (team_size < 1) team_size = 256 / vector_length;
#endif

  rows_per_team = rows_per_thread * team_size;

  if (rows_per_team < 0) {
    int nnz_per_team = 4096;
    int conc         = execution_space::concurrency();
    while ((conc * nnz_per_team * 4 > nnz) && (nnz_per_team > 256))
      nnz_per_team /= 2;
    int tmp_nnz_per_row = nnz / numRows;
    rows_per_team = (nnz_per_team + tmp_nnz_per_row - 1) / tmp_nnz_per_row;
  }

  return rows_per_team;
}

template <class AMatrix, class AVMatrix, class AView, class IntView,
          class XYVector, class XYVVector>
void getSPMVInputs(AMatrix *myMatrices, AVMatrix *myVectorMatrices,
                   AView values, IntView rowOffsets, IntView colIndices,
                   XYVector x, XYVector y, XYVVector xv, XYVVector yv,
                   const typename IntView::non_const_value_type Blk,
                   const typename IntView::non_const_value_type nnz,
                   const bool random,
                   const typename IntView::non_const_value_type max_offset,
                   const typename IntView::non_const_value_type offset,
                   const typename IntView::non_const_value_type N,
                   typename AMatrix::non_const_value_type *s_a,
                   typename AMatrix::non_const_value_type *s_b,
                   typename AVMatrix::non_const_value_type *s_av,
                   typename AVMatrix::non_const_value_type *s_bv) {
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename IntView::non_const_value_type int_type;
  typedef typename AVMatrix::non_const_value_type vector_type;
  typedef typename AView::array_layout layout;

  if (random)
    SPDSparseRandomMatrices<value_type, local_ordinal_type>(
        Blk, nnz, N, rowOffsets, colIndices, values);
  else
    SPDSparseMatrices<value_type, local_ordinal_type>(
        Blk, max_offset, offset, N, rowOffsets, colIndices, values);

  Kokkos::View<vector_type **, layout> vector_values("values",
                                                     N / vector_length, nnz);
  Kokkos::View<value_type **[vector_length], layout> vector_values_data(
      (value_type *)vector_values.data(), N / vector_length, nnz);

  Kokkos::parallel_for(
      N / vector_length, KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < nnz; ++j)
          for (int k = 0; k < vector_length; ++k)
            vector_values_data(i, j, k) = values(i * vector_length + k, j);
      });

  graph_type myGraph(colIndices, rowOffsets);

  if (std::is_same<layout, Kokkos::LayoutRight>::value) {
    for (int i_matrix = 0; i_matrix < N; ++i_matrix)
      myMatrices[i_matrix] =
          AMatrix("test matrix", Blk, subview(values, i_matrix, Kokkos::ALL()),
                  myGraph);

    for (int i_matrix = 0; i_matrix < N / vector_length; ++i_matrix)
      myVectorMatrices[i_matrix] =
          AVMatrix("test matrix", Blk,
                   subview(vector_values, i_matrix, Kokkos::ALL()), myGraph);
  } else
    std::cout << "Crs Matrices are not created when using the left layout."
              << std::endl;

  std::fill_n(s_a, N, 1.0);
  std::fill_n(s_b, N, 0.0);
  std::fill_n(s_av, N / vector_length, 1.0);
  std::fill_n(s_bv, N / vector_length, 0.0);

  Kokkos::View<value_type **[vector_length], layout> xv_data(
      (value_type *)xv.data(), N / vector_length, Blk);
  Kokkos::View<value_type **[vector_length], layout> yv_data(
      (value_type *)yv.data(), N / vector_length, Blk);

  Kokkos::deep_copy(x, 1.);
  Kokkos::deep_copy(y, 0.);

  Kokkos::parallel_for(
      N / vector_length, KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < Blk; ++j)
          for (int k = 0; k < vector_length; ++k) {
            xv_data(i, j, k) = x(i * vector_length + k, j);
            yv_data(i, j, k) = y(i * vector_length + k, j);
          }
      });
}
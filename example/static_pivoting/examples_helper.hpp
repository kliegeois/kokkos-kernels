//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.4
//       Copyright (2021) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER

template <typename MemberType, class MatrixType1, class MatrixType2, class VectorType1, class VectorType2>
KOKKOS_INLINE_FUNCTION void TeamStaticPivoting(const MemberType &member, 
                                               const MatrixType1 A, 
                                               const MatrixType2 PDAD, 
                                               const VectorType1 Y, 
                                               const VectorType2 PDY, 
                                               const VectorType2 D2, 
                                               const VectorType2 tmp_v_1, 
                                               const VectorType2 tmp_v_2) {
  using value_type = typename MatrixType1::non_const_value_type;
  const int n = A.extent(0);

  for (int i = 0; i < n; ++i) {
    D2(i) = 0.;
    tmp_v_1(i) = 0;
    tmp_v_2(i) = 1.;
    for (int j = 0; j < n; ++j) {
      if (D2(i) < Kokkos::abs(A(j, i)))
        D2(i) = Kokkos::abs(A(j, i));
      if (tmp_v_1(i) < Kokkos::abs(A(i, j)))
        tmp_v_1(i) = Kokkos::abs(A(i, j));
    }
    D2(i) = 1./D2(i);
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A(i, j) *= D2(j);
    }
  }

  for (int i = 0; i < n; ++i) {
    value_type D1_i = 0.;
    for (int j = 0; j < n; ++j) {
      if (D1_i < Kokkos::abs(A(i, j)))
        D1_i = Kokkos::abs(A(i, j));
    }
    D1_i = 1./D1_i;
    for (int j = 0; j < n; ++j) {
      A(i, j) *= D1_i;
    }
    Y(i) *= D1_i;
  }

  for (int i = 0; i < n; ++i) {
    int row_index = 0;
    int col_index = 0;
    value_type tmp_0 = 0.;
    value_type tmp_1 = 0.;
    for (int j = 0; j < n; ++j) {
      if (tmp_0 < tmp_v_1(j)) {
        tmp_0 = tmp_v_1(j);
        row_index = j;
      }
    }
    for (int j = 0; j < n; ++j) {
      if (tmp_1 < Kokkos::abs(A(row_index, j) * tmp_v_2(j))) {
        tmp_1 = Kokkos::abs(A(row_index, j) * tmp_v_2(j));
        col_index = j;
      }
    }
    tmp_v_1(row_index) = 0.;
    tmp_v_2(col_index) = 0.;

    for (int j = 0; j < n; ++j) {
      PDAD(col_index, j) = A(row_index, j);
    }
    PDY(col_index) = Y(row_index);
  }
}

template <class MatrixType1, class MatrixType2, class VectorType1, class VectorType2>
KOKKOS_INLINE_FUNCTION void SerialStaticPivoting(const MatrixType1 A, 
                                                 const MatrixType2 PDAD, 
                                                 const VectorType1 Y, 
                                                 const VectorType2 PDY, 
                                                 const VectorType2 D2, 
                                                 const VectorType2 tmp_v_1, 
                                                 const VectorType2 tmp_v_2) {
  using value_type = typename MatrixType1::non_const_value_type;
  const int n = A.extent(0);

  for (int i = 0; i < n; ++i) {
    D2(i) = 0.;
    tmp_v_1(i) = 0;
    tmp_v_2(i) = 1.;
    for (int j = 0; j < n; ++j) {
      if (D2(i) < Kokkos::abs(A(j, i)))
        D2(i) = Kokkos::abs(A(j, i));
      if (tmp_v_1(i) < Kokkos::abs(A(i, j)))
        tmp_v_1(i) = Kokkos::abs(A(i, j));
    }
    D2(i) = 1./D2(i);
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A(i, j) *= D2(j);
    }
  }

  for (int i = 0; i < n; ++i) {
    value_type D1_i = 0.;
    for (int j = 0; j < n; ++j) {
      if (D1_i < Kokkos::abs(A(i, j)))
        D1_i = Kokkos::abs(A(i, j));
    }
    D1_i = 1./D1_i;
    for (int j = 0; j < n; ++j) {
      A(i, j) *= D1_i;
    }
    Y(i) *= D1_i;
  }

  for (int i = 0; i < n; ++i) {
    int row_index = 0;
    int col_index = 0;
    value_type tmp_0 = 0.;
    value_type tmp_1 = 0.;
    for (int j = 0; j < n; ++j) {
      if (tmp_0 < tmp_v_1(j)) {
        tmp_0 = tmp_v_1(j);
        row_index = j;
      }
    }
    for (int j = 0; j < n; ++j) {
      if (tmp_1 < Kokkos::abs(A(row_index, j) * tmp_v_2(j))) {
        tmp_1 = Kokkos::abs(A(row_index, j) * tmp_v_2(j));
        col_index = j;
      }
    }
    tmp_v_1(row_index) = 0.;
    tmp_v_2(col_index) = 0.;

    for (int j = 0; j < n; ++j) {
      PDAD(col_index, j) = A(row_index, j);
    }
    PDY(col_index) = Y(row_index);
  }
}

template <typename MemberType, class VectorType1, class VectorType2, class VectorType3>
KOKKOS_INLINE_FUNCTION void TeamScale(const MemberType &member,const VectorType1 X, const VectorType2 D, const VectorType3 DX) {
  const int n = X.extent(0);

  for (int i = 0; i < n; ++i) {
    DX(i) = D(i) * X(i);
  }
}

template <class VectorType1, class VectorType2, class VectorType3>
KOKKOS_INLINE_FUNCTION void SerialScale(const VectorType1 X, const VectorType2 D, const VectorType3 DX) {
  const int n = X.extent(0);

  for (int i = 0; i < n; ++i) {
    DX(i) = D(i) * X(i);
  }
}

template <typename MemberType, class MatrixType, class VectorType>
KOKKOS_INLINE_FUNCTION void TeamGESV(const MemberType &member,
                                     const MatrixType A,
                                     const VectorType X,
                                     const VectorType Y) {
  using ScratchPadMatrixViewType = Kokkos::View<
      typename MatrixType::non_const_value_type**,
      typename MatrixType::array_layout,
      typename MatrixType::execution_space::scratch_memory_space>;
  using ScratchPadVectorViewType = Kokkos::View<
      typename VectorType::non_const_value_type*,
      typename VectorType::array_layout,
      typename VectorType::execution_space::scratch_memory_space>;
  const int n = A.extent(0);

  ScratchPadMatrixViewType tmp(member.team_scratch(0), n, n + 4);
  auto PDAD = Kokkos::subview(tmp, Kokkos::ALL, Kokkos::make_pair(0, n));
  auto PDY = Kokkos::subview(tmp, Kokkos::ALL, n);
  auto D2 = Kokkos::subview(tmp, Kokkos::ALL, n+1);
  auto tmp_v_1 = Kokkos::subview(tmp, Kokkos::ALL, n+2);
  auto tmp_v_2 = Kokkos::subview(tmp, Kokkos::ALL, n+3);

  TeamStaticPivoting(member, A, PDAD, Y, PDY, D2, tmp_v_1, tmp_v_2);
  member.team_barrier();

  KokkosBatched::TeamLU<MemberType, KokkosBatched::Algo::Level3::Unblocked>::invoke(member, PDAD);
  member.team_barrier();

  KokkosBatched::TeamTrsm<MemberType, KokkosBatched::Side::Left, KokkosBatched::Uplo::Lower,
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Diag::Unit,
            KokkosBatched::Algo::Level3::Unblocked>::invoke(member, 1.0, PDAD, PDY);
  member.team_barrier();

  KokkosBatched::TeamTrsm<MemberType, KokkosBatched::Side::Left, KokkosBatched::Uplo::Upper,
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Diag::NonUnit,
            KokkosBatched::Algo::Level3::Unblocked>::invoke(member, 1.0, PDAD, PDY);
  member.team_barrier();

  TeamScale(member, PDY, D2, X);
  member.team_barrier();
}

template <class MatrixType, class VectorType>
KOKKOS_INLINE_FUNCTION void SerialGESV(const MatrixType A, const VectorType X, const VectorType Y, const MatrixType tmp) {
  const int n = A.extent(0);

  auto PDAD = Kokkos::subview(tmp, Kokkos::ALL, Kokkos::make_pair(0, n));
  auto PDY = Kokkos::subview(tmp, Kokkos::ALL, n);
  auto D2 = Kokkos::subview(tmp, Kokkos::ALL, n+1);
  auto tmp_v_1 = Kokkos::subview(tmp, Kokkos::ALL, n+2);
  auto tmp_v_2 = Kokkos::subview(tmp, Kokkos::ALL, n+3);

  SerialStaticPivoting(A, PDAD, Y, PDY, D2, tmp_v_1, tmp_v_2);

  KokkosBatched::SerialLU<KokkosBatched::Algo::Level3::Unblocked>::invoke(PDAD);

  KokkosBatched::SerialTrsm<KokkosBatched::Side::Left, KokkosBatched::Uplo::Lower,
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Diag::Unit,
            KokkosBatched::Algo::Level3::Unblocked>::invoke(1.0, PDAD, PDY);

  KokkosBatched::SerialTrsm<KokkosBatched::Side::Left, KokkosBatched::Uplo::Upper,
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Diag::NonUnit,
            KokkosBatched::Algo::Level3::Unblocked>::invoke(1.0, PDAD, PDY);

  SerialScale(PDY, D2, X);  
}

template <class XType>
void write2DArrayToMM(std::string name, const XType x) {
  std::ofstream myfile;
  myfile.open(name);

  auto x_h = Kokkos::create_mirror_view(x);

  Kokkos::deep_copy(x_h, x);

  if (XType::Rank == 2) {
    myfile << "%% MatrixMarket 2D Array\n%" << std::endl;
    myfile << x_h.extent(0) << " " << x_h.extent(1) << std::endl;

    for (size_t i = 0; i < x_h.extent(0); ++i) {
        for (size_t j = 0; j < x_h.extent(1); ++j) {
            myfile << std::setprecision (15) << x_h(i, j) << " ";
        }
        myfile << std::endl;
    }

    myfile.close();
  }
}

template <class XType>
void write3DArrayToMM(std::string name, const XType x) {
  std::ofstream myfile;
  myfile.open(name);

  auto x_h = Kokkos::create_mirror_view(x);

  Kokkos::deep_copy(x_h, x);

  if (XType::Rank == 3) {
    myfile << "%% MatrixMarket 3D Array\n%" << std::endl;
    myfile << x_h.extent(0) << " " << x_h.extent(1) << " " << x_h.extent(2) << std::endl;

    for (size_t i = 0; i < x_h.extent(0); ++i) {
        myfile << "Slice " << i << std::endl;
        for (size_t j = 0; j < x_h.extent(1); ++j) {
            for (size_t k = 0; k < x_h.extent(2); ++k) {
                myfile << std::setprecision (15) << x_h(i, j, k) << " ";
            }
            myfile << std::endl;
        }
    }

    myfile.close();
  }
}

void readSizesFromMM(std::string name, int &nrows, int &ncols, int &nnz,
                     int &N) {
  std::ifstream input(name);
  while (input.peek() == '%')
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::string line_sizes;

  getline(input, line_sizes);

  std::stringstream iss(line_sizes);

  int number;
  std::vector<int> sizes;
  while (iss >> number) sizes.push_back(number);

  nrows = sizes[0];
  ncols = sizes[1];

  nnz = 0;
  N   = 0;

  if (sizes.size() >= 3) nnz = sizes[2];

  if (sizes.size() == 4) N = sizes[3];
}

template <class XType>
void readArrayFromMM(std::string name, const XType &x) {
  std::ifstream input(name);

  while (input.peek() == '%')
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  typename XType::HostMirror x_h = Kokkos::create_mirror_view(x);

  for (size_t i = 0; i < x_h.extent(0); ++i)
    for (size_t j = 0; j < x_h.extent(1); ++j) input >> x_h(i, j);

  input.close();

  Kokkos::deep_copy(x, x_h);

/*
  std::ofstream myfile;
  myfile.open("x-data.txt");


  for (size_t i = 0; i < x_h.extent(0); ++i) {
    for (size_t j = 0; j < x_h.extent(1); ++j) {
      myfile << std::setprecision (15) << x_h(i, j) << " ";
    }
    myfile << std::endl;
  }

  myfile.close();
  */
}


template <typename IntView, typename VectorViewType>
void create_tridiagonal_batched_matrices(const int nnz, const int BlkSize,
                                         const int N, const IntView &r,
                                         const IntView &c,
                                         const VectorViewType &D,
                                         const VectorViewType &X,
                                         const VectorViewType &B) {
  Kokkos::Random_XorShift64_Pool<
      typename VectorViewType::device_type::execution_space>
      random(13718);
  Kokkos::fill_random(
      X, random,
      Kokkos::reduction_identity<typename VectorViewType::value_type>::prod());
  Kokkos::fill_random(
      B, random,
      Kokkos::reduction_identity<typename VectorViewType::value_type>::prod());

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
        D_host(l, i) = typename VectorViewType::value_type(2.0);
      }
      c_host(i) = current_col;
      ++current_col;
    } else {
      for (int l = 0; l < N; ++l) {
        D_host(l, i) = typename VectorViewType::value_type(-1.0);
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
}


template <typename MatrixViewType, typename VectorViewType>
void create_saddle_point_matrices(const MatrixViewType &A, const VectorViewType &Y, const int n_2 = 4) {
    Kokkos::Random_XorShift64_Pool<
        typename MatrixViewType::device_type::execution_space>
        random(13718);
    const int N = A.extent(0);
    const int n = A.extent(1);
    const int n_1 = n - n_2;

    const int n_dim = n_2 - 1;
    MatrixViewType xs("xs", N, n_1, n_dim);
    VectorViewType ys("ys", N, n_1);

    Kokkos::fill_random(
        xs, random,
        Kokkos::reduction_identity<typename MatrixViewType::value_type>::prod());
    Kokkos::fill_random(
        ys, random,
        Kokkos::reduction_identity<typename VectorViewType::value_type>::prod());

    auto xs_host = Kokkos::create_mirror_view(xs);
    auto ys_host = Kokkos::create_mirror_view(ys);
    auto A_host = Kokkos::create_mirror_view(A);
    auto Y_host = Kokkos::create_mirror_view(Y);

    Kokkos::deep_copy(xs_host, xs);
    Kokkos::deep_copy(ys_host, ys);

    for (int i = 0; i < n_1; ++i) {
        for (int j = 0; j < n_1; ++j) {
            auto xs_j = Kokkos::subview(xs_host, Kokkos::ALL, j, Kokkos::ALL);
            for (int l = 0; l < N; ++l) {
                auto xs_i = Kokkos::subview(xs_host, l, i, Kokkos::ALL);
                auto xs_j = Kokkos::subview(xs_host, l, j, Kokkos::ALL);
                typename MatrixViewType::value_type d = 0;
                for (int k = 0; k < n_dim; ++k)
                    d += Kokkos::pow(xs_i(k) - xs_j(k), 2);
                d = Kokkos::sqrt(d);
                A_host(l, i, j) = Kokkos::pow(d, 5);
            }
        }
        for (int l = 0; l < N; ++l) {
            A_host(l, i, n_1) = (typename MatrixViewType::value_type) 1.0;
            A_host(l, n_1, i) = (typename MatrixViewType::value_type) 1.0;
            for (int k = 0; k < n_dim; ++k) {
                A_host(l, i, n_1 + k + 1) = xs_host(l, i, k);
                A_host(l, n_1 + k + 1, i) = xs_host(l, i, k);
            }
            Y_host(l, i) = ys_host(l, i);
        }
    }
    for (int i = n_1; i < n; ++i) {
        for (int l = 0; l < N; ++l) {
            Y_host(l, i) = (typename MatrixViewType::value_type) 0.0;
        }
    }

    Kokkos::deep_copy(A, A_host);
    Kokkos::deep_copy(Y, Y_host);

    Kokkos::fence();
}
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

template <class XType>
void scale(const XType x, double s=1.) {

  typename XType::HostMirror x_h = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(x_h, x);

  for (size_t i = 0; i < x_h.extent(0); ++i)
    for (size_t j = 0; j < x_h.extent(1); ++j)
      x_h(i, j) *= s;

  Kokkos::deep_copy(x, x_h);
}

template <class XType>
void writeArrayToMM(std::string name, const XType x) {
  std::ofstream myfile;
  myfile.open(name);

  typename XType::HostMirror x_h = Kokkos::create_mirror_view(x);

  Kokkos::deep_copy(x_h, x);

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

template <class VType, class IntType>
void writeCRSToMM(std::string name, const VType &V, const IntType &r,
                   const IntType &c) {
  std::ofstream myfile;
  myfile.open(name);

  auto V_h = Kokkos::create_mirror_view(V);
  auto r_h = Kokkos::create_mirror_view(r);
  auto c_h = Kokkos::create_mirror_view(c);

  Kokkos::deep_copy(V_h, V);
  Kokkos::deep_copy(r_h, r);
  Kokkos::deep_copy(c_h, c);

  myfile << "%%MatrixMarket batched CRS matrix\n%" << std::endl;
  myfile << r_h.extent(0) - 1 << " " << r_h.extent(0) - 1 << " " << V_h.extent(1) << " " << V_h.extent(0) << std::endl;

  for (size_t i_row = 0; i_row < r_h.extent(0) - 1; ++i_row) {
    for (size_t i_nnz = r_h(i_row); i_nnz < r_h(i_row+1); ++i_nnz) {
      myfile << i_row + 1 << " " << c_h(i_nnz) + 1 << " ";
      for (size_t j = 0; j < V_h.extent(0); ++j) {  
        myfile << std::setprecision (15) << V_h(j, i_nnz) << " ";
      }
      myfile << std::endl;
    }
  }

  myfile.close();
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

template <class VType, class IntType>
void readCRSFromMM(std::string name, const VType &V, const IntType &r,
                   const IntType &c) {
  std::ifstream input(name);

  while (input.peek() == '%')
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  typename VType::HostMirror V_h   = Kokkos::create_mirror_view(V);
  typename IntType::HostMirror r_h = Kokkos::create_mirror_view(r);
  typename IntType::HostMirror c_h = Kokkos::create_mirror_view(c);

  int current_row = 0;
  int read_row;

  size_t nnz = c_h.extent(0);
  int nrows = r_h.extent(0)-1;

  r_h(0) = 0;

  for (size_t i = 0; i < nnz; ++i) {
    input >> read_row >> c_h(i);
    --read_row;
    --c_h(i);
    for (int tmp_row = current_row+1; tmp_row <= read_row; ++tmp_row)
      r_h(tmp_row) = i;
    current_row = read_row;

    // if (VType::Rank == 1)
    //  input >> V_h(i);
    if (VType::Rank == 2)
      for (size_t j = 0; j < V_h.extent(0); ++j) input >> V_h(j, i);
  }

  r_h(nrows) = nnz;

  input.close();

  Kokkos::deep_copy(V, V_h);
  Kokkos::deep_copy(r, r_h);
  Kokkos::deep_copy(c, c_h);

/*
  std::ofstream myfile;
  myfile.open("a-data.txt");


  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = r_h(i); j < r_h(i+1); ++j) {
      myfile << std::setprecision (15) << i+1 << " " << c_h(j)+1 << " " << V_h(0, j) << std::endl;
    }
  }

  myfile.close();
  */
}

template <class VType, class IntType>
void getInvDiagFromCRS(const VType &V, const IntType &r,
                   const IntType &c, const VType &diag) {
  auto diag_values_host = Kokkos::create_mirror_view(diag);
  auto values_host      = Kokkos::create_mirror_view(V);
  auto row_ptr_host     = Kokkos::create_mirror_view(r);
  auto colIndices_host  = Kokkos::create_mirror_view(c);

  Kokkos::deep_copy(values_host, V);
  Kokkos::deep_copy(row_ptr_host, r);
  Kokkos::deep_copy(colIndices_host, c);

  int current_index;
  int N = diag.extent(0);
  int BlkSize = diag.extent(1);

  for (int i = 0; i < BlkSize; ++i) {
    for (current_index = row_ptr_host(i); current_index < row_ptr_host(i + 1);
          ++current_index) {
      if (colIndices_host(current_index) == i) break;
    }
    for (int j = 0; j < N; ++j) {
      diag_values_host(j, i) = 1./values_host(j, current_index);
    }
  }

  Kokkos::deep_copy(diag, diag_values_host);

/*
  std::ofstream myfile;
  myfile.open("a-diag.txt");


  for (size_t i = 0; i < BlkSize; ++i) {
    myfile << std::setprecision (15) << i+1 << " " << diag_values_host(0, i) << std::endl;
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
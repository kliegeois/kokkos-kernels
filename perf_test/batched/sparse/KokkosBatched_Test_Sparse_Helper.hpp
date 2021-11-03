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
      myfile << x_h(i, j) << " ";
    }
    myfile << std::endl;
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

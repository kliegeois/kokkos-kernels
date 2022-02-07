import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import run_test_nvprof, getBuildDirectory
from create_matrices import *
from read_pele_matrices import *
import os


def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return 2*(nnz+nrows)*number_of_matrices*bytes_per_entry


def main():
    tic = time.perf_counter()
    Ns = np.arange(5100, 5300, 50)

    specie = 'gri30'

    input_folder = 'pele_data/jac-'+specie+'-typvals/'
    n_files = 72

    directory = getBuildDirectory()

    data_d = 'Pele_SPMV_NVPROF_' + specie + '_data_1'

    rows_per_thread=1
    team_size=32
    implementations_left = [0, 1, 2, 3]
    implementations_right = [0, 1, 2, 3]
    n_implementations_left = len(implementations_left)
    n_implementations_right = len(implementations_right)

    n1 = 5
    n2 = 10

    n_quantiles = 7

    CPU_time_left = np.zeros((n_implementations_left, len(Ns), n_quantiles))
    throughput_left = np.zeros((n_implementations_left, len(Ns), n_quantiles))
    CPU_time_right = np.zeros((n_implementations_right, len(Ns), n_quantiles))
    throughput_right = np.zeros((n_implementations_right, len(Ns), n_quantiles))
    nnzs = np.zeros((len(Ns), ))

    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    name_A = data_d+'/A.mm'
    name_B = data_d+'/B.mm'
    name_X = data_d+'/X'
    name_timers = data_d+'/timers'

    implementation = 3

    for i in range(0, len(Ns)):
        r, c, V, n = read_matrices(input_folder, n_files, Ns[i])
        nnzs[i] = len(r)
        n_ops = compute_n_ops(n, nnzs[i], Ns[i])

        B = read_vectors(input_folder, Ns[i], n)
    
        mmwrite(name_A, V, r, c, n, n)
        mmwrite(name_B, B)

        testfile = data_d+'/nvprof.'+str(implementation)+'.'+str(n)+'.'+str(nnzs[i])+'.'+str(Ns[i])

        nvprof_exe = '/home/projects/ppc64le-pwr9-nvidia/cuda/10.2.2/bin/nvprof'
        run_test_nvprof(nvprof_exe, directory+'/KokkosBatched_Test_SPMV', testfile, name_A, name_B, name_X, rows_per_thread, team_size, n1=n1, n2=n2, implementation=implementation, layout='Left', extra_args=' -P')
        run_test_nvprof(nvprof_exe, directory+'/KokkosBatched_Test_SPMV', testfile, name_A, name_B, name_X, rows_per_thread, team_size, n1=n1, n2=n2, implementation=implementation, layout='Right', extra_args=' -P')

    toc = time.perf_counter()
    print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

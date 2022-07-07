import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import *
from create_matrices import *
from read_pele_matrices import *
import os
import argparse


def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return 2*(nnz+nrows)*number_of_matrices*bytes_per_entry


def main():
    tic = time.perf_counter()
    Ns = np.array([224, 256])

    parser = argparse.ArgumentParser(description='Postprocess the results.')
    parser.add_argument('--specie', metavar='specie', default='gri30',
                        help='used specie')
    args = parser.parse_args()

    specie = args.specie

    input_folder = 'pele_data/jac-'+specie+'-typvals/'
    if specie == 'gri30':
        n_files = 90
    if specie == 'isooctane':
        n_files = 72

    Ns *= n_files

    input_folder = 'pele_data/jac-'+specie+'-typvals/'

    directory = getBuildDirectory()
    hostname = getHostName()

    data_d = 'Pele_SPMV_NVPROF_' + specie + '_data_2'

    rows_per_thread=1
    n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team_max, team_size, vector_length = getParameters(specie, 'left', hostname)

    n1 = 2
    n2 = 3

    n_quantiles = 7
    nnzs = np.zeros((len(Ns), ))

    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    name_A = data_d+'/A.mm'
    name_B = data_d+'/B.mm'
    name_X = data_d+'/X'
    name_timers = data_d+'/timers'

    np.savetxt(data_d+'/team_params.txt', np.array([team_size, vector_length, N_team_max]))

    implementations = [0, 3]

    for implementation in implementations:
        for N_team in range(1, N_team_max+1):
            for i in range(0, len(Ns)):
                r, c, V, n = read_matrices(input_folder, n_files, Ns[i])
                nnz = len(r)

                B = read_vectors(input_folder, Ns[i], n)
            
                mmwrite(name_A, V, r, c, n, n)
                mmwrite(name_B, B)

                testfile = data_d+'/nvprof.'+str(implementation)+'.'+str(n)+'.'+str(nnz)+'.'+str(Ns[i])+'.'+str(N_team)

                nvprof_exe = '/home/projects/ppc64le-pwr9-nvidia/cuda/10.2.2/bin/nvprof'
                run_test_nvprof(nvprof_exe, directory+'/KokkosBatched_Test_SPMV', testfile, name_A, name_B, name_X, rows_per_thread, team_size, n1=n1, n2=n2,
                    implementation=implementation, layout='Left', extra_args=' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team) +' -P')
                run_test_nvprof(nvprof_exe, directory+'/KokkosBatched_Test_SPMV', testfile, name_A, name_B, name_X, rows_per_thread, team_size, n1=n1, n2=n2,
                    implementation=implementation, layout='Right', extra_args=' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team) +' -P')

    toc = time.perf_counter()
    print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

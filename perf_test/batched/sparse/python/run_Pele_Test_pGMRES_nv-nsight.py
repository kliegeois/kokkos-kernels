import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import run_test, getBuildDirectory
from create_matrices import *
from read_pele_matrices import *
import os


def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return 2*(nnz+nrows)*number_of_matrices*bytes_per_entry


def getSortedIndices(specie, order):
    if specie == 'gri30':
        if order == 'descending':
            indices = np.array([31, 44, 57, 30, 33, 34, 35, 36, 37, 38, 39, 59, 40, 42, 43, 55, 45,
                46, 47, 48, 49, 50, 51, 41, 52, 60, 62, 79, 78, 77, 76, 75, 74, 61,
                72, 71, 73, 69, 70, 64, 65, 63, 67, 68, 66, 53, 56, 58, 82, 83, 84,
                85, 86, 87, 89, 88, 27, 28, 54, 32, 29,  3,  4,  5, 12, 81, 80,  7,
                8,  9, 10, 11,  6, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 13,  2,  1,  0])
        else:
            indices = np.array([ 0,  1,  2, 80, 81, 26, 25, 24, 23, 22, 21, 20, 18, 17, 16, 19, 14,
                13, 12, 11, 10,  9,  8,  7,  6,  5, 15,  3,  4, 56, 58, 82, 87, 84,
                85, 86, 54, 83, 88, 89, 27, 28, 29, 32, 40, 66, 67, 68, 69, 70, 71,
                72, 73, 76, 75, 65, 77, 78, 79, 34, 33, 30, 74, 39, 64, 62, 41, 42,
                43, 38, 45, 46, 47, 48, 49, 63, 50, 52, 53, 37, 55, 36, 57, 35, 59,
                60, 61, 51, 44, 31])
        return indices


def main():
    tic = time.perf_counter()
    N = 100

    specie = 'gri30'
    scaled = True
    order = 'descending'
    indices = getSortedIndices(specie,order)
    sort = True
    n_iterations = 10
    tol = 1e-8

    input_folder = 'pele_data/jac-'+specie+'-typvals/'
    if specie == 'gri30':
        n_files = 90
    if specie == 'isooctane':
        n_files = 72

    N *= n_files

    directory = getBuildDirectory()

    if sort:
        data_d = 'Pele_pGMRES_' + specie + '_data_Scaled_Jacobi_'+str(n_iterations)+'_sorted_nv'
    else:
        data_d = 'Pele_pGMRES_' + specie + '_data_Scaled_Jacobi_'+str(n_iterations)+'_unsorted_nv'

    rows_per_thread = 1
    team_size = 56 #32
    vector_length = 4 #8
    N_team = 16
    implementations_left = [3]
    implementations_right = [3]
    n_implementations_left = len(implementations_left)
    n_implementations_right = len(implementations_right)

    n1 = 1
    n2 = 1

    n_quantiles = 7

    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    name_A = data_d+'/A.mm'
    name_B = data_d+'/B.mm'
    name_X = data_d+'/X'
    name_timers = data_d+'/timers'

    r, c, V, n = read_matrices(input_folder, n_files, N, scaled, indices=indices, sort=sort)

    B = read_vectors(input_folder, N, n, scaled, indices=indices, sort=sort)

    mmwrite(name_A, V, r, c, n, n)
    mmwrite(name_B, B)

    options = '--section WarpStateStats --section SchedulerStats --section Occupancy --section SourceCounters'
    run_test('nv-nsight-cu-cli '+options+' -f -o '+data_d+'/report '+directory+'/KokkosBatched_Test_GMRES', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_left, layout='Left', extra_args=' -P -C -res '+data_d+'/P_res_l -n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team))
    toc = time.perf_counter()
    print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

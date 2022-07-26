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
    N = 224

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

    N *= n_files

    directory = getBuildDirectory()
    hostname = getHostName()

    n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters(specie, 'left', hostname)

    if not os.path.isdir(hostname):
        os.mkdir(hostname)
    data_d = hostname + '/Pele_SPMV_' + specie + '_data_SPMV_vec_m_default'

    rows_per_thread=1
    implementations_left = [0, 3]
    implementations_right = [0, 3]
    n_implementations_left = len(implementations_left)
    n_implementations_right = len(implementations_right)

    n1 = 2
    n2 = 3

    team_size = -1
    vector_length = -1
    N_team_min = 1
    N_team_max = 32
    N_teams = np.arange(N_team_min, N_team_max+1)

    n_quantiles = 7

    CPU_time_left = np.zeros((n_implementations_left, len(N_teams), n_quantiles))
    throughput_left = np.zeros((n_implementations_left, len(N_teams), n_quantiles))
    CPU_time_right = np.zeros((n_implementations_right, len(N_teams), n_quantiles))
    throughput_right = np.zeros((n_implementations_right, len(N_teams), n_quantiles))

    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    name_A = data_d+'/A.mm'
    name_B = data_d+'/B.mm'
    name_X = data_d+'/X'
    name_timers = data_d+'/timers'

    np.savetxt(data_d+'/team_params.txt', np.array([team_size, vector_length, N_team_max]))

    r, c, V, n = read_matrices(input_folder, n_files, N)
    if specie == 'gri30':
        nrows = 54
        nnz = 2560
    if specie == 'isooctane':
        nrows = 144
        nnz = 6135    
    n_ops = compute_n_ops(nrows, nnz, N)

    B = create_Vector(n, N)

    mmwrite(name_A, V, r, c, n, n)
    mmwrite(name_B, B)

    for i in range(0, len(N_teams)):
        N_team = N_teams[i]

        data = run_test(directory+'/KokkosBatched_Test_SPMV', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_left, layout='Left',
            extra_args=' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team))
        for j in range(0, n_implementations_left):
            CPU_time_left[j,i,:] = data[j,:]
        throughput_left[:,i,:] = n_ops/CPU_time_left[:,i,:]
        data = run_test(directory+'/KokkosBatched_Test_SPMV', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_right, layout='Right',
            extra_args=' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team))
        for j in range(0, n_implementations_right):
            CPU_time_right[j,i,:] = data[j,:]
        throughput_right[:,i,:] = n_ops/CPU_time_right[:,i,:]

        for j in range(0, n_implementations_left):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_left[j])+'_l.txt', CPU_time_left[j,:,:])
            np.savetxt(data_d+'/throughput_'+str(implementations_left[j])+'_l.txt', throughput_left[j,:,:])
        for j in range(0, n_implementations_right):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_right[j])+'_r.txt', CPU_time_right[j,:,:])
            np.savetxt(data_d+'/throughput_'+str(implementations_right[j])+'_r.txt', throughput_right[j,:,:])


if __name__ == "__main__":
    main()

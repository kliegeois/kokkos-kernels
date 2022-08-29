import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import *
from create_matrices import *
import os
import argparse


def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return 2*(nnz+nrows)*number_of_matrices*bytes_per_entry


def main():
    #tic = time.perf_counter()

    r, c, n = create_2D_Laplacian_graph(20, 10)
    Ns = np.array([1, 16, 32, 64, 96, 128, 160, 192, 224, 256])
    Ns *= 100

    parser = argparse.ArgumentParser(description='Postprocess the results.')
    parser.add_argument('--implementation', metavar='implementation', default=0,
                        help='used implementation')
    parser.add_argument('-d', action="store_true", default=False)
    args = parser.parse_args()

    implementation = int(args.implementation)

    default_params = args.d

    directory = getBuildDirectory()
    hostname = getHostName()

    if not os.path.isdir(hostname):
        os.mkdir(hostname)
    data_d = hostname + '/CG_data_Laplacian'

    if default_params:
        data_d += '_default_params'

    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    rows_per_thread=1
    implementations_left = [implementation]
    implementations_right = [implementation]
    n_implementations_left = len(implementations_left)
    n_implementations_right = len(implementations_right)

    n1 = 10
    n2 = 100

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

    max_offset = 3
    offset = 4
    verify = False

    for i in range(0, len(Ns)):

        V = create_SPD(n, r, c, Ns[i])
        B = create_Vector(n, Ns[i])
    
        mmwrite(name_A, V, r, c, n, n)
        mmwrite(name_B, B)

        N_team, team_size, vector_length, implementation = getParametersCG('left', hostname, implementation)

        if default_params:
            N_team = 8
            team_size = -1
            vector_length = -1

        extra_args = ' -n_iterations 20 -tol 1e-8'
        extra_args += ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)

        data = run_test(directory+'/KokkosBatched_Test_CG', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_left, layout='Left', extra_args=extra_args)
        for j in range(0, n_implementations_left):
            CPU_time_left[j,i,:] = data[j,:]

        N_team, team_size, vector_length, implementation = getParametersCG('right', hostname, implementation)

        if default_params:
            N_team = 8
            team_size = -1
            vector_length = -1

        extra_args = ' -n_iterations 20 -tol 1e-8'
        extra_args += ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)

        data = run_test(directory+'/KokkosBatched_Test_CG', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_right, layout='Right', extra_args=extra_args)
        for j in range(0, n_implementations_right):
            CPU_time_right[j,i,:] = data[j,:]

        for j in range(0, n_implementations_left):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_left[j])+'_l.txt', CPU_time_left[j,:,:])
        for j in range(0, n_implementations_right):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_right[j])+'_r.txt', CPU_time_right[j,:,:])
    
        np.savetxt(data_d+'/Ns.txt', Ns)


    #toc = time.perf_counter()
    #print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

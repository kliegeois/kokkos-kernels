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
    Ns = 256#00

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

    n1 = 2
    n2 = 10

    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    name_A = data_d+'/A.mm'
    name_B = data_d+'/B.mm'
    name_X = data_d+'/X'
    name_timers = data_d+'/timers'
    name_its = data_d+'/iterations'
    name_res = data_d+'/res'

    max_offset = 3
    offset = 4
    verify = False

    V = create_SPD(n, r, c, Ns)
    B = create_Vector(n, Ns)

    mmwrite(name_A, V, r, c, n, n)
    mmwrite(name_B, B)

    N_team, team_size, vector_length, implementation = getParametersCG('left', hostname, implementation)

    if default_params:
        N_team = 8
        team_size = -1
        vector_length = -1

    extra_args = ' -n_iterations 200' # -tol 1e-8'
    extra_args += ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team) + ' -C -its ' + name_its + ' -res ' + name_res

    data = run_test(directory+'/KokkosBatched_Test_CG', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_left, layout='Left', extra_args=extra_args)

    n_iterations = mmread(name_its+str(implementation)+'.mm')
    print(n_iterations)
    print(np.mean(n_iterations))
    print(np.std(n_iterations))


if __name__ == "__main__":
    main()

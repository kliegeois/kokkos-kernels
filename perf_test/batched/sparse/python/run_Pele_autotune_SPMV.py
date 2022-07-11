import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import *
from create_matrices import *
from read_pele_matrices import *
import os
import argparse
from bayes_opt import BayesianOptimization

name_A = ''
name_B = ''
name_X = ''
name_timers = ''
rows_per_thread = 1
n1 = 1
n2 = 1
n_ops = 1

def run_analysis_left_float(m, team_size, vector_length):
    return run_analysis_left(int(m), int(team_size), int(vector_length))
 

def run_analysis_left(m, team_size, vector_length):
    assert type(m) == int
    assert type(team_size) == int
    assert type(vector_length) == int

    if team_size*vector_length > 1024:
        return 0

    directory = getBuildDirectory()
    hostname = getHostName()

    parser = argparse.ArgumentParser(description='Postprocess the results.')
    parser.add_argument('--specie', metavar='specie', default='gri30',
                        help='used specie')
    args = parser.parse_args()
    specie = args.specie

    data_d = hostname + '/Pele_SPMV_' + specie + '_data_SPMV_vec_autotune'

    name_A = data_d+'/A.mm'
    name_B = data_d+'/B.mm'
    name_X = data_d+'/X'
    name_timers = data_d+'/timers'

    n1 = 5
    n2 = 10

    rows_per_thread = 1
    try:
        data = run_test(directory+'/KokkosBatched_Test_SPMV', name_A, name_B, name_X, name_timers, rows_per_thread, np.int(team_size), n1=n1, n2=n2, implementations=[3], layout='Left',
            extra_args=' -vector_length '+str(np.int(vector_length))+ ' -N_team '+str(np.int(m)))
        return n_ops/data[0,3]
    except: 
        return 0

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

    hostname = getHostName()

    if not os.path.isdir(hostname):
        os.mkdir(hostname)
    data_d = hostname + '/Pele_SPMV_' + specie + '_data_SPMV_vec_autotune'

    rows_per_thread=1

    n1 = 5
    n2 = 10

    n_quantiles = 7

    N_team_max = 16

    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    name_A = data_d+'/A.mm'
    name_B = data_d+'/B.mm'
    name_X = data_d+'/X'
    name_timers = data_d+'/timers'

    r, c, V, n = read_matrices(input_folder, n_files, N)
    nnz = len(c)
    n_ops = compute_n_ops(len(r)-1, nnz, N)

    B = create_Vector(n, N)

    mmwrite(name_A, V, r, c, n, n)
    mmwrite(name_B, B)


    # Bounded region of parameter space
    pbounds = {'m': (1, 32), 'team_size': (1, 256), 'vector_length': (1, 256)}

    optimizer = BayesianOptimization(
        f=run_analysis_left_float,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=250,
        alpha=1e-3,
    )    

    print(optimizer.max)

if __name__ == "__main__":
    main()

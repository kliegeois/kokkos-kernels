import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import *
from create_matrices import *
from read_pele_matrices import *
import os
import argparse
from skopt.space import Integer
from skopt import gp_minimize
from skopt.utils import use_named_args
from collections import namedtuple


def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return 2*(nnz+nrows)*number_of_matrices*bytes_per_entry


def run_analysis(params, fixed_params):
    if params.team_size*params.vector_length > fixed_params.max_size:
        return 0

    directory = getBuildDirectory()

    data = run_test(directory+'/KokkosBatched_Test_SPMV', fixed_params.name_A, fixed_params.name_B, 
        fixed_params.name_X, fixed_params.name_timers, fixed_params.rows_per_thread, params.team_size, n1=fixed_params.n1, n2=fixed_params.n2, implementations=[3], layout=fixed_params.layout,
        extra_args=' -vector_length '+str(params.vector_length)+ ' -N_team '+str(params.m))
    return -fixed_params.n_ops/data[0,3]


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
    data_d = hostname + '/Pele_SPMV_' + specie + '_data_SPMV_vec_autotune_skopt'

    rows_per_thread=1

    n1 = 5
    n2 = 10

    layout = 'Left'

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


    dim1 = Integer(1, 32, name='m')
    dim2 = Integer(1, 256, name='team_size')
    dim3 = Integer(1, 256, name='vector_length')

    dimensions = [dim1, dim2, dim3]

    VariableParams = namedtuple('VariableParams', 'm team_size vector_length')
    FixedParams = namedtuple('FixedParams', 'name_A name_B name_X name_timers n1 n2 rows_per_thread layout n_ops max_size')

    # define fixed params
    fixed_args = FixedParams(name_A, name_B, name_X, name_timers, n1, n2, rows_per_thread, layout, n_ops, 1024)

    @use_named_args(dimensions)
    def objective(m, team_size, vector_length):
        variable_args = VariableParams(m, team_size, vector_length)
        return run_analysis(variable_args, fixed_args)


    res_gp = gp_minimize(objective, dimensions,
        n_calls=50, n_random_starts=10, random_state=0)

    print(res_gp)


if __name__ == "__main__":
    main()

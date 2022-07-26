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
import json

def clean(name_timers, implementations):
    for implementation in implementations:
        if os.path.exists(name_timers+"_"+str(implementation)+"_right.txt"):
            os.remove(name_timers+"_"+str(implementation)+"_right.txt")
        if os.path.exists(name_timers+"_"+str(implementation)+"_left.txt"):
            os.remove(name_timers+"_"+str(implementation)+"_left.txt")

def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return 2*(nnz+nrows)*number_of_matrices*bytes_per_entry


def run_analysis(params, fixed_params):
    if params.team_size*params.vector_length > fixed_params.max_size:
        return 0

    clean(fixed_params.name_timers, [0, 1, 2, 3])

    directory = getBuildDirectory()

    key = str(params.vector_length)+'_'+str(params.m)+'_'+str(params.team_size)

    if key in fixed_params.previous_points:
        print('point '+key+ ' was already evaluated.')
    else:
        try:
            data = run_test(directory+'/KokkosBatched_Test_GMRES', fixed_params.name_A, fixed_params.name_B, 
                fixed_params.name_X, fixed_params.name_timers, fixed_params.rows_per_thread, params.team_size,
                n1=fixed_params.n1, n2=fixed_params.n2, implementations=[fixed_params.implementation], layout=fixed_params.layout,
                extra_args=' -vector_length '+str(params.vector_length)+ ' -N_team '+str(params.m))
            fixed_params.previous_points[key] = -1/data[0,3]
        except:
            fixed_params.previous_points[key] = 0.


    with open(fixed_params.log_file_name, 'w') as file:
        file.write(json.dumps(fixed_params.previous_points))

    return fixed_params.previous_points[key]


def main():
    tic = time.perf_counter()
    N = 224

    parser = argparse.ArgumentParser(description='Postprocess the results.')
    parser.add_argument('--specie', metavar='specie', default='gri30',
                        help='used specie')
    parser.add_argument('-r', action="store_true", default=False)
    args = parser.parse_args()

    if args.r:
        layout = 'Right'
    else:
        layout = 'Left'

    specie = args.specie

    input_folder = 'pele_data/jac-'+specie+'-typvals/'
    if specie == 'gri30':
        n_files = 90
    if specie == 'isooctane':
        n_files = 72

    N *= n_files

    hostname = getHostName()

    m_min = 1
    m_max = 32
    team_size_min = 1
    team_size_max = 1
    vector_length_min = 1
    vector_length_max = 1

    if hostname == 'weaver':
        team_size_min = 1
        team_size_max = 134
        vector_length_min = 1
        vector_length_max = 8
        max_size = 1024
        n_calls = 100
        n_random_starts = 10
    elif hostname == 'caraway':
        team_size_min = 1
        team_size_max = 256
        vector_length_min = 1
        vector_length_max = 8
        max_size = 1024
        n_calls = 100
        n_random_starts = 10
    elif hostname == 'inouye':
        team_size_min = 1
        team_size_max = 1.1
        vector_length_min = 8
        vector_length_max = 8.1
        max_size = 1024
        n_calls = 32
        n_random_starts = 1
    elif hostname == 'blake':
        team_size_min = 1
        team_size_max = 1.1
        vector_length_min = 8
        vector_length_max = 8.1
        max_size = 1024
        n_calls = 32
        n_random_starts = 1

    if not os.path.isdir(hostname):
        os.mkdir(hostname)
    data_d = hostname + '/Pele_' + specie + '_data_GMRES_autotune_skopt_' + layout

    rows_per_thread=1

    n1 = 2
    n2 = 5

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

    implementation = 0

    mmwrite(name_A, V, r, c, n, n)
    mmwrite(name_B, B)


    dim1 = Integer(m_min, m_max, name='m')
    dim2 = Integer(team_size_min, team_size_max, name='team_size')
    dim3 = Integer(vector_length_min, vector_length_max, name='vector_length')

    dimensions = [dim1, dim2, dim3]

    VariableParams = namedtuple('VariableParams', 'm team_size vector_length')
    FixedParams = namedtuple('FixedParams', 'name_A name_B name_X name_timers n1 n2 rows_per_thread layout n_ops max_size implementation previous_points log_file_name')

    # define fixed params
    fixed_args = FixedParams(name_A, name_B, name_X, name_timers, n1, n2, rows_per_thread, layout, n_ops, max_size, implementation, {}, data_d+'/log.txt')

    @use_named_args(dimensions)
    def objective(m, team_size, vector_length):
        variable_args = VariableParams(m, team_size, vector_length)
        return run_analysis(variable_args, fixed_args)

    if hostname != 'blake':
        res_gp = gp_minimize(objective, dimensions,
            n_calls=n_calls, n_random_starts=n_random_starts, random_state=0)

        print(res_gp)
        toc = time.perf_counter()
        print("Elapsed time = " + str(toc-tic) + " seconds")

        with open(data_d+'/results.txt', 'w') as f:
            print(res_gp, file=f)
            print("Elapsed time = " + str(toc-tic) + " seconds", file=f)
            print(res_gp.x, file=f)
            print(res_gp.fun, file=f)
    else:
        ms = np.arange(m_min, m_max+1)
        output = np.zeros((len(ms),))

        for i in range(0, len(ms)):
            variable_args = VariableParams(ms[i], team_size_min, vector_length_min)         
            output[i] = run_analysis(variable_args, fixed_args)
        
        index = np.argmin(output)
        toc = time.perf_counter()

        with open(data_d+'/results.txt', 'w') as f:
            print(ms, file=f)
            print(output, file=f)
            print("Elapsed time = " + str(toc-tic) + " seconds", file=f)
            print(ms[index], file=f)
            print(output[index], file=f)
        
if __name__ == "__main__":
    main()

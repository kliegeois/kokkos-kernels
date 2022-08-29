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
    #tic = time.perf_counter()
    Ns = np.array([1, 16, 32, 64, 96, 128, 160, 192, 224, 256])

    parser = argparse.ArgumentParser(description='Postprocess the results.')
    parser.add_argument('--specie', metavar='specie', default='gri30',
                        help='used specie')
    parser.add_argument('--implementation', metavar='implementation', default=0,
                        help='used implementation')
    parser.add_argument('-s', action="store_true", default=False)
    parser.add_argument('-d', action="store_true", default=False)
    args = parser.parse_args()

    specie = args.specie
    scaled = True
    order = 'descending'
    indices = getSortedIndices(specie,order)
    sort = args.s

    implementation = int(args.implementation)

    input_folder = 'pele_data/jac-'+specie+'-typvals/'
    if specie == 'gri30':
        n_files = 90
    if specie == 'isooctane':
        n_files = 72

    Ns *= 100

    directory = getBuildDirectory()
    hostname = getHostName()

    default_params = args.d

    n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length, implementation = getParametersGMRES(specie, 'left', hostname, implementation)

    #n_iterations = 10
    #tol = 0

    if default_params:
        N_team = 8
        team_size = -1
        vector_length = -1
    
    if not os.path.isdir(hostname):
        os.mkdir(hostname)
    
    data_d = hostname + '/Pele_pGMRES_' + specie + '_data_Scaled_Jacobi_'+str(n_iterations)+'_'+str(ortho_strategy)+'_'+str(arnoldi_level)+'_'+str(other_level)
    if sort:
        data_d += '_sorted'
    else:
        data_d += '_unsorted'

    if default_params:
        data_d += '_default_params'

    rows_per_thread = 1
    implementations_left = [implementation]
    implementations_right = [implementation]
    n_implementations_left = len(implementations_left)
    n_implementations_right = len(implementations_right)

    n1 = 20
    n2 = 20

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

    for i in range(0, len(Ns)):
        r, c, V, n = read_matrices(input_folder, n_files, Ns[i], scaled, indices=indices, sort=sort)
        nnzs[i] = len(r)
        n_ops = compute_n_ops(n, nnzs[i], Ns[i])

        B = read_vectors(input_folder, Ns[i], n, scaled, indices=indices, sort=sort)
    
        mmwrite(name_A, V, r, c, n, n)
        mmwrite(name_B, B)
        n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length, implementation = getParametersGMRES(specie, 'left', hostname, implementation)
        if default_params:
            N_team = 8
            team_size = -1
            vector_length = -1
        extra_arg = ' -n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
        extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
        data = run_test(directory+'/KokkosBatched_Test_GMRES', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_left, layout='Left', extra_args=' -P -C -res '+data_d+'/P_res_l '+extra_arg)
        for j in range(0, n_implementations_left):
            CPU_time_left[j,i,:] = data[j,:]
        throughput_left[:,i,:] = n_ops/CPU_time_left[:,i,:]
        n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length, implementation = getParametersGMRES(specie, 'right', hostname, implementation)
        if default_params:
            N_team = 8
            team_size = -1
            vector_length = -1
        extra_arg = ' -n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
        extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
        data = run_test(directory+'/KokkosBatched_Test_GMRES', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_right, layout='Right', extra_args=' -P -C -res '+data_d+'/P_res_r '+extra_arg)
        for j in range(0, n_implementations_right):
            CPU_time_right[j,i,:] = data[j,:]
        throughput_right[:,i,:] = n_ops/CPU_time_right[:,i,:]

        for j in range(0, n_implementations_left):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_left[j])+'_l.txt', CPU_time_left[j,:,:])
            np.savetxt(data_d+'/throughput_'+str(implementations_left[j])+'_l.txt', throughput_left[j,:,:])
        for j in range(0, n_implementations_right):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_right[j])+'_r.txt', CPU_time_right[j,:,:])
            np.savetxt(data_d+'/throughput_'+str(implementations_right[j])+'_r.txt', throughput_right[j,:,:])
        np.savetxt(data_d+'/Ns.txt', Ns)
        np.savetxt(data_d+'/nnzs.txt', nnzs)

    #toc = time.perf_counter()
    #print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

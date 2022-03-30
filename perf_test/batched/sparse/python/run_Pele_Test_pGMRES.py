import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import run_test, getHostName, getBuildDirectory
from create_matrices import *
from read_pele_matrices import *
import os
import argparse


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
    if specie == 'isooctane':
        if order == 'descending':
            indices = np.array([71, 60, 50, 49, 37, 66, 33, 57, 41, 16, 43, 14, 40, 17, 15, 13, 45,
                46, 47, 55, 38, 58, 36, 34, 65, 32, 31, 30, 35, 28,  9, 25, 10, 11,
                20, 52, 53, 54,  6, 56, 29, 51, 61, 62, 63, 64, 67, 68, 69,  5, 59,
                7, 48, 27, 26, 24, 23, 70, 22,  8, 21, 39, 42, 12, 44,  4, 19,  2,
                3, 18,  1,  0])
        else:
            indices = np.array([ 0,  1, 18,  2,  3,  4, 19, 23, 24, 26, 27, 56, 53, 54, 52, 51, 70,
                48, 39, 29, 22, 21, 59,  5,  6,  7,  8, 69, 44, 67, 68, 12, 64, 63,
                62, 61, 42, 65, 58, 47, 46, 45, 55, 35, 20, 38,  9, 36, 10, 11, 34,
                25, 32, 31, 30, 28, 13, 17, 15, 40, 14, 41, 57, 43, 16, 33, 66, 37,
                50, 49, 60, 71])
    return indices


def getParameters(specie, hostname):
    tol = 1e-8
    if hostname == 'weaver':
        if specie == 'gri30':
            n_iterations = 8
            ortho_strategy = 0
            arnoldi_level = 11
            other_level = 0
            N_team = 16
            team_size = -1
            vector_length = 16
        if specie == 'isooctane':
            n_iterations = 20
            ortho_strategy = 0
            arnoldi_level = 11
            other_level = 0
            N_team = 6
            team_size = -1
            vector_length = 6
    if hostname == 'caraway':
        if specie == 'gri30':
            n_iterations = 8
            ortho_strategy = 0
            arnoldi_level = 11
            other_level = 0
            N_team = 16
            team_size = 16
            vector_length = 16
        if specie == 'isooctane':
            n_iterations = 20
            ortho_strategy = 0
            arnoldi_level = 11
            other_level = 0
            N_team = 8
            team_size = 32
            vector_length = 8
    if hostname == 'inouye':
        if specie == 'gri30':
            n_iterations = 8
            ortho_strategy = 0
            arnoldi_level = 11
            other_level = 0
            N_team = 8
            team_size = -1
            vector_length = 8
        if specie == 'isooctane':
            n_iterations = 20
            ortho_strategy = 0
            arnoldi_level = 11
            other_level = 0
            N_team = 8
            team_size = -1
            vector_length = 8
    if hostname == 'blake':
        if specie == 'gri30':
            n_iterations = 8
            ortho_strategy = 0
            arnoldi_level = 11
            other_level = 0
            N_team = 8
            team_size = -1
            vector_length = 8
        if specie == 'isooctane':
            n_iterations = 20
            ortho_strategy = 0
            arnoldi_level = 11
            other_level = 0
            N_team = 8
            team_size = -1
            vector_length = 8
    return n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length

def main():
    tic = time.perf_counter()
    Ns = np.array([1, 16, 32, 64, 96, 128, 160, 192, 224, 256])

    parser = argparse.ArgumentParser(description='Postprocess the results.')
    parser.add_argument('--specie', metavar='specie', default='gri30',
                        help='used specie')
    parser.add_argument('-s', action="store_true", default=False)
    args = parser.parse_args()

    specie = args.specie
    scaled = True
    order = 'descending'
    indices = getSortedIndices(specie,order)
    sort = args.s


    input_folder = 'pele_data/jac-'+specie+'-typvals/'
    if specie == 'gri30':
        n_files = 90
    if specie == 'isooctane':
        n_files = 72

    Ns *= n_files

    directory = getBuildDirectory()
    hostname = getHostName()

    n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters(specie, hostname)

    #n_iterations = 10
    #tol = 0
    
    if not os.path.isdir(hostname):
        os.mkdir(hostname)
    if sort:
        data_d = hostname + '/Pele_pGMRES_' + specie + '_data_Scaled_Jacobi_'+str(n_iterations)+'_'+str(ortho_strategy)+'_'+str(arnoldi_level)+'_'+str(other_level)+'_sorted'
    else:
        data_d = hostname + '/Pele_pGMRES_' + specie + '_data_Scaled_Jacobi_'+str(n_iterations)+'_'+str(ortho_strategy)+'_'+str(arnoldi_level)+'_'+str(other_level)+'_unsorted'

    rows_per_thread = 1
    implementations_left = [3]
    implementations_right = [3]
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
        extra_arg = '-n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
        extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
        data = run_test(directory+'/KokkosBatched_Test_GMRES', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_left, layout='Left', extra_args=' -P -C -res '+data_d+'/P_res_l '+extra_arg)
        for j in range(0, n_implementations_left):
            CPU_time_left[j,i,:] = data[j,:]
        throughput_left[:,i,:] = n_ops/CPU_time_left[:,i,:]
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

    toc = time.perf_counter()
    #print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

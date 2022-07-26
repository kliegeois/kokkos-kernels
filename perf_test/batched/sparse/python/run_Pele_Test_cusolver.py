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

    n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters(specie, 'left', hostname)

    #n_iterations = 10
    #tol = 0
    
    if not os.path.isdir(hostname):
        os.mkdir(hostname)
    if sort:
        data_d = hostname + '/Pele_cusolve_' + specie + '_data_Scaled_Jacobi_cusolver_sorted'
    else:
        data_d = hostname + '/Pele_cusolve_' + specie + '_data_Scaled_Jacobi_cusolver_unsorted'

    rows_per_thread = 1
    implementations_sp = [0]
    implementations_dn = [0]
    n_implementations_sp = len(implementations_sp)
    n_implementations_dn = len(implementations_dn)

    n1 = 20
    n2 = 20

    n_quantiles = 7

    CPU_time_sp = np.zeros((n_implementations_sp, len(Ns), n_quantiles))
    throughput_sp = np.zeros((n_implementations_sp, len(Ns), n_quantiles))
    CPU_time_dn = np.zeros((n_implementations_dn, len(Ns), n_quantiles))
    throughput_dn = np.zeros((n_implementations_dn, len(Ns), n_quantiles))
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
        n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters(specie, 'left', hostname)
        extra_arg = '-n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
        extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
        data = run_test(directory+'/KokkosBatched_Test_cusolverSp', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_sp, layout='Right', extra_args=' -P -C -res '+data_d+'/P_res_l '+extra_arg)
        for j in range(0, n_implementations_sp):
            CPU_time_sp[j,i,:] = data[j,:]
        throughput_sp[:,i,:] = n_ops/CPU_time_sp[:,i,:]
        n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters(specie, 'right', hostname)
        extra_arg = '-n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
        extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
        data = run_test(directory+'/KokkosBatched_Test_cusolverDn', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_dn, layout='Right', extra_args=' -P -C -res '+data_d+'/P_res_r '+extra_arg)
        for j in range(0, n_implementations_dn):
            CPU_time_dn[j,i,:] = data[j,:]
        throughput_dn[:,i,:] = n_ops/CPU_time_dn[:,i,:]

        for j in range(0, n_implementations_sp):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_sp[j])+'_sp.txt', CPU_time_sp[j,:,:])
            np.savetxt(data_d+'/throughput_'+str(implementations_sp[j])+'_sp.txt', throughput_sp[j,:,:])
        for j in range(0, n_implementations_dn):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_dn[j])+'_dn.txt', CPU_time_dn[j,:,:])
            np.savetxt(data_d+'/throughput_'+str(implementations_dn[j])+'_dn.txt', throughput_dn[j,:,:])
        np.savetxt(data_d+'/Ns.txt', Ns)
        np.savetxt(data_d+'/nnzs.txt', nnzs)

    #toc = time.perf_counter()
    #print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

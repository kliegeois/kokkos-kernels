import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import *
from create_matrices import *
from read_pele_matrices import *
import os
import argparse

def clean(data_d, implementations):
    for implementation in implementations:
        if os.path.exists(data_d+"/timers_"+str(implementation)+"_right.txt"):
            os.remove(data_d+"/timers_"+str(implementation)+"_right.txt")
        if os.path.exists(data_d+"/timers_"+str(implementation)+"_left.txt"):
            os.remove(data_d+"/timers_"+str(implementation)+"_left.txt")


def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return 2*(nnz+nrows)*number_of_matrices*bytes_per_entry

def main():
    #tic = time.perf_counter()

    Ns = np.array([1, 16, 32, 64, 96, 128, 160, 192, 224, 256, 512, 1024]) * 90
    max_offset = 3
    offset = 4
    blk = 80
    N_sequential = 20

    directory = getBuildDirectory()
    hostname = getHostName()

    n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters('gri30', 'left', hostname)
    
    #N_team = 16
    #team_size = -1
    #vector_length = -1
    
    if not os.path.isdir(hostname):
        os.mkdir(hostname)
    data_d = hostname + '/cusolve_N'

    rows_per_thread = 1
    implementations_sp = [0]
    implementations_dn = [0]
    implementations_CG = [3]
    implementations_GMRES = [3]
    n_implementations_sp = len(implementations_sp)
    n_implementations_dn = len(implementations_dn)
    n_implementations_CG = len(implementations_CG)
    n_implementations_GMRES = len(implementations_GMRES)

    n1 = 20
    n2 = 20

    n_quantiles = 7

    CPU_time_sp = np.zeros((n_implementations_sp, len(Ns), n_quantiles))
    CPU_time_dn = np.zeros((n_implementations_dn, len(Ns), n_quantiles))
    CPU_time_CG_left = np.zeros((n_implementations_CG, len(Ns), n_quantiles))
    CPU_time_CG_right = np.zeros((n_implementations_CG, len(Ns), n_quantiles))
    CPU_time_GMRES_left = np.zeros((n_implementations_GMRES, len(Ns), n_quantiles))
    CPU_time_GMRES_right = np.zeros((n_implementations_GMRES, len(Ns), n_quantiles))

    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    name_A = data_d+'/A.mm'
    name_B = data_d+'/B.mm'
    name_X = data_d+'/X'
    name_timers = data_d+'/timers'


    r, c = create_strided_graph(blk, max_offset, offset)
    V = create_SPD(blk, r, c, N_sequential)
    B = create_Vector(blk, N_sequential)

    mmwrite(name_A, V, r, c, blk, blk)
    mmwrite(name_B, B)

    clean(data_d, [0,3])

    extra_arg = '-n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
    extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
    try:
        data = run_test(directory+'/KokkosBatched_Test_cusolverSp', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_sp, layout='Right', extra_args=' -P -C -res '+data_d+'/P_res_l '+extra_arg)
        for j in range(0, n_implementations_sp):
            for i in range(0, len(Ns)):
                CPU_time_sp[j,i,:] = data[j,:] * Ns[i] / N_sequential
    except:
        for i in range(0, len(Ns)):
            CPU_time_sp[:,i,:] = 0.

    for i in range(0, len(Ns)):

        V = create_SPD(blk, r, c, Ns[i])
        B = create_Vector(blk, Ns[i])
    
        mmwrite(name_A, V, r, c, blk, blk)
        mmwrite(name_B, B)

        clean(data_d, [0,3])
        n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters('gri30', 'left', hostname)

        extra_arg = '-n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
        extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
        try:
            data = run_test(directory+'/KokkosBatched_Test_cusolverDn', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_dn, layout='Right', extra_args=' -P -C -res '+data_d+'/P_res_r '+extra_arg)
            for j in range(0, n_implementations_dn):
                CPU_time_dn[j,i,:] = data[j,:]
        except:
            CPU_time_dn[:,i,:] = 0.

        clean(data_d, [0,3])

        n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters('gri30', 'left', hostname)
        extra_arg = '-n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
        extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
        try:
            data = run_test(directory+'/KokkosBatched_Test_CG', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_GMRES, layout='Left', extra_args=' -P -C -res '+data_d+'/P_res_l '+extra_arg)
            for j in range(0, n_implementations_CG):
                CPU_time_CG_left[j,i,:] = data[j,:]
        except:
            CPU_time_CG_left[:,i,:] = 0.

        clean(data_d, [0,3])

        try:
            data = run_test(directory+'/KokkosBatched_Test_GMRES', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_GMRES, layout='Left', extra_args=' -P -C -res '+data_d+'/P_res_l '+extra_arg)
            for j in range(0, n_implementations_GMRES):
                CPU_time_GMRES_left[j,i,:] = data[j,:]
        except:
            CPU_time_GMRES_left[:,i,:] = 0.

        clean(data_d, [0,3])

        n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters('gri30', 'right', hostname)
        extra_arg = '-n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
        extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
        try:
            data = run_test(directory+'/KokkosBatched_Test_CG', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_GMRES, layout='Right', extra_args=' -P -C -res '+data_d+'/P_res_l '+extra_arg)
            for j in range(0, n_implementations_CG):
                CPU_time_CG_right[j,i,:] = data[j,:]
        except:
            CPU_time_CG_right[:,i,:] = 0.

        clean(data_d, [0,3])

        try:
            data = run_test(directory+'/KokkosBatched_Test_GMRES', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_GMRES, layout='Right', extra_args=' -P -C -res '+data_d+'/P_res_l '+extra_arg)
            for j in range(0, n_implementations_GMRES):
                CPU_time_GMRES_right[j,i,:] = data[j,:]
        except:
            CPU_time_GMRES_right[:,i,:] = 0.

        for j in range(0, n_implementations_sp):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_sp[j])+'_sp.txt', CPU_time_sp[j,:,:])
        for j in range(0, n_implementations_dn):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_dn[j])+'_dn.txt', CPU_time_dn[j,:,:])
        for j in range(0, n_implementations_CG):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_dn[j])+'_CG_left.txt', CPU_time_CG_left[j,:,:])
            np.savetxt(data_d+'/CPU_time_'+str(implementations_dn[j])+'_CG_right.txt', CPU_time_CG_right[j,:,:])
        for j in range(0, n_implementations_GMRES):
            np.savetxt(data_d+'/CPU_time_'+str(implementations_dn[j])+'_GMRES_left.txt', CPU_time_GMRES_left[j,:,:])
            np.savetxt(data_d+'/CPU_time_'+str(implementations_dn[j])+'_GMRES_right.txt', CPU_time_GMRES_right[j,:,:])
        np.savetxt(data_d+'/Ns.txt', Ns)

    #toc = time.perf_counter()
    #print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

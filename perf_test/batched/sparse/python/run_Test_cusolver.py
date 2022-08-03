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

    n_node_1D_Js = np.arange(2, 31, 1)
    n_node_1D_I = 10

    Bs = np.zeros((len(n_node_1D_Js), ), dtype=int)

    max_offset = 3
    offset = 4
    N = 20000
    N_sequential = 20
    ratio_N = N/N_sequential

    directory = getBuildDirectory()
    hostname = getHostName()

    n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters('isooctane', 'left', hostname)
    
    N_team = 6
    team_size = -1
    vector_length = -1

    n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters('gri30', 'left', hostname)

    N_team = 8
    team_size = -1
    vector_length = 16

    n_iterations = 20

    #n_iterations = 10
    #tol = 0
    
    if not os.path.isdir(hostname):
        os.mkdir(hostname)
    data_d = hostname + '/cusolve_8'

    rows_per_thread = 1
    implementations_sp = [0]
    implementations_dn = [0]
    implementations_CG = [0]
    implementations_GMRES = [0]
    n_implementations_sp = len(implementations_sp)
    n_implementations_dn = len(implementations_dn)
    n_implementations_CG = len(implementations_CG)
    n_implementations_GMRES = len(implementations_GMRES)

    n1 = 3
    n2 = 5

    n_quantiles = 7

    CPU_time_sp = np.zeros((n_implementations_sp, len(Bs), n_quantiles))
    CPU_time_dn = np.zeros((n_implementations_dn, len(Bs), n_quantiles))
    CPU_time_CG_left = np.zeros((n_implementations_CG, len(Bs), n_quantiles))
    CPU_time_CG_right = np.zeros((n_implementations_CG, len(Bs), n_quantiles))
    CPU_time_GMRES_left = np.zeros((n_implementations_GMRES, len(Bs), n_quantiles))
    CPU_time_GMRES_right = np.zeros((n_implementations_GMRES, len(Bs), n_quantiles))

    nnzs = np.zeros((len(Bs), ))

    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    name_A = data_d+'/A.mm'
    name_B = data_d+'/B.mm'
    name_X_cusolver_sp = data_d+'/X_cusolver_sp'
    name_X_cusolver_dn = data_d+'/X_cusolver_dn'
    name_X_CG = data_d+'/X_CG'
    name_X_GMRES = data_d+'/X_GMRES'
    name_timers = data_d+'/timers'

    run_cusolvers_Sp = True
    run_cusolvers_Dn = False
    run_CG = False
    run_GMRES = False

    run_left = True
    run_right = False

    for i in range(0, len(Bs)):
        r, c, Bs[i] = create_2D_Laplacian_graph(n_node_1D_I, n_node_1D_Js[i])
        nnzs[i] = len(c)

        V = create_SPD(Bs[i], r, c, N_sequential)
        B = create_Vector(Bs[i], N_sequential)
    
        mmwrite(name_A, V, r, c, Bs[i], Bs[i])
        mmwrite(name_B, B)

        #n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters('isooctane', 'left', hostname)

        if run_cusolvers_Sp:
            clean(data_d, [0,1,2,3])

            extra_arg = ' -N_team '+str(N_team)
            try:
                data = run_test(directory+'/KokkosBatched_Test_cusolverSp', name_A, name_B, name_X_cusolver_sp, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_sp, layout='Right', extra_args= extra_arg)
                for j in range(0, n_implementations_sp):
                    CPU_time_sp[j,i,:] = data[j,:] * ratio_N
            except:
                CPU_time_sp[:,i,:] = 0.

        V = create_SPD(Bs[i], r, c, N)
        B = create_Vector(Bs[i], N)
    
        mmwrite(name_A, V, r, c, Bs[i], Bs[i])
        mmwrite(name_B, B)

        if run_cusolvers_Dn:
            clean(data_d, [0,1,2,3])

            extra_arg = '-n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
            extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
            try:
                data = run_test(directory+'/KokkosBatched_Test_cusolverDn', name_A, name_B, name_X_cusolver_dn, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_dn, layout='Right', extra_args=' -P -C -res '+data_d+'/P_res_r '+extra_arg)
                for j in range(0, n_implementations_dn):
                    CPU_time_dn[j,i,:] = data[j,:]
            except:
                CPU_time_dn[:,i,:] = 0.

        #n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters('isooctane', 'left', hostname)
        #N_team, team_size, vector_length = getParametersCG('left', hostname)
        extra_arg = '-n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
        extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
        if run_CG and run_left:
            for j in range(0, n_implementations_CG):
                clean(data_d, [0,1,2,3])

                try:
                    data = run_test(directory+'/KokkosBatched_Test_CG', name_A, name_B, name_X_CG, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=[implementations_CG[j]], layout='Left', extra_args=' -P -C -res '+data_d+'/P_res_l '+extra_arg)
                    CPU_time_CG_left[j,i,:] = data[0,:]
                except:
                    CPU_time_CG_left[j,i,:] = 0.

        if run_GMRES and run_left:
            for j in range(0, n_implementations_GMRES):
                clean(data_d, [0,1,2,3])

                try:
                    data = run_test(directory+'/KokkosBatched_Test_GMRES', name_A, name_B, name_X_GMRES, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=[implementations_GMRES[j]], layout='Left', extra_args=' -P -C -res '+data_d+'/P_res_GMRES_l'+str(i)+'_ '+extra_arg)
                    CPU_time_GMRES_left[j,i,:] = data[0,:]
                except:
                    CPU_time_GMRES_left[j,i,:] = 0.

        #n_iterations, tol, ortho_strategy, arnoldi_level, other_level, N_team, team_size, vector_length = getParameters('isooctane', 'right', hostname)
        #N_team, team_size, vector_length = getParametersCG('right', hostname)
        extra_arg = '-n_iterations '+str(n_iterations)+' -tol '+str(tol) + ' -vector_length '+str(vector_length)+ ' -N_team '+str(N_team)+' -ortho_strategy '+str(ortho_strategy)
        extra_arg += ' -arnoldi_level '+str(arnoldi_level) + ' -other_level '+str(other_level)
        if run_CG and run_right:
            for j in range(0, n_implementations_CG):
                clean(data_d, [0,1,2,3])

                try:
                    data = run_test(directory+'/KokkosBatched_Test_CG', name_A, name_B, name_X_CG, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=[implementations_CG[j]], layout='Right', extra_args=' -P -C -res '+data_d+'/P_res_l '+extra_arg)
                    CPU_time_CG_right[j,i,:] = data[0,:]
                except:
                    CPU_time_CG_right[j,i,:] = 0.

        clean(data_d, [0,1,2,3])

        if run_GMRES and run_right:
            for j in range(0, n_implementations_GMRES):
                clean(data_d, [0,1,2,3])

                try:
                    data = run_test(directory+'/KokkosBatched_Test_GMRES', name_A, name_B, name_X_GMRES, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=[implementations_GMRES[j]], layout='Right', extra_args=' -P -C -res '+data_d+'/P_res_GMRES_r'+str(i)+'_ '+extra_arg)
                    CPU_time_GMRES_right[j,i,:] = data[0,:]
                except:
                    CPU_time_GMRES_right[j,i,:] = 0.

        if run_cusolvers_Sp:
            for j in range(0, n_implementations_sp):
                np.savetxt(data_d+'/CPU_time_'+str(implementations_sp[j])+'_sp.txt', CPU_time_sp[j,:,:])
        if run_cusolvers_Dn:
            for j in range(0, n_implementations_dn):
                np.savetxt(data_d+'/CPU_time_'+str(implementations_dn[j])+'_dn.txt', CPU_time_dn[j,:,:])
        if run_CG:
            for j in range(0, n_implementations_CG):
                if run_left:
                    np.savetxt(data_d+'/CPU_time_'+str(implementations_CG[j])+'_CG_left.txt', CPU_time_CG_left[j,:,:])
                if run_right:
                    np.savetxt(data_d+'/CPU_time_'+str(implementations_CG[j])+'_CG_right.txt', CPU_time_CG_right[j,:,:])
        if run_GMRES:
            for j in range(0, n_implementations_GMRES):
                if run_left:
                    np.savetxt(data_d+'/CPU_time_'+str(implementations_GMRES[j])+'_GMRES_left.txt', CPU_time_GMRES_left[j,:,:])
                if run_right:
                    np.savetxt(data_d+'/CPU_time_'+str(implementations_GMRES[j])+'_GMRES_right.txt', CPU_time_GMRES_right[j,:,:])
        np.savetxt(data_d+'/Bs.txt', Bs)
        np.savetxt(data_d+'/nnzs.txt', nnzs)

    #toc = time.perf_counter()
    #print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

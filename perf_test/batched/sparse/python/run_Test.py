import numpy as np
import subprocess
import time
import os
import socket
import re


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


def getParameters(specie, layout, hostname):
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
    elif hostname == 'caraway':
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
    elif hostname == 'inouye':
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
    elif hostname == 'blake':
        if specie == 'gri30':
            n_iterations = 8
            ortho_strategy = 0
            arnoldi_level = 11
            other_level = 0
            '''
            if layout == 'right':
                N_team = 1
                team_size = 1
                vector_length = 8
            if layout == 'left':
                N_team = 32
                team_size = 1
                vector_length = 20
            '''
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
    else:
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


def getHostName():
    host_name = socket.gethostname().split(".")[0]
    host_name = re.sub('[0-9]+$', '', host_name)
    return host_name


def getBuildDirectory():
    host_name = getHostName()

    if os.path.exists('binary_dir_'+host_name+'.txt'):
        with open('binary_dir_'+host_name+'.txt') as f:
            directory = f.read()
    return directory


def run_test(exec_name, A_file_name, B_file_name, X_file_name, timer_filename, rows_per_thread=1, team_size=8, n1=10, n2=10, implementations=[0], layout='Left', quantiles=[0,0.1,0.2,0.5,0.8,0.9,1.], extra_args=''):
    n_implementations = len(implementations)
    exe = exec_name + ' -A ' +A_file_name+ ' -B '+B_file_name
    exe += ' -X ' + X_file_name + ' -timers '+ timer_filename
    exe += ' -n1 '+ str(n1)+' -n2 '+ str(n2)+' -rows_per_thread '+str(rows_per_thread)
    exe += ' -team_size '+str(team_size)
    for implementation in implementations:
        exe += ' -implementation '+str(implementation)
        if layout == 'Left':
            exe += ' -l'
            y_name = '_l.txt'
            time_name = '_left.txt'
        if layout == 'Right':
            exe += ' -r'
            y_name = '_r.txt'
            time_name = '_right.txt'
    exe += extra_args
    for i in range(0, n_implementations):
        current_name = 'y_'+str(implementations[i])+y_name
        if os.path.exists(current_name):
            os.remove(current_name)
    subprocess.call(exe, shell=True)

    n_quantiles = len(quantiles)

    data = np.zeros((n_implementations, n_quantiles))
    for i in range(0, n_implementations):
        tmp = np.loadtxt(timer_filename+'_'+str(implementations[i])+time_name)
        if tmp.size > 1:
            data[i, :] = np.quantile(tmp, quantiles)
        elif tmp.size > 0:
            data[i, :] = tmp
    return data


def run_test_nvprof(nvprof_exe, exec_name, testfile, A_file_name, B_file_name, X_file_name, rows_per_thread=1, team_size=8, n1=10, n2=10, implementation=0, layout='Left', quantiles=[0,0.1,0.2,0.5,0.8,0.9,1.], extra_args=''):
    exe = exec_name + ' -A ' +A_file_name+ ' -B '+B_file_name + ' -X ' + X_file_name
    exe += ' -n1 '+ str(n1)+' -n2 '+ str(n2)+' -rows_per_thread '+str(rows_per_thread)
    exe += ' -team_size '+str(team_size)+ ' -implementation '+str(implementation)
    exe += extra_args
    if layout == 'Left':
        exe += ' -l'
        nvprof_name = '.left.txt'
    if layout == 'Right':
        exe += ' -r'
        nvprof_name = '.right.txt'
    
    nv_prof=nvprof_exe + ' --metrics achieved_occupancy,dram_read_bytes,dram_read_throughput,dram_read_transactions,dram_utilization,dram_write_throughput,gst_throughput,gld_throughput,l2_l1_read_throughput,l2_tex_read_throughput,l2_tex_write_throughput,local_load_throughput,local_store_throughput,shared_load_throughput,shared_store_throughput,global_hit_rate,gld_efficiency,gld_requested_throughput,global_load_requests,shared_efficiency,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_dp_mul,sysmem_read_bytes,sysmem_read_throughput,sysmem_read_transactions,sysmem_read_utilization,tex_cache_throughput,tex_cache_transactions,texture_load_requests,nc_cache_global_hit_rate,l2_l1_read_hit_rate,l1_cache_local_hit_rate,l1_cache_global_hit_rate,l2_tex_read_hit_rate,local_hit_rate,l1_cache_hit_rate,tex_cache_hit_rate,sm_efficiency,warp_execution_efficiency,branch_efficiency,stall_inst_fetch,stall_memory_dependency,stall_exec_dependency,stall_memory_throttle,stall_pipe_busy,stall_not_selected,flop_dp_efficiency,gst_efficiency,global_atomic_requests'
    nv_prof+= ' ' + exe + ' |& tee ' + testfile +nvprof_name
    print(nv_prof)

    subprocess.call(nv_prof, shell=True)

import numpy as np
import subprocess
import time
import os
import socket
import re


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

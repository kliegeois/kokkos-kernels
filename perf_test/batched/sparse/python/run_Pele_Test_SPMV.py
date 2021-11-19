import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import run_test
from create_matrices import *
from read_pele_matrices import *
import os


def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return 2*(nnz+nrows)*number_of_matrices*bytes_per_entry


def main():
    tic = time.perf_counter()
    Ns = np.arange(1000,1501, 10)

    input_folder = 'pele_data/jac-dodecane_lu-typvals/'
    n_files = 78

    with open('binary_dir.txt') as f:
        directory = f.read()

    data_d = 'Pele_SPMV_data_1'

    rows_per_thread=2
    team_size=8
    implementations_left = [0, 1, 2, 3]
    implementations_right = [0, 1, 2, 3]
    n_implementations_left = len(implementations_left)
    n_implementations_right = len(implementations_right)

    n1 = 5
    n2 = 10

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
        r, c, V, n = read_matrices(input_folder, n_files, Ns[i])
        nnzs[i] = len(r)
        n_ops = compute_n_ops(n, nnzs[i], Ns[i])

        B = create_Vector(n, Ns[i])
    
        mmwrite(name_A, V, r, c, n, n)
        mmwrite(name_B, B)

        data = run_test(directory+'/KokkosBatched_Test_SPMV', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_left, layout='Left')
        for j in range(0, n_implementations_left):
            CPU_time_left[j,i,:] = data[j,:]
        throughput_left[:,i,:] = n_ops/CPU_time_left[:,i,:]
        data = run_test(directory+'/KokkosBatched_Test_SPMV', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_right, layout='Right')
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
    print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

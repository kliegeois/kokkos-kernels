import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import run_test
from create_matrices import *
import os


def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return 2*(nnz+nrows)*number_of_matrices*bytes_per_entry


def verify_SPMV(name_A, name_B, name_X, implementation, layout):
    
    B = mmread(name_B)
    N = np.shape(B)[1]

    if layout == 'Left':
        X = mmread(name_X+str(implementation)+'_l.mm')
    if layout == 'Right':
        X = mmread(name_X+str(implementation)+'_r.mm')
        
    for i in range(0, N):
        b = B[:, i]
        A = mmread(name_A, i)

        x = X[:, i]

        res = np.linalg.norm(A.dot(b.transpose())-x.transpose())
        if res > 1e-10:
            print('res = '+str(res))
            return False
    return True


def main():
    tic = time.perf_counter()
    Bs = np.arange(10,501, 10)

    with open('binary_dir.txt') as f:
        directory = f.read()

    rows_per_thread=1
    team_size=8
    N = 1600*team_size*rows_per_thread
    implementations_left = [0, 1, 2, 3]
    implementations_right = [0, 1, 2, 3]
    n_implementations_left = len(implementations_left)
    n_implementations_right = len(implementations_right)

    n1 = 10
    n2 = 100

    n_quantiles = 7

    CPU_time_left = np.zeros((n_implementations_left, len(Bs), n_quantiles))
    throughput_left = np.zeros((n_implementations_left, len(Bs), n_quantiles))
    CPU_time_right = np.zeros((n_implementations_right, len(Bs), n_quantiles))
    throughput_right = np.zeros((n_implementations_right, len(Bs), n_quantiles))
    nnzs = np.zeros((len(Bs), ))

    if not os.path.isdir('data'):
        os.mkdir('data')

    name_A = 'data/A.mm'
    name_B = 'data/B.mm'
    name_X = 'data/X'
    name_timers = 'data/timers'

    max_offset = 3
    offset = 4
    verify = False

    for i in range(0, len(Bs)):
        r, c = create_strided_graph(Bs[i], max_offset, offset)
        nnzs[i] = len(r)
        n_ops = compute_n_ops(Bs[i], nnzs[i], N)

        V = create_SPD(Bs[i], r, c, N)
        B = create_Vector(Bs[i], N)
    
        mmwrite(name_A, V, r, c, Bs[i], Bs[i])
        mmwrite(name_B, B)

        data = run_test(directory+'/KokkosBatched_Test_SPMV', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_left, layout='Left')
        for j in range(0, n_implementations_left):
            if verify:
                res = verify_SPMV(name_A, name_B, name_X, implementations_left[j], 'Left')
                if res == False:
                    data[j,:] *= 0
            CPU_time_left[j,i,:] = data[j,:]
        throughput_left[:,i,:] = n_ops/CPU_time_left[:,i,:]
        data = run_test(directory+'/KokkosBatched_Test_SPMV', name_A, name_B, name_X, name_timers, rows_per_thread, team_size, n1=n1, n2=n2, implementations=implementations_right, layout='Right')
        for j in range(0, n_implementations_right):
            if verify:
                res = verify_SPMV(name_A, name_B, name_X, implementations_right[j], 'Right')
                if res == False:
                    data[j,:] *= 0
            CPU_time_right[j,i,:] = data[j,:]
        throughput_right[:,i,:] = n_ops/CPU_time_right[:,i,:]

        for j in range(0, n_implementations_left):
            np.savetxt('data/CPU_time_'+str(implementations_left[j])+'_l.txt', CPU_time_left[j,:,:])
            np.savetxt('data/throughput_'+str(implementations_left[j])+'_l.txt', throughput_left[j,:,:])
        for j in range(0, n_implementations_right):
            np.savetxt('data/CPU_time_'+str(implementations_right[j])+'_r.txt', CPU_time_right[j,:,:])
            np.savetxt('data/throughput_'+str(implementations_right[j])+'_r.txt', throughput_right[j,:,:])
        np.savetxt('data/Bs.txt', Bs)
        np.savetxt('data/nnzs.txt', nnzs)

    toc = time.perf_counter()
    print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

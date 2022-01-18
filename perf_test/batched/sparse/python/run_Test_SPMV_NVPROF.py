import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import run_test_nvprof
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
    B = 54
    Ns = np.arange(5100, 5300, 50)

    with open('binary_dir.txt') as f:
        directory = f.read()

    data_d = 'SPMV_data_1'

    rows_per_thread=1
    team_size=32
    implementation = 3

    n1 = 2
    n2 = 10

    nnzs = np.zeros((len(Ns), ))

    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    name_A = data_d+'/A.mm'
    name_B = data_d+'/B.mm'
    name_X = data_d+'/X'
    name_timers = data_d+'/timers'

    max_offset = 3
    offset = 4
    verify = False

    for i in range(0, len(Ns)):
        r, c = create_strided_graph(B, max_offset, offset)
        nnzs[i] = len(r)
        n_ops = compute_n_ops(B, nnzs[i], Ns[i])

        V = create_SPD(B, r, c, Ns[i])
        Bv = create_Vector(B, Ns[i])
    
        mmwrite(name_A, V, r, c, B, B)
        mmwrite(name_B, Bv)

        testfile = data_d+'/nvprof.'+str(implementation)+'.'+str(B)+'.'+str(nnzs[i])+'.'+str(Ns[i])

        nvprof_exe = '/home/projects/ppc64le-pwr9-nvidia/cuda/10.2.2/bin/nvprof'
        run_test_nvprof(nvprof_exe, directory+'/KokkosBatched_Test_SPMV', testfile, name_A, name_B, name_X, rows_per_thread, team_size, n1=n1, n2=n2, implementation=implementation, layout='Left', extra_args=' -P')
        run_test_nvprof(nvprof_exe, directory+'/KokkosBatched_Test_SPMV', testfile, name_A, name_B, name_X, rows_per_thread, team_size, n1=n1, n2=n2, implementation=implementation, layout='Right', extra_args=' -P')


    toc = time.perf_counter()
    print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

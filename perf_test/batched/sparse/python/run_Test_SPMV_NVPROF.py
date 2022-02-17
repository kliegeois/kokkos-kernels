import numpy as np

import time
from test_io import mmwrite, mmread
from run_Test import run_test_nvprof, getBuildDirectory
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

    directory = getBuildDirectory()

    data_d = 'SPMV_NVPROF_data_1'

    rows_per_thread=1
    team_size=32
    N = 1600*team_size
    rows_per_thread=1
    team_size = 16
    vector_length = 16
    N_team = 16
    implementation = 3

    n1 = 2
    n2 = 10

    nnzs = np.zeros((len(Bs), ))

    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    name_A = data_d+'/A.mm'
    name_B = data_d+'/B.mm'
    name_X = data_d+'/X'
    name_timers = data_d+'/timers'

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

        testfile = data_d+'/nvprof.'+str(implementation)+'.'+str(Bs[i])+'.'+str(nnzs[i])+'.'+str(N)

        nvprof_exe = '/home/projects/ppc64le-pwr9-nvidia/cuda/10.2.2/bin/nvprof'
        run_test_nvprof(nvprof_exe, directory+'/KokkosBatched_Test_SPMV', testfile, name_A, name_B, name_X, rows_per_thread, team_size, n1=n1, n2=n2, implementation=implementation, layout='Left', extra_args=' -P -vector_length '+str(vector_length)+ ' -N_team '+str(N_team))
        run_test_nvprof(nvprof_exe, directory+'/KokkosBatched_Test_SPMV', testfile, name_A, name_B, name_X, rows_per_thread, team_size, n1=n1, n2=n2, implementation=implementation, layout='Right', extra_args=' -P -vector_length '+str(vector_length)+ ' -N_team '+str(N_team))


    toc = time.perf_counter()
    print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

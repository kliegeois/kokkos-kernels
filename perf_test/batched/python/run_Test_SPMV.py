import numpy as np
import subprocess

n_quantiles = 7

def run_test_spmv(N=128, B=200, nnz_per_row=5, n=10000, rows_per_thread=1, team_size=8, n_implementation=4, verify=True):
    for i in range(0, n_implementation):
        subprocess.call('./KokkosBatched_Test_SPMV -N ' +str(N)+ ' -B '+str(B)+' -nnz_per_row '+ str(nnz_per_row)+' -n '+ str(n)+' -rows_per_thread '+str(rows_per_thread)+' -team_size '+str(team_size)+ ' -implementation '+str(i), shell=True)
    if verify:
        y_0 = np.loadtxt('y_0.txt')
        tol = 1e-5
    data = np.zeros((n_implementation, n_quantiles))
    for i in range(0, n_implementation):
        data[i, :] = np.loadtxt('timer_'+str(i)+'.txt')
        if verify and i > 0:
            y_i = np.loadtxt('y_'+str(i)+'.txt')
            if len(y_i) is not len(y_0):
                data[i, :] = -1
            else:
                error = np.amax(np.abs(y_0-y_i))
                if error > tol:
                    data[i, :] = -1
    return data


def compute_n_ops(nrows, nnz_per_row, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return 2*nrows*(nnz_per_row+1)*number_of_matrices*bytes_per_entry


def main():
    N = 5120
    Bs = np.arange(50,300)
    nnz_per_row=10
    n=20000
    rows_per_thread=1
    team_size=8
    n_implementation=4

    CPU_time = np.zeros((n_implementation, len(Bs), n_quantiles))
    throughput = np.zeros((n_implementation, len(Bs), n_quantiles))
    for i in range(0, len(Bs)):
        n_ops = compute_n_ops(Bs[i], nnz_per_row, N)
        data = run_test_spmv(N, Bs[i], nnz_per_row, n, rows_per_thread, team_size, n_implementation)
        for j in range(0, n_implementation):
            CPU_time[j,i,:] = data[j,:]
        throughput[:,i,:] = n_ops/CPU_time[:,i,:]
    
    for j in range(0, n_implementation):
        np.savetxt('CPU_time_'+str(j)+'.txt', CPU_time[j,:,:])
        np.savetxt('throughput_'+str(j)+'.txt', throughput[j,:,:])
    np.savetxt('Bs.txt', Bs)


if __name__ == "__main__":
    main()

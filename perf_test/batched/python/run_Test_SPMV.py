import numpy as np
import subprocess
import time

n_quantiles = 7

def run_test_spmv(N=128, B=200, nnz_per_row=5, n=10000, rows_per_thread=1, team_size=8, n_implementations=4, layout='Left', verify=True):
    exe = './KokkosBatched_Test_SPMV -N ' +str(N)+ ' -B '+str(B)+' -nnz_per_row '+ str(nnz_per_row)+' -n1 10 -n2 '+ str(n)+' -rows_per_thread '+str(rows_per_thread)+' -team_size '+str(team_size)+ ' -n_implementations '+str(n_implementations)
    if layout == 'Left':
        exe += ' -l'
        y_name = '_l.txt'
        time_name = '_left.txt'
    if layout == 'Right':
        exe += ' -r'
        y_name = '_r.txt'
        time_name = '_right.txt'
    subprocess.call(exe, shell=True)
    if verify:
        y_0 = np.loadtxt('y_0'+y_name)
        tol = 1e-5
    data = np.zeros((n_implementations, n_quantiles))
    for i in range(0, n_implementations):
        data[i, :] = np.loadtxt('timer_'+str(i)+time_name)
        if verify and i > 0:
            y_i = np.loadtxt('y_'+str(i)+y_name)
            if len(y_i) != len(y_0):
                print("Strange, one is of length "+str(len(y_i))+ " and the other "+str(len(y_0)) +" size = " + str(len(y_i)) + " i = " + str(i))
                #data[i, :] = -1
            else:
                error = np.amax(np.abs(y_0-y_i))
                if error > tol:
                    print("Strange, error = "+str(error)+" size = " + str(len(y_i)) + " i = " + str(i))
                    #data[i, :] = -1
    return data


def compute_n_ops(nrows, nnz_per_row, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return 2*nrows*(nnz_per_row+1)*number_of_matrices*bytes_per_entry


def main():
    tic = time.perf_counter()
    N = 12800
    Bs = np.arange(50,200)
    nnz_per_row=10
    n=100
    rows_per_thread=1
    team_size=8
    n_implementations=15

    CPU_time_left = np.zeros((n_implementations, len(Bs), n_quantiles))
    throughput_left = np.zeros((n_implementations, len(Bs), n_quantiles))
    CPU_time_right = np.zeros((n_implementations, len(Bs), n_quantiles))
    throughput_right = np.zeros((n_implementations, len(Bs), n_quantiles))
    for i in range(0, len(Bs)):
        n_ops = compute_n_ops(Bs[i], nnz_per_row, N)
        data = run_test_spmv(N, Bs[i], nnz_per_row, n, rows_per_thread, team_size, n_implementations, layout='Left')
        for j in range(0, n_implementations):
            CPU_time_left[j,i,:] = data[j,:]
        throughput_left[:,i,:] = n_ops/CPU_time_left[:,i,:]
        data = run_test_spmv(N, Bs[i], nnz_per_row, n, rows_per_thread, team_size, n_implementations, layout='Right')
        for j in range(0, n_implementations):
            CPU_time_right[j,i,:] = data[j,:]
        throughput_right[:,i,:] = n_ops/CPU_time_right[:,i,:]

        for j in range(0, n_implementations):
            np.savetxt('CPU_time_'+str(j)+'_l.txt', CPU_time_left[j,:,:])
            np.savetxt('throughput_'+str(j)+'_l.txt', throughput_left[j,:,:])
            np.savetxt('CPU_time_'+str(j)+'_r.txt', CPU_time_right[j,:,:])
            np.savetxt('throughput_'+str(j)+'_r.txt', throughput_right[j,:,:])
        np.savetxt('Bs.txt', Bs)

    toc = time.perf_counter()
    print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

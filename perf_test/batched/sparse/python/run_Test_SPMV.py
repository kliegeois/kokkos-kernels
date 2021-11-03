import numpy as np
import subprocess
import time
import os

n_quantiles = 7

def run_test_spmv(N=128, B=200, nnz_per_row=5, n=10000, rows_per_thread=1, team_size=8, implementations=[0], layout='Left', verify=True):
    n_implementations = len(implementations)
    exe = './KokkosBatched_Test_SPMV -N ' +str(N)+ ' -B '+str(B)+' -nnz_per_row '+ str(nnz_per_row)+' -n1 10 -n2 '+ str(n)+' -rows_per_thread '+str(rows_per_thread)+' -team_size '+str(team_size)
    for implementation in implementations:
        exe = exe + ' -implementation '+str(implementation)
        if layout == 'Left':
            exe += ' -l'
            y_name = '_l.txt'
            time_name = '_left.txt'
        if layout == 'Right':
            exe += ' -r'
            y_name = '_r.txt'
            time_name = '_right.txt'
    for i in range(0, n_implementations):
        current_name = 'y_'+str(implementations[i])+y_name
        if os.path.exists(current_name):
            os.remove(current_name)
    subprocess.call(exe, shell=True)
    if verify:
        y_0 = np.loadtxt('y_0'+y_name)
        tol = 1e-5
    data = np.zeros((n_implementations, n_quantiles))
    nnz = np.loadtxt('nnz.txt')
    for i in range(0, n_implementations):
        data[i, :] = np.loadtxt('timer_'+str(implementations[i])+time_name)
        if verify and i > 0:
            y_i = np.loadtxt('y_'+str(implementations[i])+y_name)
            if np.linalg.norm(y_i) == 0.:
                print("Strange, the norm is zero for i = " + str(i))
                raise NameError("Wrong y")
            if len(y_i) != len(y_0):
                print("Strange, one is of length "+str(len(y_i))+ " and the other "+str(len(y_0)) +" size = " + str(len(y_i)) + " i = " + str(i))
                raise NameError("Wrong y")
            else:
                error = np.amax(np.abs(y_0-y_i))
                if error > tol:
                    print("Strange, error = "+str(error)+" size = " + str(len(y_i)) + " i = " + str(i))
                    raise NameError("Wrong y")
    return data, nnz


def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    return (2*nnz+2*nrows)*number_of_matrices*bytes_per_entry


def main():
    tic = time.perf_counter()
    Bs = np.arange(10,501, 10)
    nnz_per_row=10
    n=100
    rows_per_thread=1
    team_size=32
    N = team_size*1600
    implementations_left = [0, 1, 2, 3]
    implementations_right = [0, 1, 2, 3]
    n_implementations_left = len(implementations_left)
    n_implementations_right = len(implementations_right)

    CPU_time_left = np.zeros((n_implementations_left, len(Bs), n_quantiles))
    throughput_left = np.zeros((n_implementations_left, len(Bs), n_quantiles))
    CPU_time_right = np.zeros((n_implementations_right, len(Bs), n_quantiles))
    throughput_right = np.zeros((n_implementations_right, len(Bs), n_quantiles))
    nnzs = np.zeros((len(Bs), ))
    for i in range(0, len(Bs)):
        data, nnz = run_test_spmv(N, Bs[i], nnz_per_row, n, rows_per_thread, team_size, implementations_left, layout='Left')
        nnzs[i] = nnz
        n_ops = compute_n_ops(Bs[i], nnz, N)
        for j in range(0, n_implementations_left):
            CPU_time_left[j,i,:] = data[j,:]
        throughput_left[:,i,:] = n_ops/CPU_time_left[:,i,:]
        data, nnz = run_test_spmv(N, Bs[i], nnz_per_row, n, rows_per_thread, team_size, implementations_right, layout='Right')
        for j in range(0, n_implementations_right):
            CPU_time_right[j,i,:] = data[j,:]
        throughput_right[:,i,:] = n_ops/CPU_time_right[:,i,:]

        for j in range(0, n_implementations_left):
            np.savetxt('CPU_time_'+str(implementations_left[j])+'_l.txt', CPU_time_left[j,:,:])
            np.savetxt('throughput_'+str(implementations_left[j])+'_l.txt', throughput_left[j,:,:])
        for j in range(0, n_implementations_right):
            np.savetxt('CPU_time_'+str(implementations_right[j])+'_r.txt', CPU_time_right[j,:,:])
            np.savetxt('throughput_'+str(implementations_right[j])+'_r.txt', throughput_right[j,:,:])
        np.savetxt('Bs.txt', Bs)
        np.savetxt('nnzs.txt', nnzs)

    toc = time.perf_counter()
    print(f"Elapsed time {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()

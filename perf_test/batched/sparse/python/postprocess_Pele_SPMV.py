import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tikzplotlib
from test_io import mmread

def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8, unit='B/sec'):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    nops = 2*(nnz+nrows)*number_of_matrices
    if unit=='B/sec':
        return nops*bytes_per_entry
    if unit=='GFLOPS':
        return nops/1e9


def plot_limits(Bs, ax, nnz_per_row, N, memory_limits=True, peak_limits=False, n_GPUs=1, throughput=True, unit='B/sec'):

    # V100 GPU up to 900 GB/sec = 9e11
    bandwidth = 9e11 * n_GPUs
    optimistic = bandwidth * 2/12
    pessimistic = bandwidth * 2/20
    reused_optimistic = bandwidth * 2/8
    # V100 GPU up to 7.8 TFLOPS = 7.8e12
    peak_boost = 7.8e12 * n_GPUs
    peak_base = 6.1e12 * n_GPUs
    if unit == 'B/sec':
        optimistic *= 8
        pessimistic *= 8
        reused_optimistic *= 8
        peak_boost *= 8
        peak_base *= 8
    if unit == 'GFLOPS':
        optimistic /= 1e9
        pessimistic /= 1e9
        reused_optimistic /= 1e9
        peak_boost /= 1e9
        peak_base /= 1e9
    if memory_limits:
        if throughput:
            ax.plot([Bs[0], Bs[-1]], [reused_optimistic, reused_optimistic], '--', label='Reused optimistic memory bandwidth bound')
            ax.plot([Bs[0], Bs[-1]], [optimistic, optimistic], '--', label='Optimistic memory bandwidth bound')
            ax.plot([Bs[0], Bs[-1]], [pessimistic, pessimistic], '--', label='Pessimistic memory bandwidth bound')
        else:
            reused_optimistic_scale = (nnz_per_row+1)*N*8*8
            optimistic_scale = (nnz_per_row+1)*N*12*8
            pessimistic_scale = (nnz_per_row+1)*N*20*8
            if unit == 'B/sec':
                reused_optimistic_scale /= 8
                optimistic_scale /= 8
                pessimistic_scale /= 8
            ax.plot([Bs[0], Bs[-1]], [(Bs[0]*reused_optimistic_scale)/bandwidth, (Bs[-1]*reused_optimistic_scale)/bandwidth], '--', label='Reused optimistic memory bandwidth bound')
            ax.plot([Bs[0], Bs[-1]], [(Bs[0]*optimistic_scale)/bandwidth, (Bs[-1]*optimistic_scale)/bandwidth], '--', label='Optimistic memory bandwidth bound')
            ax.plot([Bs[0], Bs[-1]], [(Bs[0]*pessimistic_scale)/bandwidth, (Bs[-1]*pessimistic_scale)/bandwidth], '--', label='Pessimistic memory bandwidth bound')
    if peak_limits:
        if throughput:
            ax.plot([Bs[0], Bs[-1]], [peak_boost, peak_boost], '--')
            ax.plot([Bs[0], Bs[-1]], [peak_base, peak_base], '--')


def plot_quantiles(x, ys, ax, alpha=0.2, i_skip=0, label='None', dashed=False, plot_fill=False):
    n_quantiles = np.shape(ys)[1]
    i_median = 0 #int(np.floor(n_quantiles/2))
    if dashed:
        line = ax.plot(x, ys[:, i_median], '--')
    else:
        line = ax.plot(x, ys[:, i_median])
    if label != 'None':
        line[0].set_label(label)
    if plot_fill:
        for i in range(i_skip, i_median):
            ax.fill_between(x, ys[:, i], ys[:, n_quantiles-i-1],
                            color=line[0].get_color(), alpha=alpha)

def ginkgo_data(specie):
    if specie == 'gri30':
        n = np.array([90,  1440,  2880,  5760, 11520, 17280])
        time = np.array([5.231000e-05, 1.643590e-04, 3.038220e-04, 5.694870e-04, 1.073582e-03, 1.574097e-03])

        n = 90*np.array([  1,  16,  32, 128, 192])
        time = np.array([0.00010079, 0.00038695, 0.00071774, 0.0029122 , 0.00437534])
    if specie == 'isooctane':
        n = np.array([72,  1152,  2304,  4608,  9216, 13824])
        time = np.array([0.00010629, 0.00040689, 0.00069134, 0.00127151, 0.0024188,  0.00382598])

        n = 72*np.array([  1,  16,  32, 128, 192])
        time = np.array([0.00075267, 0.00531965, 0.01029381, 0.03522193, 0.05265616])
    return n, time

hosts = ['blake', 'caraway', 'weaver']
species = ['gri30', 'isooctane']
unit = 'GFLOPS'

for host in hosts:
    for specie in species:

        base = host+'/Pele_SPMV_'+specie+'_data_SPMV_vec_tuned/'
        if host == 'caraway':
            implementations = [3]
        else:
            implementations = [3]
        n_implementations = len(implementations)
        n_quantiles = 7

        team_params = np.loadtxt(base+'team_params.txt')
        n_ginkgo, time_gingko = ginkgo_data(specie)

        Ns = np.loadtxt(base+'Ns.txt')

        n_ops = np.zeros((len(Ns),))

        if specie == 'gri30':
            nrows = 54
            nnz = 2560
        if specie == 'isooctane':
            nrows = 144
            nnz = 6135

        for i in range(len(n_ops)):
            n_ops[i] = compute_n_ops(nrows, nnz, Ns[i], bytes_per_entry=8, unit=unit)

        CPU_time_l = np.zeros((n_implementations, len(Ns), n_quantiles))
        CPU_time_r = np.zeros((n_implementations, len(Ns), n_quantiles))
        throughput_l = np.zeros((n_implementations, len(Ns), n_quantiles))
        throughput_r = np.zeros((n_implementations, len(Ns), n_quantiles))

        for i in range(0, n_implementations):
            CPU_time_l[i,:,:] = np.loadtxt(base+'CPU_time_'+str(implementations[i])+'_l.txt')
            CPU_time_r[i,:,:] = np.loadtxt(base+'CPU_time_'+str(implementations[i])+'_r.txt')
            throughput_l[i,:,:] = np.loadtxt(base+'throughput_'+str(implementations[i])+'_l.txt')
            throughput_r[i,:,:] = np.loadtxt(base+'throughput_'+str(implementations[i])+'_r.txt')
            for j in range(len(n_ops)):
                throughput_l[i,j,:] = n_ops[j]/CPU_time_l[i,j,:]
                throughput_r[i,j,:] = n_ops[j]/CPU_time_r[i,j,:]
            np.savetxt(base+'throughput_'+str(implementations[i])+'_l_updated.txt', throughput_l[i,:,:])
            np.savetxt(base+'throughput_'+str(implementations[i])+'_r_updated.txt', throughput_r[i,:,:])

        nnz_per_row = 10
        N = 12800

        fig = plt.figure()
        ax = plt.gca()
        for i in range(0, n_implementations):
            if i < 5:
                plot_quantiles(Ns, CPU_time_l[i,:,:], ax, i_skip=0, label='Impl '+str(i) + ' left')
                plot_quantiles(Ns, CPU_time_r[i,:,:], ax, i_skip=0, label='Impl '+str(i) + ' right')
            else:
                plot_quantiles(Ns, CPU_time_l[i,:,:], ax, i_skip=0, label='Impl '+str(i-5) + ' left with cache', dashed=True)
                plot_quantiles(Ns, CPU_time_r[i,:,:], ax, i_skip=0, label='Impl '+str(i-5) + ' right with cache', dashed=True)

        plt.grid()
        #ax.set_ylim(0., 0.006)
        #plot_limits(Ns, ax, nnz_per_row, N, throughput=False)

        ax.set_xlabel('Number of matrices')
        ax.set_ylabel('Wall-clock time [sec]')
        plt.title('team_size = '+str(team_params[0])+', vector_length = '+str(team_params[1])+ ', N_team = '+str(team_params[2]))
        #ax.set_ylim(0, 0.008)
        #ax.set_xlim(1000, 1500)

        legend = ax.legend(loc='best', shadow=True)

        plt.savefig(base+'wall-clock time.png')
        tikzplotlib.save(base+'wall-clock time.tex')

        fig = plt.figure()
        ax = plt.gca()
        for i in range(0, n_implementations):
            if i < 5:
                plot_quantiles(Ns, throughput_l[i,:,:], ax, i_skip=0, label='Impl '+str(i) + ' left')
                plot_quantiles(Ns, throughput_r[i,:,:], ax, i_skip=0, label='Impl '+str(i) + ' right')
            elif i < 10:
                plot_quantiles(Ns, throughput_l[i,:,:], ax, i_skip=0, label='Impl '+str(i-5) + ' left with cache', dashed=True)
                plot_quantiles(Ns, throughput_r[i,:,:], ax, i_skip=0, label='Impl '+str(i-5) + ' right with cache', dashed=True)
            else:
                plot_quantiles(Ns, throughput_l[i,:,:], ax, i_skip=0, label='Impl '+str(i-5) + ' left with global cache', dashed=True)
                plot_quantiles(Ns, throughput_r[i,:,:], ax, i_skip=0, label='Impl '+str(i-5) + ' right with global cache', dashed=True)
            max_l_throughput = np.amax(throughput_l[i,:,:])/1e12
            max_r_throughput = np.amax(throughput_r[i,:,:])/1e12
            if max_l_throughput > 0.8:
                print('Impl '+str(i) + ' left max throughput = '+str(max_l_throughput))
            if max_r_throughput > 0.8:
                print('Impl '+str(i) + ' right max throughput = '+str(max_r_throughput))

        if host == 'weaver':
            plot_limits(Ns, ax, nnz_per_row, N, unit=unit)
        ax.set_xlabel('Number of matrices')
        ax.set_ylabel('Throughput ['+unit+']')
        plt.title('team_size = '+str(team_params[0])+', vector_length = '+str(team_params[1])+ ', N_team = '+str(team_params[2]))
        #ax.set_xlim(1000, 1500)

        legend = ax.legend(loc='best', shadow=True)

        '''
        for i in range(56, 256, 8):
            ax.plot([i,i], [0,3e11], 'k--')
        '''
        plt.savefig(base+'throughput.png')

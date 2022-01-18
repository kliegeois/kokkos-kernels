import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tikzplotlib


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
    i_median = int(np.floor(n_quantiles/2))
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
    if specie == 'isooctane':
        n = np.array([72,  1152,  2304,  4608,  9216, 13824])
        time = np.array([0.00010629, 0.00040689, 0.00069134, 0.00127151, 0.0024188,  0.00382598])
    return n, time

specie = 'gri30'

data_base = 'Pele_pGMRES_'+specie+'_data_all/'
implementations = [3]
n_implementations = len(implementations)
n_quantiles = 7

n_ginkgo, time_gingko = ginkgo_data(specie)

team_sizes = np.array([1, 2, 4, 8, 16, 32])
rows_per_threads = np.array([1, 2, 4, 8, 16, 32, 64])

CPU_l = np.zeros((len(team_sizes), len(rows_per_threads), 6))
CPU_r = np.zeros((len(team_sizes), len(rows_per_threads), 6))

for i_team in range(0, len(team_sizes)):
    for i_per_thread in range(0, len(rows_per_threads)):
        base = data_base + '/' + str(team_sizes[i_team]) + '_' + str(rows_per_threads[i_per_thread])+'/'
        
        Ns = np.loadtxt(base+'Ns.txt')
        CPU_time_l = np.zeros((n_implementations, len(Ns), n_quantiles))
        CPU_time_r = np.zeros((n_implementations, len(Ns), n_quantiles))

        for i in range(0, n_implementations):
            CPU_time_l[i,:,:] = np.loadtxt(base+'CPU_time_'+str(implementations[i])+'_l.txt')
            CPU_time_r[i,:,:] = np.loadtxt(base+'CPU_time_'+str(implementations[i])+'_r.txt')

        CPU_l[i_team, i_per_thread, :] = CPU_time_l[0,:,3]
        CPU_r[i_team, i_per_thread, :] = CPU_time_r[0,:,3]

print(np.amin(CPU_l[:,:,-1]))
print(np.amin(CPU_r[:,:,-1]))


result_l = np.where(CPU_l[:,:,-1] == np.amin(CPU_l[:,:,-1]))
result_r = np.where(CPU_r[:,:,-1] == np.amin(CPU_r[:,:,-1]))

print(result_l)
print(result_r)


fig = plt.figure()
ax = plt.gca()
CS = plt.contourf(CPU_l[:,:,-1])
plt.plot(result_l[1][0],result_l[0][0], 'g*')
cbar = fig.colorbar(CS)
ax.set_xticklabels(rows_per_threads)
ax.set_yticklabels(team_sizes)
plt.savefig(data_base+'CPU_l.png')

fig = plt.figure()
ax = plt.gca()
CS = plt.contourf(CPU_r[:,:,-1])
plt.plot(result_r[1][0],result_r[0][0], 'g*')
cbar = fig.colorbar(CS)
ax.set_xticklabels(rows_per_threads)
ax.set_yticklabels(team_sizes)
plt.savefig(data_base+'CPU_r.png')

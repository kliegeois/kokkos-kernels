import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_limits(Bs, ax, nnz_per_row, N, memory_limits=True, peak_limits=False, n_GPUs=1, throughput=True):

    # V100 GPU up to 900 GB/sec = 9e11
    bandwidth = 9e11 * n_GPUs
    optimistic = bandwidth * 2/12
    pessimistic = bandwidth * 2/20

    # V100 GPU up to 7.8 TFLOPS = 7.8e12
    peak_boost = 7.8e12 * n_GPUs
    peak_base = 6.1e12 * n_GPUs
    if memory_limits:
        if throughput:
            ax.plot([Bs[0], Bs[-1]], [optimistic, optimistic], label='Optimistic memory bandwidth bound')
            ax.plot([Bs[0], Bs[-1]], [pessimistic, pessimistic], label='Pessimistic memory bandwidth bound')
        else:
            optimistic_scale = (nnz_per_row+1)*N*12*8
            pessimistic_scale = (nnz_per_row+1)*N*20*8
            ax.plot([Bs[0], Bs[-1]], [(Bs[0]*optimistic_scale)/bandwidth, (Bs[-1]*optimistic_scale)/bandwidth], label='Optimistic memory bandwidth bound')
            ax.plot([Bs[0], Bs[-1]], [(Bs[0]*pessimistic_scale)/bandwidth, (Bs[-1]*pessimistic_scale)/bandwidth], label='Pessimistic memory bandwidth bound')
    if peak_limits:
        if throughput:
            ax.plot([Bs[0], Bs[-1]], [peak_boost, peak_boost])
            ax.plot([Bs[0], Bs[-1]], [peak_base, peak_base])


def plot_quantiles(x, ys, ax, alpha=0.2, i_skip=0, label='None'):
    n_quantiles = np.shape(ys)[1]
    i_median = int(np.floor(n_quantiles/2))
    line = ax.plot(x, ys[:, i_median])
    if label != 'None':
        line[0].set_label(label)
    for i in range(i_skip, i_median):
        ax.fill_between(x, ys[:, i], ys[:, n_quantiles-i-1],
                        color=line[0].get_color(), alpha=alpha)


base = 'data_0/'
n_implementations = 3
n_quantiles = 7

Bs = np.loadtxt(base+'Bs.txt')
CPU_time = np.zeros((n_implementations, len(Bs), n_quantiles))
throughput = np.zeros((n_implementations, len(Bs), n_quantiles))

for i in range(0, n_implementations):
    CPU_time[i,:,:] = np.loadtxt(base+'CPU_time_'+str(i)+'.txt')
    throughput[i,:,:] = np.loadtxt(base+'throughput_'+str(i)+'.txt')

nnz_per_row = 10
N = 12800

fig = plt.figure()
ax = plt.gca()
for i in range(0, n_implementations):
    plot_quantiles(Bs, CPU_time[i,:,:], ax, i_skip=0, label='Impl '+str(i))

plot_limits(Bs, ax, nnz_per_row, N, throughput=False)

ax.set_xlabel('Number of rows')
ax.set_ylabel('Wall-clock time [sec]')
ax.set_ylim(0, 0.008)
ax.set_xlim(50, 250)

legend = ax.legend(loc='best', shadow=True)

plt.savefig('wall-clock time.png')

fig = plt.figure()
ax = plt.gca()
for i in range(0, n_implementations):
    plot_quantiles(Bs, throughput[i,:,:], ax, i_skip=0, label='Impl '+str(i))

plot_limits(Bs, ax, nnz_per_row, N)
ax.set_xlabel('Number of rows')
ax.set_ylabel('Throughput [FLOPS]')
ax.set_xlim(50, 250)

legend = ax.legend(loc='best', shadow=True)
plt.savefig('throughput.png')

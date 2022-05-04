import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tikzplotlib
from test_io import mmread
import os
import argparse

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
    i_median = 0#int(np.floor(n_quantiles/2))
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
        time = np.array([0.00035001, 0.00063808, 0.00096812, 0.00317675, 0.00465993])
    if specie == 'isooctane':
        n = np.array([72,  1152,  2304,  4608,  9216, 13824])
        time = np.array([0.00010629, 0.00040689, 0.00069134, 0.00127151, 0.0024188,  0.00382598])

        n = 72*np.array([  1,  16,  32, 128, 192])
        time = np.array([0.00075267, 0.00531965, 0.01029381, 0.03522193, 0.05265616])
    return n, time

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

specie = 'gri30'

parser = argparse.ArgumentParser(description='Postprocess the results.')
parser.add_argument('--path', type=dir_path, metavar='basedir',
                    help='basedir of the results')

args = parser.parse_args()
base = args.path +'/'

implementations = [3]
n_implementations = len(implementations)
n_quantiles = 7

n_ginkgo, time_gingko = ginkgo_data(specie)

Ns = np.loadtxt(base+'Ns.txt')
CPU_time_l = np.zeros((n_implementations, len(Ns), n_quantiles))
CPU_time_r = np.zeros((n_implementations, len(Ns), n_quantiles))
throughput_l = np.zeros((n_implementations, len(Ns), n_quantiles))
throughput_r = np.zeros((n_implementations, len(Ns), n_quantiles))

for i in range(0, n_implementations):
    CPU_time_l[i,:,:] = np.loadtxt(base+'CPU_time_'+str(implementations[i])+'_l.txt')
    CPU_time_r[i,:,:] = np.loadtxt(base+'CPU_time_'+str(implementations[i])+'_r.txt')

n_timers = 20
n_timers_1 = 9

all_percent_cost_l = np.zeros((n_timers, len(Ns)))
all_percent_cost_r = np.zeros((n_timers, len(Ns)))

for i in range(0, len(Ns)):
    percent_cost_l = np.zeros((n_timers,))
    percent_cost_r = np.zeros((n_timers,))
    l_internal_timers = mmread(base+'timers_internal_'+str(int(Ns[i]))+'_3_left.mm')
    r_internal_timers = mmread(base+'timers_internal_'+str(int(Ns[i]))+'_3_right.mm')

    for ii in range(0, l_internal_timers.shape[0]):
        for jj in range(0, l_internal_timers.shape[1]):
            percent_cost_l[ii] += l_internal_timers[ii,jj]
    percent_cost_l[-1] = 0
    for ii in range(0, n_timers_1):
        percent_cost_l[-1] += percent_cost_l[ii]
    for ii in range(0, len(percent_cost_l)):
        percent_cost_l[ii] /= percent_cost_l[-1]
    for ii in range(0, r_internal_timers.shape[0]):
        for jj in range(0, r_internal_timers.shape[1]):
            percent_cost_r[ii] += r_internal_timers[ii,jj]
    percent_cost_r[-1] = 0
    for ii in range(0, n_timers_1):
        percent_cost_r[-1] += percent_cost_r[ii]
    for ii in range(0, len(percent_cost_r)):
        percent_cost_r[ii] /= percent_cost_r[-1]

    all_percent_cost_l[:, i] = percent_cost_l
    all_percent_cost_r[:, i] = percent_cost_r

names = ['init', 'spmv', 'prec', 'norm', 'inner', 'update', 'Givens', 'TRSM', 'reconstruction', 'total']

fig = plt.figure()
ax = plt.gca()
for i in range(0, n_timers_1):
    line = plt.plot(Ns, 100*all_percent_cost_l[i,:])
    line[0].set_label(names[i]+' left')
    plt.grid()    
legend = ax.legend(loc='best', shadow=True)
plt.savefig(base+'left_percent.png')
tikzplotlib.save(base+'left_percent.tex')

fig = plt.figure()
ax = plt.gca()
for i in range(0, n_timers_1):
    line = plt.plot(Ns, 100*all_percent_cost_r[i,:])
    line[0].set_label(names[i]+' right')
    plt.grid()    
legend = ax.legend(loc='best', shadow=True)
plt.savefig(base+'right_percent.png')
tikzplotlib.save(base+'right_percent.tex')

fig = plt.figure()
ax = plt.gca()
for i in range(0, n_timers_1):
    line = plt.plot(Ns, all_percent_cost_l[i,:]*CPU_time_l[0,:,0])
    line[0].set_label(names[i]+' left')
    plt.grid()    
legend = ax.legend(loc='best', shadow=True)
plt.savefig(base+'left_times.png')
tikzplotlib.save(base+'left_times.tex')

fig = plt.figure()
ax = plt.gca()
for i in range(0, n_timers_1):
    line = plt.plot(Ns, all_percent_cost_r[i,:]*CPU_time_r[0,:,0])
    line[0].set_label(names[i]+' right')
    plt.grid()    
legend = ax.legend(loc='best', shadow=True)
plt.savefig(base+'right_times.png')
tikzplotlib.save(base+'right_times.tex')

for i in range(0, n_timers):
    print(all_percent_cost_l[i,:])
    print(all_percent_cost_r[i,:])


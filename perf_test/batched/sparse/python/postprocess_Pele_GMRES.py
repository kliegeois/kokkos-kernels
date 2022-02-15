import string
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tikzplotlib
from test_io import mmread
from run_Test import getHostName
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
        line = ax.plot(x, ys[:, i_median], '*-')
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

        n = 90*np.array([  1, 16, 32, 64, 96, 128, 160, 192, 224, 256])
        time = np.array([0.0001136 , 0.00042409, 0.00078691, 0.00150753, 0.00222447, 0.0029204 , 0.00365782, 0.00438156, 0.00508972, 0.00581532])
        #time = np.array([0.00035001, 0.00063808, 0.00096812, 0.00317675, 0.00465993])
    if specie == 'isooctane':
        n = np.array([72,  1152,  2304,  4608,  9216, 13824])
        time = np.array([0.00010629, 0.00040689, 0.00069134, 0.00127151, 0.0024188,  0.00382598])

        n = 72*np.array([  1, 16, 32, 64, 96, 128, 160, 192, 224, 256])
        time = np.array([0.00075706, 0.00531739, 0.00953753, 0.01777734, 0.02652215, 0.03524549, 0.04393908, 0.05258815, 0.06129656, 0.06998952])
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

hostname = getHostName()

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
    throughput_l[i,:,:] = np.loadtxt(base+'throughput_'+str(implementations[i])+'_l.txt')
    throughput_r[i,:,:] = np.loadtxt(base+'throughput_'+str(implementations[i])+'_r.txt')

nnz_per_row = 10
N = 12800

fig = plt.figure()
ax = plt.gca()
for i in range(0, n_implementations):
    if i < 5:
        plot_quantiles(Ns, CPU_time_l[i,:,:], ax, i_skip=0, label='Impl '+str(i) + ' left')
        plot_quantiles(Ns, CPU_time_r[i,:,:], ax, i_skip=0, label='Impl '+str(i) + ' right')
        m,b = np.polyfit(Ns, CPU_time_l[i,:,0], 1)
        print(str(i)+' left '+str(m)+ ' '+str(b))
        m,b = np.polyfit(Ns, CPU_time_r[i,:,0], 1)
        print(str(i)+' right '+str(m)+ ' '+str(b))
    else:
        plot_quantiles(Ns, CPU_time_l[i,:,:], ax, i_skip=0, label='Impl '+str(i-5) + ' left with cache', dashed=True)
        plot_quantiles(Ns, CPU_time_r[i,:,:], ax, i_skip=0, label='Impl '+str(i-5) + ' right with cache', dashed=True)

if hostname == 'weaver':
    m,b = np.polyfit(n_ginkgo, time_gingko, 1)
    print('ginkgo '+str(m)+ ' '+str(b))

    line = plt.plot(n_ginkgo, time_gingko, '--')
    line[0].set_label('Ginkgo')
plt.grid()
#ax.set_ylim(0., 0.006)
#plot_limits(Ns, ax, nnz_per_row, N, throughput=False)

ax.set_xlabel('Number of matrices')
ax.set_ylabel('Wall-clock time [sec]')
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

#plot_limits(Ns, ax, nnz_per_row, N)
ax.set_xlabel('Number of matrices')
ax.set_ylabel('Throughput [B/s]')
#ax.set_xlim(1000, 1500)

legend = ax.legend(loc='best', shadow=True)

'''
for i in range(56, 256, 8):
    ax.plot([i,i], [0,3e11], 'k--')
'''
plt.savefig(base+'throughput.png')

left_res_name = base + 'P_res_l3.mm'
right_res_name = base + 'P_res_r3.mm'

left_res = mmread(left_res_name)
right_res = mmread(right_res_name)

N_left = len(left_res[0,:])
N_right = len(right_res[0,:])

min_left_res = np.ones((N_left, ))
min_right_res = np.ones((N_right, ))

iteration_count_left = np.ones((N_left, ))
iteration_count_right = np.ones((N_right, ))

for i in range(0, N_left):
    for j in range(0, len(left_res[:,0])):
        if min_left_res[i] > left_res[j,i] and left_res[j,i] > 0:
            min_left_res[i] = left_res[j,i]
            iteration_count_left[i] = j
for i in range(0, N_right):
    for j in range(0, len(right_res[:,0])):        
        if min_right_res[i] > right_res[j,i] and right_res[j,i] > 0:
            min_right_res[i] = right_res[j,i]
            iteration_count_right[i] = j

iteration_count_ginkgo = np.loadtxt('pele_data/'+specie+'_iterations_count.txt')

#indices = np.argsort(-iteration_count_left[0:len(iteration_count_ginkgo)])
#print(repr(indices))

plt.figure()
plt.subplot(211)
plt.semilogy(min_left_res)
plt.subplot(212)
plt.semilogy(min_right_res)
plt.savefig(base+'res.png')
plt.figure()
plt.subplot(211)
plt.plot(iteration_count_left)
plt.plot(iteration_count_ginkgo)
plt.subplot(212)
plt.plot(iteration_count_right)
plt.savefig(base+'iteration_count.png')
#plt.show()

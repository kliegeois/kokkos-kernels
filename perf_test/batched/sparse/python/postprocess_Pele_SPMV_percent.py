import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tikzplotlib
from test_io import mmread

def get_best_throughput(host, specie):
    if host == 'weaver':
        bw = 831.27
        if specie == 'gri30':
            N_team = 16
        else:
            N_team = 6
        return 2*N_team*bw/(8*N_team+4)
    if host == 'blake':
        bw = 152.86
        N_team = 8
        return 2*N_team*bw/(8*N_team+4)
    if host == 'caraway':
        bw = 816.06
        if specie == 'gri30':
            N_team = 16
        else:
            N_team = 8
        return 2*N_team*bw/(8*N_team+4)
    if host == 'inouye':
        bw = 313.59
        N_team = 8
        return 2*N_team*bw/(8*N_team+4)

def get_pessimistic_throughput(host, specie):
    if host == 'weaver':
        bw = 831.27
        if specie == 'gri30':
            N_team = 16
        else:
            N_team = 6
        return 2*N_team*bw/(16*N_team+4)
    if host == 'blake':
        bw = 152.86
        N_team = 8
        return 2*N_team*bw/(16*N_team+4)
    if host == 'caraway':
        bw = 816.06
        if specie == 'gri30':
            N_team = 16
        else:
            N_team = 8
        return 2*N_team*bw/(16*N_team+4)
    if host == 'inouye':
        bw = 313.59
        N_team = 8
        return 2*N_team*bw/(16*N_team+4)



hosts = ['weaver', 'blake', 'caraway']
species = ['gri30', 'isooctane']
unit = 'GFLOPS'

print_left = True
print_right = False

fig = plt.figure(figsize=(12, 6), dpi=120)
ax = plt.gca()
for host in hosts:
    for specie in species:

        if host == 'blake' or host == 'inouye':
            base = host+'/Pele_SPMV_'+specie+'_data_SPMV_vec/'
        else:
            base = host+'/Pele_SPMV_'+specie+'_data/'
        implementation = 3
        i_quantile = 3

        Ns = np.loadtxt(base+'Ns.txt')
        best_throughput = get_best_throughput(host, specie)
        pessimistic_throughput = get_pessimistic_throughput(host, specie)

        plt.plot(Ns, 100*pessimistic_throughput/best_throughput*np.ones(Ns.shape), '--')

        if print_left:
            throughput_l = np.loadtxt(base+'throughput_'+str(implementation)+'_l_updated.txt')[:,i_quantile]
            plt.plot(Ns, 100*throughput_l/best_throughput, label='Left '+specie+' '+host)

        if print_right:
            throughput_r = np.loadtxt(base+'throughput_'+str(implementation)+'_r_updated.txt')[:,i_quantile]
            plt.plot(Ns, 100*throughput_r/best_throughput, label='Right '+specie+' '+host)
plt.grid()
ax.set_xlabel('Number of matrices')
ax.set_ylabel('Percent of throughput')
legend = ax.legend(loc='best', shadow=True)

ax.set_xlim(0., 30000)
ax.set_ylim(0, 100)

if print_left and not print_right:
    plt.savefig('throughput percent left.png')
    tikzplotlib.save('throughput percent left.tex')
elif print_right and not print_left:
    plt.savefig('throughput percent right.png')
    tikzplotlib.save('throughput percent right.tex')
else:
    plt.savefig('throughput percent.png')
    tikzplotlib.save('throughput percent.tex')
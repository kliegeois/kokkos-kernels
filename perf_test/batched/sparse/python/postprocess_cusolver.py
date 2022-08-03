import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tikzplotlib
from test_io import mmread

plt.style.use('plot_style.txt')


colourWheel =['#0072bd','#d95319', '#edb120','#7e2f8e','#77ac30','#4dbeee','#a2142f','#006e2d']
dashesStyles = [[3,1],
            [1000,1],
            [2,1,10,1]]

def plot_results(base, function_of_N, implementations_CG, implementations_GMRES):
    fig = plt.figure(figsize=(6, 4), dpi=120)
    ax = plt.gca()
    j = 0

    CPU_time_sp_QR = np.loadtxt(base+'CPU_time_0_sp.txt')[:,3]
    CPU_time_sp_Chol = np.loadtxt(base+'CPU_time_1_sp.txt')[:,3]
    CPU_time_dn = np.loadtxt(base+'CPU_time_0_dn.txt')[:,3]

    CPU_time_CG_left = np.loadtxt(base+'CPU_time_0_CG_left.txt')[:,3]
    CPU_time_CG_right = np.loadtxt(base+'CPU_time_0_CG_right.txt')[:,3]

    if function_of_N:
        x = np.loadtxt(base+'Ns.txt')
    else:
        x = np.loadtxt(base+'Bs.txt')

    print_left = True
    print_right = False
    print_cusolver_sparse = True
    print_cusolver_dense = True
    print_CG = False
    print_GMRES = True

    if print_cusolver_sparse:
        indices = np.argwhere(CPU_time_sp_QR > 0.)
        plt.semilogy(x[indices], CPU_time_sp_QR[indices], label='cusolver sparse QR', color=colourWheel[j%len(colourWheel)],
            linestyle = '-',
            dashes=dashesStyles[j%len(dashesStyles)])
        j +=1
        indices = np.argwhere(CPU_time_sp_Chol > 0.)
        plt.semilogy(x[indices], CPU_time_sp_Chol[indices], label='cusolver sparse Chol', color=colourWheel[j%len(colourWheel)],
            linestyle = '-',
            dashes=dashesStyles[j%len(dashesStyles)])
    j +=1
    if print_cusolver_dense:
        indices = np.argwhere(CPU_time_dn > 0.)
        plt.semilogy(x[indices], CPU_time_dn[indices], label='cusolver batched dense', color=colourWheel[j%len(colourWheel)],
            linestyle = '-',
            dashes=dashesStyles[j%len(dashesStyles)])
        j +=1
    if print_CG and print_left :
        for impl in implementations_CG:
            CPU_time_CG_left = np.loadtxt(base+'CPU_time_'+str(impl)+'_CG_left.txt')[:,3]
            indices = np.argwhere(CPU_time_CG_left > 0.)
            plt.semilogy(x[indices], CPU_time_CG_left[indices], label='CG left '+str(impl), color=colourWheel[j%len(colourWheel)],
                linestyle = '-',
                dashes=dashesStyles[j%len(dashesStyles)])
            j +=1
    if print_CG and print_right :
        for impl in implementations_CG:
            CPU_time_CG_right = np.loadtxt(base+'CPU_time_'+str(impl)+'_CG_right.txt')[:,3]
            indices = np.argwhere(CPU_time_CG_right > 0.)
            plt.semilogy(x[indices], CPU_time_CG_right[indices], label='CG right '+str(impl), color=colourWheel[j%len(colourWheel)],
                linestyle = '-',
                dashes=dashesStyles[j%len(dashesStyles)])
            j +=1
    if print_GMRES and print_left :
        for impl in implementations_GMRES:
            CPU_time_GMRES_left = np.loadtxt(base+'CPU_time_'+str(impl)+'_GMRES_left.txt')[:,3]
            indices = np.argwhere(CPU_time_GMRES_left > 0.)
            plt.semilogy(x[indices], CPU_time_GMRES_left[indices], label='Batched GMRES', color=colourWheel[j%len(colourWheel)],
                linestyle = '-',
                dashes=dashesStyles[j%len(dashesStyles)])
            j +=1
    if print_GMRES and print_right :
        for impl in implementations_GMRES:
            CPU_time_GMRES_right = np.loadtxt(base+'CPU_time_'+str(impl)+'_GMRES_right.txt')[:,3]
            indices = np.argwhere(CPU_time_GMRES_right > 0.)
            plt.semilogy(x[indices], CPU_time_GMRES_right[indices], label='GMRES right '+str(impl), color=colourWheel[j%len(colourWheel)],
                linestyle = '-',
                dashes=dashesStyles[j%len(dashesStyles)])
            j +=1      

    plt.grid()
    if function_of_N:
        ax.set_xlabel('Number of matrices')
    else:
        ax.set_xlabel('Number of rows')
    ax.set_xlim(np.amin(x), np.amax(x))
    ax.set_ylabel('Wall clock time [sec]')
    legend = ax.legend(loc='best', shadow=False, ncol=2)

    plt.savefig(base+'wall clock time.png', transparent=True)
    tikzplotlib.save(base+'wall clock time.tex')

#plot_results('weaver/cusolve_2/', False)
#plot_results('weaver/cusolve_3/', False)
plot_results('weaver/cusolve_6/', False, [0], [0])
#plot_results('weaver/cusolve_N/', True)

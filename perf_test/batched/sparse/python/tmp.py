import numpy as np

def print_array(arr):
    """
    prints a 2-D numpy array in a nicer format
    """
    for a in arr:
        for elem in a:
            print("{}".format(elem).rjust(3), end=" ")
        print(end="\n")

def print_results(filename_time, filename_N):
    print(filename_time)
    tmp = np.loadtxt(filename_time)
    results = np.zeros((tmp.shape[0], 2))
    results[:,1] = tmp[:,3]
    results[:,0] = np.loadtxt(filename_N)
    print_array(results)

filename_times_blake = ['blake/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted/CPU_time_0_l.txt',
    'blake/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted/CPU_time_0_l.txt',
    'blake/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted/CPU_time_0_r.txt',
    'blake/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted/CPU_time_0_r.txt',
    'blake/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted/CPU_time_0_l.txt',
    'blake/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted/CPU_time_0_l.txt',
    'blake/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted/CPU_time_0_r.txt',
    'blake/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted/CPU_time_0_r.txt']
filename_times_weaver = ['weaver/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted/CPU_time_2_l.txt',
    'weaver/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted/CPU_time_2_l.txt',
    'weaver/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted/CPU_time_2_r.txt',
    'weaver/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted/CPU_time_2_r.txt',
    'weaver/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted/CPU_time_0_l.txt',
    'weaver/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted/CPU_time_0_l.txt',
    'weaver/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted/CPU_time_0_r.txt',
    'weaver/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted/CPU_time_0_r.txt']
filename_times_caraway = ['caraway/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted/CPU_time_2_l.txt',
    'caraway/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted/CPU_time_2_l.txt',
    'caraway/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted/CPU_time_2_r.txt',
    'caraway/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted/CPU_time_2_r.txt',
    'caraway/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted/CPU_time_0_l.txt',
    'caraway/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted/CPU_time_0_l.txt',
    'caraway/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted/CPU_time_0_r.txt',
    'caraway/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted/CPU_time_0_r.txt']

filename_N = 'blake/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted/Ns.txt'


filename_times_CG = ['caraway/CG_data_Laplacian/CPU_time_0_l.txt',
    'caraway/CG_data_Laplacian/CPU_time_0_r.txt',
    'weaver/CG_data_Laplacian/CPU_time_0_l.txt',
    'weaver/CG_data_Laplacian/CPU_time_0_r.txt',
    'blake/CG_data_Laplacian/CPU_time_0_l.txt',
    'blake/CG_data_Laplacian/CPU_time_0_r.txt',
    'blake/CG_data_Laplacian/CPU_time_2_l.txt',
    'blake/CG_data_Laplacian/CPU_time_2_r.txt']

filename_N = 'caraway/CG_data_Laplacian/Ns.txt'

filename_times_blake = ['blake/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted_default_params/CPU_time_0_l.txt',
    'blake/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted_default_params/CPU_time_0_l.txt',
    'blake/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted_default_params/CPU_time_0_r.txt',
    'blake/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted_default_params/CPU_time_0_r.txt',
    'blake/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted_default_params/CPU_time_0_l.txt',
    'blake/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted_default_params/CPU_time_0_l.txt',
    'blake/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted_default_params/CPU_time_0_r.txt',
    'blake/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted_default_params/CPU_time_0_r.txt']
filename_times_weaver = ['weaver/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted_default_params/CPU_time_2_l.txt',
    'weaver/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted_default_params/CPU_time_2_l.txt',
    'weaver/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted_default_params/CPU_time_2_r.txt',
    'weaver/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted_default_params/CPU_time_2_r.txt',
    'weaver/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted_default_params/CPU_time_0_l.txt',
    'weaver/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted_default_params/CPU_time_0_l.txt',
    'weaver/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted_default_params/CPU_time_0_r.txt',
    'weaver/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted_default_params/CPU_time_0_r.txt']
filename_times_caraway = ['caraway/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted_default_params/CPU_time_2_l.txt',
    'caraway/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted_default_params/CPU_time_2_l.txt',
    'caraway/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted_default_params/CPU_time_2_r.txt',
    'caraway/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_unsorted_default_params/CPU_time_2_r.txt',
    'caraway/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted_default_params/CPU_time_0_l.txt',
    'caraway/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted_default_params/CPU_time_0_l.txt',
    'caraway/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_sorted_default_params/CPU_time_0_r.txt',
    'caraway/Pele_pGMRES_isooctane_data_Scaled_Jacobi_20_0_11_0_unsorted_default_params/CPU_time_0_r.txt']

filename_N = 'blake/Pele_pGMRES_gri30_data_Scaled_Jacobi_8_0_11_0_sorted_default_params/Ns.txt'

filename_times_CG = ['caraway/CG_data_Laplacian_default_params/CPU_time_0_l.txt',
    'caraway/CG_data_Laplacian_default_params/CPU_time_0_r.txt',
    'weaver/CG_data_Laplacian_default_params/CPU_time_0_l.txt',
    'weaver/CG_data_Laplacian_default_params/CPU_time_0_r.txt',
    'blake/CG_data_Laplacian_default_params/CPU_time_0_l.txt',
    'blake/CG_data_Laplacian_default_params/CPU_time_0_r.txt']

filename_N = 'caraway/CG_data_Laplacian_default_params/Ns.txt'


for filename_time in filename_times_CG:
    print_results(filename_time,filename_N)
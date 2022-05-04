import sys
import os

import subprocess

import argparse
# from mpi4py import MPI
import numpy as np
import time



from test_io import mmwrite, mmread
from run_Test import run_test, getHostName, getBuildDirectory
from create_matrices import *
from read_pele_matrices import *

exec_name = ""
A_file_name = ""
B_file_name = ""
X_file_name = ""
timer_filename = ""


def objectives(point):
    global exec_name, A_file_name, B_file_name, X_file_name, timer_filename

    team_size = 1 #point['team_size']
    N_team = point['N_team']
    vector_length = point['vector_length']

    n1 = 2
    n2 = 2

    tol = 1e-8
    n_iterations = 7
    ortho_strategy = 0
    other_level = 0
    implementation = 3
    layout = 'Right'

    exe = exec_name + ' -A ' +A_file_name+ ' -B '+B_file_name
    exe += ' -X ' + X_file_name + ' -timers '+ timer_filename
    exe += ' -n1 '+ str(n1)+' -n2 '+ str(n2)

    exe += ' -vector_length '+str(vector_length)
    exe += ' -N_team '+str(N_team)
    exe += ' -team_size '+str(team_size)

    exe += ' -implementation '+str(implementation)
    exe += ' -other_level '+str(other_level)
    exe += ' -n_iterations '+str(n_iterations)
    exe += ' -tol '+str(tol)
    exe += ' -ortho_strategy '+str(ortho_strategy)
    if layout == 'Left':
        exe += ' -l'
        y_name = '_l.txt'
        time_name = '_left.txt'
    if layout == 'Right':
        exe += ' -r'
        y_name = '_r.txt'
        time_name = '_right.txt'
    current_name = 'y_'+str(implementation)+y_name
    if os.path.exists(current_name):
        os.remove(current_name)
    subprocess.call(exe, shell=True)

    tmp = np.loadtxt(timer_filename+'_'+str(implementation)+time_name)
    mean_time = np.mean(tmp)
    return [mean_time]



def create_intput_files(input_folder, n_files, N, scaled, indices, sort, name_A, name_B):
    r, c, V, n = read_matrices(input_folder, n_files, N, scaled, indices=indices, sort=sort)
    B = read_vectors(input_folder, N, n, scaled, indices=indices, sort=sort)

    mmwrite(name_A, V, r, c, n, n)
    mmwrite(name_B, B)    


def getSortedIndices(specie, order):
    if specie == 'gri30':
        if order == 'descending':
            indices = np.array([31, 44, 57, 30, 33, 34, 35, 36, 37, 38, 39, 59, 40, 42, 43, 55, 45,
                46, 47, 48, 49, 50, 51, 41, 52, 60, 62, 79, 78, 77, 76, 75, 74, 61,
                72, 71, 73, 69, 70, 64, 65, 63, 67, 68, 66, 53, 56, 58, 82, 83, 84,
                85, 86, 87, 89, 88, 27, 28, 54, 32, 29,  3,  4,  5, 12, 81, 80,  7,
                8,  9, 10, 11,  6, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 13,  2,  1,  0])
        else:
            indices = np.array([ 0,  1,  2, 80, 81, 26, 25, 24, 23, 22, 21, 20, 18, 17, 16, 19, 14,
                13, 12, 11, 10,  9,  8,  7,  6,  5, 15,  3,  4, 56, 58, 82, 87, 84,
                85, 86, 54, 83, 88, 89, 27, 28, 29, 32, 40, 66, 67, 68, 69, 70, 71,
                72, 73, 76, 75, 65, 77, 78, 79, 34, 33, 30, 74, 39, 64, 62, 41, 42,
                43, 38, 45, 46, 47, 48, 49, 63, 50, 52, 53, 37, 55, 36, 57, 35, 59,
                60, 61, 51, 44, 31])
    if specie == 'isooctane':
        if order == 'descending':
            indices = np.array([71, 60, 50, 49, 37, 66, 33, 57, 41, 16, 43, 14, 40, 17, 15, 13, 45,
                46, 47, 55, 38, 58, 36, 34, 65, 32, 31, 30, 35, 28,  9, 25, 10, 11,
                20, 52, 53, 54,  6, 56, 29, 51, 61, 62, 63, 64, 67, 68, 69,  5, 59,
                7, 48, 27, 26, 24, 23, 70, 22,  8, 21, 39, 42, 12, 44,  4, 19,  2,
                3, 18,  1,  0])
        else:
            indices = np.array([ 0,  1, 18,  2,  3,  4, 19, 23, 24, 26, 27, 56, 53, 54, 52, 51, 70,
                48, 39, 29, 22, 21, 59,  5,  6,  7,  8, 69, 44, 67, 68, 12, 64, 63,
                62, 61, 42, 65, 58, 47, 46, 45, 55, 35, 20, 38,  9, 36, 10, 11, 34,
                25, 32, 31, 30, 28, 13, 17, 15, 40, 14, 41, 57, 43, 16, 33, 66, 37,
                50, 49, 60, 71])
    return indices

def main():

    specie = 'gri30'
    N = 100
    sort = True
    scaled = True

    if specie == 'gri30':
        n_files = 90
    if specie == 'isooctane':
        n_files = 72
    N *= n_files

    directory = getBuildDirectory()
    hostname = getHostName()

    if not os.path.isdir(hostname):
        os.mkdir(hostname)
    data_d = hostname + '/Pele_pGMRES_autotune'
    if not os.path.isdir(data_d):
        os.mkdir(data_d)

    input_folder = 'pele_data/jac-'+specie+'-typvals/'
    indices = getSortedIndices(specie,'descending')
    create_intput_files(input_folder, n_files, N, scaled, indices, sort, data_d+'/A.mm', data_d+'/B.mm')

    global exec_name, A_file_name, B_file_name, X_file_name, timer_filename
    exec_name = directory+'/KokkosBatched_Test_GMRES'
    A_file_name = data_d+'/A.mm'
    B_file_name = data_d+'/B.mm'
    X_file_name = data_d+'/X'
    timer_filename = data_d+'/timers'

    point = {}
    point['N_team'] = 1
    point['vector_length'] = 1
    objectives(point)


if __name__ == "__main__":
    main()

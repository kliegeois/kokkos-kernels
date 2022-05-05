#! /usr/bin/env python

# GPTune Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.
#


"""
Example of invocation of this script:

mpirun -n 1 python ./demo.py -nrun 20 -ntask 5 -perfmodel 0 -optimization GPTune

where:
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task
    -perfmodel is whether a coarse performance model is used
    -optimization is the optimization algorithm: GPTune,opentuner,hpbandster
"""


################################################################################
import sys
import os
# import mpi4py
import logging

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all

import subprocess

import argparse
# from mpi4py import MPI
import numpy as np
import time

from callopentuner import OpenTuner
from callhpbandster import HpBandSter


from test_io import mmwrite, mmread
from run_Test import run_test, getHostName, getBuildDirectory
from create_matrices import *
from read_pele_matrices import *

exec_name = ""
A_file_name = ""
B_file_name = ""
X_file_name = ""
timer_filename = ""

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=2,help='Number of cores per machine node')
    parser.add_argument('-machine', type=str,default='-1', help='Name of the computer (not hostname)')
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-perfmodel', type=int, default=0, help='Whether to use the performance model')
    parser.add_argument('-tvalue', type=float, default=1.0, help='Input task t value')

    args = parser.parse_args()

    return args


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

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    global nodes
    global cores

    # Parse command line arguments
    args = parse_args()
    ntask = args.ntask
    nrun = args.nrun
    tvalue = args.tvalue
    TUNER_NAME = args.optimization
    perfmodel = args.perfmodel

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

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
    
    team_size = Integer     (1, 2, transform="normalize", name="team_size")
    vector_length = Integer     (1, 8, transform="normalize", name="vector_length")
    N_team = Integer     (1, 32, transform="normalize", name="N_team")

    input_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    parameter_space = Space([vector_length, N_team])
    # input_space = Space([Real(0., 0.0001, "uniform", "normalize", name="t")])
    # parameter_space = Space([Real(-1., 1., "uniform", "normalize", name="x")])

    output_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    constraints = {}
    problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, None)

    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    options = Options()
    
    options['model_restarts'] = 1

    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False

    # options['objective_evaluation_parallelism'] = True
    # options['objective_multisample_threads'] = 1
    # options['objective_multisample_processes'] = 4
    # options['objective_nprocmax'] = 1

    options['model_processes'] = 1
    # options['model_threads'] = 1
    # options['model_restart_processes'] = 1

    # options['search_multitask_processes'] = 1
    # options['search_multitask_threads'] = 1
    # options['search_threads'] = 16

    # options['mpi_comm'] = None
    #options['mpi_comm'] = mpi4py.MPI.COMM_WORLD
    options['model_class'] = 'Model_GPy_LCM' #'Model_GPy_LCM'
    options['verbose'] = False
    # options['sample_algo'] = 'MCS'
    # options['sample_class'] = 'SampleLHSMDU'

    options.validate(computer=computer)

    if ntask == 1:
        giventask = [[round(tvalue,1)]]
    elif ntask == 2:
        giventask = [[round(tvalue,1)],[round(tvalue*2.0,1)]]
    else:
        giventask = [[round(tvalue*float(i+1),1)] for i in range(ntask)]

    NI=len(giventask)
    NS=nrun

    TUNER_NAME = os.environ['TUNER_NAME']

    if(TUNER_NAME=='GPTune'):
        data = Data(problem)
        gt = GPTune(problem, computer=computer, data=data, options=options,driverabspath=os.path.abspath(__file__))
        (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=int(NS/2), T_sampleflag=[True]*NI)
        # (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=NS-1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='opentuner'):
        (data,stats)=OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='cgp'):
        from callcgp import cGP
        options['EXAMPLE_NAME_CGP']='GPTune-Demo'
        options['N_PILOT_CGP']=int(NS/2)
        options['N_SEQUENTIAL_CGP']=NS-options['N_PILOT_CGP']
        (data,stats)=cGP(T=giventask, tp=problem, computer=computer, options=options, run_id="cGP")
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

if __name__ == "__main__":
    main()

 ################################################################################

import sys
import os
import numpy as np
import argparse
import pickle

# import mpi4py
from array import array
import math

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

from gptune import * # import all


from autotune.problem import *
from autotune.space import *
from autotune.search import *

from test_io import mmwrite, mmread
from run_Test import run_test, getHostName, getBuildDirectory
from create_matrices import *
from read_pele_matrices import *

# from callopentuner import OpenTuner
# from callhpbandster import HpBandSter
import math

################################################################################
def objectives(point):                          
	print('objective is not needed when options["RCI_mode"]=True')


def main():
	# Parse command line arguments
	args   = parse_args()

	# Extract arguments
	nprocmin_pernode = args.nprocmin_pernode
	optimization = args.optimization
	nrun = args.nrun
	obj = args.obj
	target=obj
	(machine, processor, nodes, cores) = GetMachineConfiguration()
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

	TUNER_NAME = 'GPTune'
	os.environ['MACHINE_NAME'] = machine

	matrices = ["gri30", "isooctane"]
	layouts = ["left", "right"]
	sorting_orders = ["u", "s"]

	# Task parameters
	matrix    = Categoricalnorm (matrices, transform="onehot", name="matrix")
	layout    = Categoricalnorm (layouts, transform="onehot", name="layout")
	sorting_order    = Categoricalnorm (sorting_orders, transform="onehot", name="sorting_order")

	# Input parameters
	team_size = Integer(1, 2, transform="normalize", name="team_size")
	vector_length = Integer(1, 32, transform="normalize", name="vector_length")
	N_team = Integer(1, 32, transform="normalize", name="N_team")

	result = Real(float("-Inf") , float("Inf"),name="time")

	IS = Space([matrix, layout, sorting_order])
	PS = Space([team_size, vector_length, N_team])
	OS = Space([result])

	constraints = {}

	problem = TuningProblem(IS, PS, OS, objectives, constraints, None)
	computer = Computer(nodes = nodes, cores = cores, hosts = None)  

	""" Set and validate options """	
	options = Options()
	options['RCI_mode'] = True
	options['model_processes'] = 1
	# options['model_threads'] = 1
	options['model_restarts'] = 1
	# options['search_multitask_processes'] = 1
	# options['model_restart_processes'] = 1
	options['distributed_memory_parallelism'] = False
	options['shared_memory_parallelism'] = False
	options['model_class'] = 'Model_LCM' # 'Model_LCM'
	options['verbose'] = False

	options.validate(computer = computer)
	
	# """ Building MLA with the given list of tasks """
	giventask = [["gri30", "left", "s"],["isooctane", "right", "u"]]		

	data = Data(problem)



	if(TUNER_NAME=='GPTune'):
		gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
		
		NI = len(giventask)
		NS = nrun
		(data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=max(NS//2, 1))
		# print("stats: ", stats)

		""" Print all input and parameter samples """	
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    matrix:%s"%(data.I[tid][0]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid].tolist())
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))


def parse_args():

	parser = argparse.ArgumentParser()

	# Problem related arguments
	parser.add_argument('-obj', type=str, default='time', help='Tuning objective (time or memory)')	
	# Machine related arguments
	parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
	parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
	parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
	parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
	# Algorithm related arguments
	parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
	parser.add_argument('-nrun', type=int, help='Number of runs per task')


	args   = parser.parse_args()
	return args


if __name__ == "__main__":
 
	main()

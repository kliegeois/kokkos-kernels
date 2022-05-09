#!/bin/bash
start=`date +%s`

# Get nrun, nprocmin_pernode, objecitve(memory or time) from command line
while getopts "a:b:c:" opt
do
   case $opt in
      a ) nrun=$OPTARG ;;
      b ) nprocmin_pernode=$OPTARG ;;
      c ) obj=$OPTARG ;;
      ? ) echo "unrecognized bash option $opt" ;; # Print helpFunction in case parameter is non-existent
   esac
done


# name of your machine, processor model, number of compute nodes, number of cores per compute node, which are defined in .gptune/meta.json
declare -a machine_info=($(python -c "from gptune import *;
(machine, processor, nodes, cores)=list(GetMachineConfiguration());
print(machine, processor, nodes, cores)"))
machine=${machine_info[0]}
processor=${machine_info[1]}
nodes=${machine_info[2]}
cores=${machine_info[3]}



database="gptune.db/batchedGMRES.json"  # the phrase batchedGMRES should match the application name defined in .gptune/meta.jason
# rm -rf $database


# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./batchedGMRES_RCI.py -nrun $nrun -obj $obj


# check whether GPTune needs more data
idx=$( jq -r --arg v0 $obj '.func_eval | map(.evaluation_result[$v0] == null) | index(true) ' $database )
if [ $idx = null ]
then
more=0
fi

# if so, call the application code
while [ ! $idx = null ]; 
do 
echo " $idx"    # idx indexes the record that has null objective function values
# write a large value to the database. This becomes useful in case the application crashes. 
bigval=1e30
jq --arg v0 $obj --argjson v1 $idx --argjson v2 $bigval '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database

declare -a input_para=($( jq -r --argjson v1 $idx '.func_eval[$v1].task_parameter' $database | jq -r '.[]'))
declare -a tuning_para=($( jq -r --argjson v1 $idx '.func_eval[$v1].tuning_parameter' $database | jq -r '.[]'))


#############################################################################
#############################################################################
# Modify the following according to your application !!! 


# get the task input parameters, the parameters should follow the sequence of definition in the python file
test_name=${input_para[0]}
IFS='_' read -r -a array <<< "$test_name"

if [ "${array[2]}" = "s" ]; then
   python ./batchedGMRES_RCI_pre.py --specie "${array[0]}" -s
else
   python ./batchedGMRES_RCI_pre.py --specie "${array[0]}"
fi

# get the tuning parameters, the parameters should follow the sequence of definition in the python file
team_size=${tuning_para[0]}
vector_length=${tuning_para[1]}
N_team=${tuning_para[2]}

exec_name=$(sed -n '1p' config_batchedGMRES.txt)
A_file_name=$(sed -n '2p' config_batchedGMRES.txt)
B_file_name=$(sed -n '3p' config_batchedGMRES.txt)
X_file_name=$(sed -n '4p' config_batchedGMRES.txt)
timer_filename=$(sed -n '5p' config_batchedGMRES.txt)
n1=$(sed -n '6p' config_batchedGMRES.txt)
n2=$(sed -n '7p' config_batchedGMRES.txt)
impl=$(sed -n '8p' config_batchedGMRES.txt)
other_level=$(sed -n '9p' config_batchedGMRES.txt)
n_iterations=$(sed -n '10p' config_batchedGMRES.txt)
tol=$(sed -n '11p' config_batchedGMRES.txt)
ortho_strategy=$(sed -n '12p' config_batchedGMRES.txt)

if [ "${array[1]}" = "left" ]; then
   ${exec_name} -A ${A_file_name} -B ${B_file_name} -X ${X_file_name} -timers ${timer_filename}\
   -n1 ${n1} -n2 ${n2} -vector_length ${vector_length} -N_team ${N_team} -team_size ${team_size} -implementation ${impl}\
   -other_level ${other_level} -n_iterations ${n_iterations} -tol ${tol} -ortho_strategy ${ortho_strategy} -l

   # get result using batchedGMRES_RCI_post.py
   declare -a result=($(python batchedGMRES_RCI_post.py --timer_filename ${timer_filename} --implementation ${impl} --layout Left))
else
   ${exec_name} -A ${A_file_name} -B ${B_file_name} -X ${X_file_name} -timers ${timer_filename}\
   -n1 ${n1} -n2 ${n2} -vector_length ${vector_length} -N_team ${N_team} -team_size ${team_size} -implementation ${impl}\
   -other_level ${other_level} -n_iterations ${n_iterations} -tol ${tol} -ortho_strategy ${ortho_strategy} -r

   # get result using batchedGMRES_RCI_post.py
   declare -a result=($(python batchedGMRES_RCI_post.py --timer_filename ${timer_filename} --implementation ${impl} --layout Right))
fi
echo $result

# write the data back to the database file
jq --arg v0 $obj --argjson v1 $idx --argjson v2 $result '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database
idx=$( jq -r --arg v0 $obj '.func_eval | map(.evaluation_result[$v0] == null) | index(true) ' $database )

#############################################################################
#############################################################################



done
done

end=`date +%s`

runtime=$((end-start))
echo "Total tuning time: $runtime"


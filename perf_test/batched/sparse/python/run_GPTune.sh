timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

export PYTHONPATH=/home/knliege/local/trilinos/albany_release/lib/:/home/knliege/local/albany/albany_release/lib/python3.8/site-packages:/home/knliege/local/trilinos/albany_release/lib/python3.8/site-packages:/home/knliege/local/trilinos/albany_release/lib/:/home/knliege/local/albany/albany_release/lib/python3.8/site-packages:/home/knliege/local/trilinos/albany_release/lib/python3.8/site-packages::/home/knliege/dev/gptune/scikit-optimize:/home/knliege/dev/gptune/GPTune:/home/knliege/dev/gptune/GPTune/GPTune

export MKL_DYNAMIC=TRUE
export OMP_DYNAMIC=FALSE
export OMP_NUM_THREADS=26
export OMP_PROC_BIND=true
#export OMP_DISPLAY_AFFINITY=TRUE
export OMP_PLACES=threads
#export OMP_DISPLAY_ENV=true

rm -rf gptune.db/*.json # do not load any database 
tp=batchedGMRES
app_json=$(echo "{\"tuning_problem_name\":\"$tp\",\"no_load_check\": \"yes\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
bash batchedGMRES_RCI.sh -a 100 -b 1 -c time
cp gptune.db/batchedGMRES.json  gptune.db/batchedGMRES.json_$(timestamp)

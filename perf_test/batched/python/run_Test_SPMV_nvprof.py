import numpy as np
import subprocess
import time


def run_test_spmv(N=12800, B=200, nnz_per_row=5, rows_per_thread=1, team_size=8, n1=10, n2=10, implementation=0):
    exec = 'KokkosBatched_Test_SPMV -N ' +str(N)+ ' -B '+str(B)+' -nnz_per_row '+ str(nnz_per_row)+' -n1 '+ str(n1)+' -n2 '+ str(n2)+' -rows_per_thread '+str(rows_per_thread)+' -team_size '+str(team_size)+ ' -implementation '+str(implementation)
    testfile = 'KokkosBatched_Test_SPMV.'+str(implementation)+'.'+str(B)+'.'+str(nnz_per_row)+'.'+str(N)+'.txt'
    nv_prof='nvprof --metrics achieved_occupancy,dram_read_bytes,dram_read_throughput,dram_read_transactions,dram_utilization,dram_write_throughput,gst_throughput,gld_throughput,l2_l1_read_throughput,l2_tex_read_throughput,l2_tex_write_throughput,local_load_throughput,local_store_throughput,shared_load_throughput,shared_store_throughput,global_hit_rate,gld_efficiency,gld_requested_throughput,global_load_requests,shared_efficiency,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_dp_mul,sysmem_read_bytes,sysmem_read_throughput,sysmem_read_transactions,sysmem_read_utilization,tex_cache_throughput,tex_cache_transactions,texture_load_requests,nc_cache_global_hit_rate,l2_l1_read_hit_rate,l1_cache_local_hit_rate,l1_cache_global_hit_rate,l2_tex_read_hit_rate,local_hit_rate,l1_cache_hit_rate,tex_cache_hit_rate,sm_efficiency,warp_execution_efficiency,branch_efficiency,stall_inst_fetch,stall_memory_dependency,stall_exec_dependency,stall_memory_throttle,stall_pipe_busy,stall_not_selected'
    nv_prof+= ' ./' + exec + ' |& tee ' + testfile

    print( nv_prof)
    subprocess.call(nv_prof, shell=True)


def main():
    N = 12800
    B = 100
    nnz_per_rows=np.arange(1,B+1)
    rows_per_thread=1
    team_size=8

    for nnz_per_row in nnz_per_rows:
        run_test_spmv(N, B, nnz_per_row, rows_per_thread, team_size, n1=10, n2=10, implementation=4)


if __name__ == "__main__":
    main()

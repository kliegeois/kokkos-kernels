import re
import pandas as pd
import numpy as np
from io import StringIO
from create_matrices import *

import tikzplotlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_n_ops(nrows, nnz, number_of_matrices, bytes_per_entry=8, unit='B/sec'):
    # 1 "+" and 1 "*" per entry of A and 1 "+" and 1 "*" per row
    nops = 2*(nnz+nrows)*number_of_matrices
    if unit=='B/sec':
        return nops*bytes_per_entry
    if unit=='GFLOPS':
        return nops/1e9

def remove_all_non_num(line):
    return re.sub('[^0-9. ]', ' ', str(line))

def get_row_data(df, metric_name, column_names):
    data = np.zeros((len(column_names), ))
    metric_found = False
    for index, row in df.iterrows():
        if row["Metric Name"] == metric_name:
            n_inv = row["Invocations"]
            for i in range(0, len(data)):
                if column_names[i] in df.columns:
                    data[i] = float(remove_all_non_num(row[column_names[i]]))
            metric_found = True
    return data, metric_found, n_inv

def add_new_ratio(df, num_name, denom_name, new_name, new_description, column_names):
    num_data, num_found, n_inv = get_row_data(df, num_name, column_names)
    denom_data, denom_found, n_inv = get_row_data(df, denom_name, column_names)
    if num_found and denom_found:
        data = [n_inv, new_name, new_description]
        columns = ["Invocations", "Metric Name", "Metric Description"]
        for i in range(0, len(num_data)):
            data.append("{:.2f}%".format(num_data[i]/denom_data[i]*100))
            columns.append(column_names[i])
        df = df.append(pd.DataFrame([data], columns=columns))
    return df


def add_new_sum(df, sum_names, new_name, new_description, column_names):
    sum_data = np.zeros((len(column_names), ))
    for index, row in df.iterrows():
        for j in range(0, len(sum_names)):
            if row["Metric Name"] == sum_names[j]:
                n_inv = row["Invocations"]
                for i in range(0, len(sum_data)):
                    if column_names[i] in df.columns:
                        sum_data[i] += float(remove_all_non_num(row[column_names[i]]))

    data = [n_inv, new_name, new_description]
    columns = ["Invocations", "Metric Name", "Metric Description"]
    for i in range(0, len(sum_data)):
        data.append("{:.2f}%".format(sum_data[i]))
        columns.append(column_names[i])
    df = df.append(pd.DataFrame([data], columns=columns))
    return df


def get_data_frame_from_file(filename, string_first_line='Functor_TestBatchedTeamVectorSpmv'):
    read_line = False
    df = pd.DataFrame({})
    with open(filename) as input_file:
        for line in input_file:
            if 'solve time =' in line:
                pattern = "solve time = (.*?) , #"
                time = re.search(pattern, line).group(1)
            if string_first_line in line:
                read_line = True
                continue
            if read_line:
                if line[0] == '=' or 'Kokkos::' in line:
                    read_line = False
                    continue
                if 'System Memory Read Utilization' in line:
                    continue
                tabulated_line = re.sub(' {2,}', '\t', line)
                df = pd.concat([df, pd.read_csv(
                    StringIO(tabulated_line), sep="\t", header=None)], ignore_index=True)

    df = df.drop([0], axis=1)
    df = df.rename(columns={1: "Invocations", 2: "Metric Name",
                   3: "Metric Description", 4: "Min", 5: "Max", 6: "Avg"})

    return df


def combine_data_frame_from_files(filenames, avg_names, string_first_line):
    df0 = get_data_frame_from_file(filenames[0], string_first_line)
    df0 = df0.rename(columns={"Avg": avg_names[0]})
    df0 = df0.drop(["Min", "Max"], axis=1)
    for i in range(1, len(filenames)):
        df = get_data_frame_from_file(filenames[i], string_first_line)
        df = df.rename(columns={"Avg": avg_names[i]})
        df0 = df0.join(df[avg_names[i]])
    df0 = add_new_ratio(df0, 'dram_read_throughput', 'gld_requested_throughput', '-', 'dram_read_throughput/gld_requested_throughput', avg_names)

    df0 = add_new_ratio(df0, 'l2_tex_read_throughput', 'tex_cache_throughput', '-', 'l2_tex_read_throughput/tex_cache_throughput', avg_names)
    df0 = add_new_ratio(df0, 'dram_read_throughput', 'l2_tex_read_throughput', '-', 'dram_read_throughput/l2_tex_read_throughput', avg_names)
    df0 = add_new_ratio(df0, 'sysmem_read_throughput', 'dram_read_throughput', '-', 'sysmem_read_throughput/dram_read_throughput', avg_names)

    df0 = add_new_ratio(df0, 'tex_cache_throughput', 'gld_throughput', '-', 'tex_cache_throughput/gld_throughput', avg_names)
    df0 = add_new_ratio(df0, 'l2_tex_read_throughput', 'gld_throughput', '-', 'l2_tex_read_throughput/gld_throughput', avg_names)
    df0 = add_new_ratio(df0, 'dram_read_throughput', 'gld_throughput', '-', 'dram_read_throughput/gld_throughput', avg_names)
    df0 = add_new_ratio(df0, 'sysmem_read_throughput', 'gld_throughput', '-', 'sysmem_read_throughput/gld_throughput', avg_names)

    df0 = add_new_sum(df0, ['stall_inst_fetch', 'stall_memory_dependency', 'stall_exec_dependency', 'stall_memory_throttle', 'stall_pipe_busy', 'stall_not_selected'], '-', 'Sum stall', avg_names)
    
    return df0

def main():

    impls = [0, 3]
    layouts = ['left', 'right']

    m_max = 33
    ms = np.arange(1, m_max)

    avg_names = []
    for m in ms:
        avg_names.append(str(m))

    base = 'Pele_SPMV_NVPROF_gri30_data_4'

    plot_throughput = True
    df = []
    throughput = []
    names = []
    for impl in impls:
        for layout in layouts:
            if plot_throughput:

                N = 224*90
                nrows = 54
                nnz = 2560
                unit = 'GFLOPS'
                n_ops = compute_n_ops(nrows, nnz, N, bytes_per_entry=8, unit=unit)

                if layout == 'left':
                    cpu_time = np.loadtxt('weaver/Pele_SPMV_gri30_data_SPMV_vec_m_default/CPU_time_'+str(impl)+'_l.txt')[:,3]
                else:
                    cpu_time = np.loadtxt('weaver/Pele_SPMV_gri30_data_SPMV_vec_m_default/CPU_time_'+str(impl)+'_r.txt')[:,3]
                
                throughput_tmp = np.copy(cpu_time)
                for i in range(0, len(throughput_tmp)):
                    if throughput_tmp[i] > 0.:
                        throughput_tmp[i] = n_ops / throughput_tmp[i]
                throughput.append(throughput_tmp)

            filenames = []
            for m in ms:
                filenames.append(base+'/nvprof.'+str(impl)+'.54.2560.20160.'+str(m)+'.'+layout+'.txt')

            if impl == 0:
                df.append(combine_data_frame_from_files(filenames, avg_names, 'BSPMV_Functor_View'))
            else:
                df.append(combine_data_frame_from_files(filenames, avg_names, 'Functor_TestBatchedTeamVectorSpmv'))

            names.append(str(impl)+'_'+layout)

    if plot_throughput:
        plt.figure()
        ax = plt.gca()
        for i in range(0, len(df)):
            plt.plot(ms, throughput[i], label=names[i])
        plt.grid()
        ax.set_xlabel('m')
        ax.set_ylabel('Throughput')
        legend = ax.legend(loc='best', shadow=False)
        plt.savefig(base+'/plot_throughput.png')
        tikzplotlib.save(base+'/plot_throughput.tex')
        plt.close()


    metrics = ['stall_exec_dependency', 'gld_efficiency', 'achieved_occupancy', 'sm_efficiency', 'warp_execution_efficiency', 'gld_requested_throughput', 'tex_cache_hit_rate', 'stall_memory_dependency', 'stall_memory_throttle', 'gst_efficiency', 'flop_dp_efficiency', 'dram_read_throughput']

    for metric in metrics:
        plt.figure()
        ax = plt.gca()
        for i in range(0, len(df)):
            row_data, found, n_inv = get_row_data(df[i], metric, avg_names)
            plt.plot(ms, row_data, label=names[i])
        plt.grid()
        ax.set_xlabel('m')
        ax.set_ylabel(metric)
        legend = ax.legend(loc='best', shadow=False)
        plt.savefig(base+'/plot_'+metric+'.png')
        tikzplotlib.save(base+'/plot_'+metric+'.tex')
        plt.close()

        if metric == 'flop_dp_efficiency':
            peak = 7.8e3
            plt.figure()
            ax = plt.gca()
            for i in range(0, len(df)):
                row_data, found, n_inv = get_row_data(df[i], metric, avg_names)
                plt.plot(ms, peak*row_data/100., label=names[i])
            plt.grid()
            ax.set_xlabel('m')
            ax.set_ylabel(metric)
            legend = ax.legend(loc='best', shadow=False)
            plt.savefig(base+'/plot_flop_dp.png')
            tikzplotlib.save(base+'/plot_flop_dp.tex')
            plt.close()            



if __name__ == "__main__":
    main()

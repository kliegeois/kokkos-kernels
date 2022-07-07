import re
import pandas as pd
import numpy as np
from io import StringIO
from create_matrices import *

def remove_all_non_num(line):
    return re.sub('[^0-9. ]', ' ', line)


def add_new_ratio(df, num_name, denom_name, new_name, new_description, column_names):
    num_found = False
    denom_found = False
    num_data = np.zeros((len(column_names), ))
    denom_data = np.ones((len(column_names), ))
    for index, row in df.iterrows():
        if row["Metric Name"] == num_name:
            n_inv = row["Invocations"]
            for i in range(0, len(num_data)):
                if column_names[i] in df.columns:
                    num_data[i] = float(remove_all_non_num(row[column_names[i]]))
            num_found = True
        if row["Metric Name"] == denom_name:
            for i in range(0, len(denom_data)):
                if column_names[i] in df.columns:
                    denom_data[i] = float(remove_all_non_num(row[column_names[i]]))
            denom_found = True
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

    impl = 0
    m_max = 17
    filenames = []
    avg_names = []
    for m in range(1, m_max):
        filenames.append('Pele_SPMV_NVPROF_gri30_data_1/nvprof.'+str(impl)+'.54.2560.20160.'+str(m)+'.left.txt')
        avg_names.append(str(m))

    if impl == 0:
        df0 = combine_data_frame_from_files(filenames, avg_names, 'BSPMV_Functor_View')
    else:
        df0 = combine_data_frame_from_files(filenames, avg_names, 'Functor_TestBatchedTeamVectorSpmv')
    df0.to_csv('test_m_'+str(impl)+'.csv')


if __name__ == "__main__":
    main()

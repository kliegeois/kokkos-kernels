import numpy as np

def create_strided_graph(n, max_offset, offset):
    r = np.array([])
    c = np.array([])
    for current_row in range(0, n):
        for i_offset in range(-max_offset,max_offset+1):
            current_col = current_row + i_offset * offset
            if current_col >= 0 and current_col < n:
                r = np.append(r, current_row)
                c = np.append(c, current_col)
    return r, c


def create_2D_Laplacian_graph(n_node_1D_I, n_node_1D_J):
    n_rows = int(n_node_1D_I * n_node_1D_J)
    r = np.array([])
    c = np.array([])
    for node_j in range(0, n_node_1D_J):
        for node_i in range(0, n_node_1D_I):
            current_row = node_j*n_node_1D_I + node_i
            current_cols = np.array([current_row])
            if node_i != 0 and node_j != 0:
                current_cols = np.append(current_cols, (node_j-1)*n_node_1D_I + (node_i-1))
            if node_j != 0:
                current_cols = np.append(current_cols, (node_j-1)*n_node_1D_I + node_i)
            if node_i != n_node_1D_I-1 and node_j != 0:
                current_cols = np.append(current_cols, (node_j-1)*n_node_1D_I + (node_i+1))
            if node_i != 0:
                current_cols = np.append(current_cols, node_j*n_node_1D_I + (node_i-1))
            if node_i != n_node_1D_I-1:
                current_cols = np.append(current_cols, node_j*n_node_1D_I + (node_i+1))
            if node_i != 0 and node_j != n_node_1D_J-1:
                current_cols = np.append(current_cols, (node_j+1)*n_node_1D_I + (node_i-1))
            if node_j != n_node_1D_J-1:
                current_cols = np.append(current_cols, (node_j+1)*n_node_1D_I + node_i)
            if node_i != n_node_1D_I-1 and node_j != n_node_1D_J-1:
                current_cols = np.append(current_cols, (node_j+1)*n_node_1D_I + (node_i+1))
            current_cols = np.sort(current_cols)
            for current_col in current_cols:
                r = np.append(r, current_row)
                c = np.append(c, current_col)
    return r, c, n_rows


def create_SPD(n, r, c, N, v_min=-1, v_max=1):
    nnz = len(r)
    V = v_min + np.random.rand(N, nnz) * (v_max-v_min)

    # Make the matrix symmetrical
    for i in range(0, nnz):
        row = r[i]
        col = c[i]
        if col > row:
            for j in range(i, nnz):
                if col == r[j] and row == c[j]:
                    V[:,j] = V[:,i]
                    break

    # Make the diagonal dominant (and therefore SPD using the Gershgorin circle theorem)
    first_i = 0
    diag_i = 0
    last_i = 0    
    for current_row in range(0, n):
        last_i = nnz
        for i in range(first_i, nnz):
            if r[i] > current_row:
                last_i = i
                break
            if c[i] == current_row:
                diag_i = i
        
        V[:,diag_i] = np.abs(V[:,diag_i])
        for i in range(first_i, last_i):
            if i != diag_i:
                V[:,diag_i] += np.abs(V[:,i])
        first_i = last_i
    return V


def create_Vector(n, N, v_min=-1, v_max=1):
    V = v_min + np.random.rand(N, n) * (v_max-v_min)
    return V

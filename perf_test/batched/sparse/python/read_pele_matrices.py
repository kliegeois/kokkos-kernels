import numpy as np
from scipy import sparse

def read_matrix(directory, j):
    name = str(j)
    n_zeros = 6-len(name)
    for i in range(0, n_zeros):
        name = '0' + name
    name = name + '.txt'

    A_j = np.loadtxt(directory+name)
    ewt = np.loadtxt(directory+'ewt.txt')

    Ajtilde = np.diag(ewt[j, :]).dot(A_j).dot(np.diag(1./ewt[j, :]))

    sA = sparse.csr_matrix(Ajtilde)

    return sA


def read_matrices(directory, n_files, n_matrices):
    n_read = min(n_files, n_matrices)
    for j in range(0, n_read):
        A_j = read_matrix(directory, j)
        if j == 0:
            r = A_j.tocoo().row
            c = A_j.tocoo().col
            V = np.zeros((n_matrices, len(r)))
        V[j, :] = A_j.data
    for j in range(n_read, n_matrices):
        V[j, :] = np.copy(V[j-n_read, :])
    return r, c, V, A_j.shape[0]

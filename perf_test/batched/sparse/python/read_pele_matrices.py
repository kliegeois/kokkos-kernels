import numpy as np

def read_matrix(directory, j):
    name = str(j)
    n_zeros = 6-len(name)
    for i in range(0, n_zeros):
        name = '0' + name
    name = name + '.txt'

    A_j = np.loadtxt(directory+name)
    ewt = np.loadtxt(directory+'ewt.txt')

    Ajtilde = np.diag(ewt[j, :]).dot(A_j).dot(np.diag(1./ewt[j, :]))

    return Ajtilde


def read_matrix_spd(directory, j):
    name = str(j)
    n_zeros = 6-len(name)
    for i in range(0, n_zeros):
        name = '0' + name
    name = name + '.txt'

    A_j = np.loadtxt(directory+name)
    ewt = np.loadtxt(directory+'ewt.txt')

    Ajtilde = np.diag(ewt[j, :]).dot(A_j).dot(np.diag(1./ewt[j, :]))

    sym = Ajtilde + Ajtilde.transpose()

    return sym


def read_matrices(directory, n_files, n_matrices):
    n_read = min(n_files, n_matrices)
    for j in range(0, n_read):
        A_j = read_matrix(directory, j)
        if j == 0:
            r = np.array([])
            c = np.array([])
            for ii in range(0, A_j.shape[0]):
                for jj in range(0, A_j.shape[1]):
                    if A_j[ii,jj] != 0.:
                        r = np.append(r, ii)
                        c = np.append(c, jj)
            V = np.zeros((n_matrices, len(r)))
        i = 0
        for ii in range(0, A_j.shape[0]):
            for jj in range(0, A_j.shape[1]):
                if A_j[ii,jj] != 0.:
                    V[j, i] = A_j[ii,jj]
                    i += 1
    for j in range(n_read, n_matrices):
        V[j, :] = np.copy(V[j-n_read, :])
    return r, c, V, A_j.shape[0]


def read_matrices_spd(directory, n_files, n_matrices):
    n_read = min(n_files, n_matrices)
    for j in range(0, n_read):
        A_j = read_matrix_spd(directory, j)
        if j == 0:
            r = np.array([])
            c = np.array([])
            for ii in range(0, A_j.shape[0]):
                for jj in range(0, A_j.shape[1]):
                    if A_j[ii,jj] != 0.:
                        r = np.append(r, ii)
                        c = np.append(c, jj)
            V = np.zeros((n_matrices, len(r)))
        i = 0
        for ii in range(0, A_j.shape[0]):
            for jj in range(0, A_j.shape[1]):
                if A_j[ii,jj] != 0.:
                    V[j, i] = A_j[ii,jj]
                    i += 1
    for j in range(n_read, n_matrices):
        V[j, :] = np.copy(V[j-n_read, :])
    return r, c, V, A_j.shape[0]

def read_vectors(directory, n_vectors, n_rows):
    tmpV = np.loadtxt(directory)
    n_read = min(tmpV.shape[0],n_vectors)
    V = np.zeros((n_vectors, n_rows))
    V[0:n_read,:] = np.copy(tmpV[0:n_read,:])
    for j in range(n_read, n_vectors):
        V[j, :] = np.copy(V[j-n_read, :])
    return V

import numpy as np

def read_matrix(directory, j, scaled=True):
    name = str(j)
    n_zeros = 6-len(name)
    for i in range(0, n_zeros):
        name = '0' + name
    name = name + '.txt'

    A_j = np.loadtxt(directory+name)
    if scaled:
        ewt = np.loadtxt(directory+'ewt.txt')
        v = ewt[j, :]
        S_1 = np.diag(v)
        S_2 = np.diag(1./v)

        tmp = np.matmul(A_j,S_2)
        Ajtilde = np.matmul(S_1,tmp)

        return Ajtilde
    else:
        return A_j

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


def read_matrices(directory, n_files, n_matrices, scaled=True, indices=None, sort=False):
    n_read = min(n_files, n_matrices)
    for j in range(0, n_read):
        A_j = read_matrix(directory, j, scaled)
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
    if sort:
        n_duplicate = int(np.ceil(1.*n_matrices/len(indices)))
        new_indices = np.zeros((n_duplicate*len(indices),), dtype=int)
        for i in range(0, len(indices)):
            for j in range(0, n_duplicate):
                new_indices[i*n_duplicate+j] = indices[i]
        tmpV = np.copy(V)
        for i in range(0, n_matrices):
            V[i, :] = tmpV[new_indices[i], :]
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

def read_vectors(directory, n_vectors, n_rows, scaled=True, indices=None, sort=False):
    tmpV = np.loadtxt(directory+'rhs.txt')
    ewt = np.loadtxt(directory+'ewt.txt')
    n_read = min(tmpV.shape[0],n_vectors)
    V = np.zeros((n_vectors, n_rows))
    for j in range(0, n_read):
        S_1 = np.diag(ewt[j, :])
        if scaled:
            V[j, :] = np.matmul(S_1,tmpV[j, :])
        else:
            V[j, :] = tmpV[j, :]
    for j in range(n_read, n_vectors):
        V[j, :] = np.copy(V[j-n_read, :])
    if sort:
        n_duplicate = int(np.ceil(1.*n_vectors/len(indices)))
        new_indices = np.zeros((n_duplicate*len(indices),), dtype=int)
        for i in range(0, len(indices)):
            for j in range(0, n_duplicate):
                new_indices[i*n_duplicate+j] = indices[i]
        tmpV = np.copy(V)
        for i in range(0, n_vectors):
            V[i, :] = tmpV[new_indices[i], :]
    
    return V

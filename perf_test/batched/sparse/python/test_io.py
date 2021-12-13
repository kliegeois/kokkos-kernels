import numpy as np
try:
    from scipy.sparse import csr_matrix
    support_scipy = True
except:
    support_scipy = False
    
def mmwrite_crs(name, m, n, r, c, v):
    with open(name, 'w') as f:
        print('%%MatrixMarket CRS matrix\n%', file=f)
        nnz = len(v)
        print(str(int(m)) + ' ' + str(int(n)) + ' ' + str(nnz), file=f)
        for i in range(0, nnz):
            print(str(int(r[i]+1)) + ' ' + str(int(c[i]+1)) + ' ' + str(v[i]), file=f)


def mmwrite_batchedCrs(name, m, n, r, c, V):
    with open(name, 'w') as f:
        print('%%MatrixMarket batched CRS matrix\n%', file=f)
        nnz = np.shape(V)[1]
        N = np.shape(V)[0]
        print(str(int(m)) + ' ' + str(int(n)) + ' ' + str(int(nnz)) + ' ' + str(int(N)), file=f)
        for i in range(0, nnz):
            print(str(int(r[i]+1)) + ' ' + str(int(c[i]+1)), file=f, end =' ')
            for j in range(0, N):
                print(str(V[j,i]), file=f, end =' ')
            print('', file=f)


def mmwrite_1D_view(name, v):
    with open(name, 'w') as f:
        print('%%MatrixMarket 1D view\n%', file=f)
        m = np.shape(v)[0]
        print(str(int(m)) + ' 1', file=f)
        for i in range(0, m):
            print(str(v[i]), file=f)


def mmwrite_2D_view(name, V):    
    with open(name, 'w') as f:
        print('%%MatrixMarket 2D view\n%', file=f)
        m = np.shape(V)[0]
        n = np.shape(V)[1]
        print(str(int(m)) + ' ' + str(int(n)), file=f)
        for i in range(0, m):
            for j in range(0, n):
                print(str(V[i,j]), file=f, end =' ')
            print('', file=f)


def mmwrite(name, V, r=0, c=0, m=0, n=0):
    if m == 0:
        if len(np.shape(V)) == 2:
            mmwrite_2D_view(name, V)
        else:
            mmwrite_1D_view(name, V)
    else:
        if len(np.shape(V)) == 2:
            mmwrite_batchedCrs(name, m, n, r, c, V)
        else:
            mmwrite_crs(name, m, n, r, c, V)


def mmread(filename, batched_id=0):

    skip_header = 0
    with open(filename, "r") as mm_file:
        for line in mm_file:
            if line[0] == '%':
                skip_header += 1
            else:
                break
    X_dim = np.genfromtxt(filename, skip_header=skip_header, max_rows=1)
    m = X_dim[0].astype(int)
    n = X_dim[1].astype(int)

    if len(X_dim) >= 3:
        is_sparse = True
        if len(X_dim) == 3:
            batched_id = 0
    else:
        is_sparse = False

    if is_sparse:
        X = np.loadtxt(filename, skiprows=skip_header+1)

        row = X[:, 0].astype(int)-1
        col = X[:, 1].astype(int)-1
        data = X[:, 2 + batched_id]

        mask = data != 0.
        m = np.amax(row)+1
        n = np.amax(col)+1
        if support_scipy:
            A = csr_matrix((data[mask], (row[mask], col[mask])), shape=(m, n))
            B = A.tocsc()
        else:
            B = np.zeros((m, n))
            for i in range(0, len(row)):
                B[row[i], col[i]] = data[i]
    else:
        X = np.loadtxt(filename, skiprows=skip_header+1)
        if n != 1:
            B = X.reshape((m, n)).T
        else:
            B = X.reshape((m,))

    return B

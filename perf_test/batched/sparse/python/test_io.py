import numpy as np

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_matrix(file):
    m = open(file, 'r')
    for i, line in enumerate(m):
        if i == 2:
            dim_x, dim_y, dim_val = line.split()
            M = np.zeros((int(dim_x), int(dim_y)))
        if i >= 3:
            x, y, val = line.split()
            M[int(x) - 1, int(y) - 1] = int(val)
    m.close()
    print(M.shape)
    return M


def read_target(file, num):
    t = open(file, 'r')
    for i, line in enumerate(t):
        if i == 2:
            dim_x, dim_val = line.split()
            if int(dim_x) != num:
                raise('Deve esserci una riga per ogni oggetto!') # type: ignore
            else:
                T = np.zeros((int(dim_x)))

        if i >= 3:
            val = int(line)
            T[i - 3] = val
    t.close()
    return T


def plot_clusters(model):
    arr1inds = model._row_assignment.argsort()
    U = np.ones(model._dataset.shape)
    X = (U.T * (model._row_assignment + 1)).T
    #plt.matshow((model._dataset*X)[arr1inds])
    plt.matshow((model._dataset*X))
    #plt.show()

    arr2inds = model._col_assignment.argsort()
    U = np.ones(model._dataset.shape)
    X = (U * (model._col_assignment + 1))
    #plt.matshow((model._dataset*X)[:,arr2inds])
    plt.matshow((model._dataset*X))    

def MovieLens():
    final = pd.read_pickle('../resources/movielens_final_3g_6_2u.pkl')
    n = np.shape(final.groupby('userID').count())[0]
    m = np.shape(final.groupby('movieID').count())[0]
    l = np.shape(final.groupby('tagID').count())[0]
    T = np.zeros((n,m,l))
    y = np.zeros(m)
    for index, row in final.iterrows():
        T[row['user_le'], row['movie_le'], row['tag_le']] = 1
        y[row['movie_le']] = row['genre_le']
    T1 = np.sum(T, axis = 2)
    return T1, y

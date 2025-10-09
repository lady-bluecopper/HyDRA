import numpy as np
from itertools import product, islice, filterfalse
from warnings import warn

def CreateMatrix(nrows = None, ncols = None, rowclust = None, colclust = None, noise = 0, v_rowclust = None, v_colclust = None, max_attempts = 1, random_state=None):

    """
    Crea una matrice di 0 e 1 con un numero fissato di clusters sulle righe e sulle colonne.
    ATTENZIONE: Per ottenere cluster perfettamente separati, fissato il numero n di clusters sulle righe
    il numero massimo di cluster sulle colonne deve essere minore di 2^n (o viceversa)

    Parameters
    ----------

    nrows: number of rows
    ncols: number of columns
    rowclust: number of row clusters.
    colclust: number of column clusters
    noise: probability of error in clusters creation
    v_rowclust: list of lenghts of each row cluster;
                Uno tra v_rowclust e rowclust deve essere valorizzato.
                v_rowclust vince su rowclust (e come nrows viene impostata la somma di v_rowclust)                
    v_colclust: list of lengths of each column cluster;
                Uno tra v_colclust e colclust deve essere valorizzato.
                v_colclust vince su colclust (e come ncols viene impostata la somma di v_colclust)
    max_attempts: parametro per la funzione replaceRandom.
                Indica il numero massimo di volte in cui si prova a generare una matrice con rumore se nei tentativi precedenti si erano ottenute matrici con intere righe o colonne interamente a zero.

    Returns
    -------

    V matrice nrows x ncols

    """
    rng = np.random.default_rng(seed = random_state)

    if (rowclust == None or nrows == None) and v_rowclust == None:
        raise ValueError('Valorizzare o nrows e rowclust oppure v_rowclust')

    if v_rowclust == None:
        v_rowclust = f(nrows, rowclust)
    else:
        rowclust = len(v_rowclust)
        nrows = sum(v_rowclust)

    if (colclust == None or ncols == None) and v_colclust == None:
        raise ValueError('Valorizzare o ncols e colclust oppure v_colclust')

    if v_colclust == None:
        v_colclust = f(ncols, colclust)
    else:
        colclust = len(v_colclust)
        ncols = sum(v_colclust)
    
    if (rowclust > nrows) or (colclust > ncols):
        raise ValueError('Il numero di cluster non può essere maggiore del numero di elementi da partizionare')

    if max(rowclust, colclust) >= 2 ** min(rowclust, colclust):
        raise ValueError('fissato il numero n di clusters sulle righe il numero massimo di cluster sulle colonne'+
                         f' deve essere minore di 2^n (e viceversa): {max(rowclust, colclust)} >= 2 ^ {min(rowclust, colclust)} = {2 **min(rowclust, colclust)}')

    m = min(rowclust, colclust)
    M = max(rowclust, colclust)
    
    
    d = list(np.diag([1] * m))
    l1 = [tuple(t) for t in d]
    l2 = list(islice(filterfalse(lambda x: sum(x) == 1, product([0,1], repeat = m)), 1, M-m + 1))
    l = l1 + l2
    l.sort(key = sum)                

    if colclust > rowclust:
        l = np.array(l).T
    else:
        l = np.array(l)
    l = np.append(l, np.arange(rowclust).reshape(rowclust, 1), axis = 1)
    l = np.append(l, np.arange(colclust + 1).reshape(1, colclust + 1), axis = 0)
    V = np.repeat(l, v_colclust + [1], axis = 1)
    V = np.repeat(V, v_rowclust + [1], axis = 0)

    target_r = V[:nrows, -1]
    target_c = V[-1, :ncols]
    V = V[:nrows, :ncols]
    
    V = replaceRandom(V, noise, rng, max_attempts = max_attempts)
    #V.astype(float)
    
    return V, target_r, target_c

def f(n, nclust):
    """
    calcola il numero di elementi (righe o colonne) da mettere in ogni cluster
    
    Parametri: 
    n = numero elementi
    nclust = numero clusters
    
    Return:
    v = lista di lunghezza nclust che in posizione i riporta il numero di elementi nell'i-simo cluster
    """
    rows_per_clust = [round(n / nclust), int(n / nclust)]
    a = rows_per_clust * int((nclust + 1) / 2)
    s = sum(a[:nclust - 1])
    a[nclust - 1] = n - s
    v = a[:nclust]
    return v

def replaceRandomBase(arr, noise, rng):
    temp = np.asarray(arr)   # Cast to numpy array
    shape = temp.shape       # Store original shape
    temp = temp.flatten()    # Flatten to 1D
    inds = rng.choice(temp.size, size = round(temp.size * noise ), replace = False)   # Get random indices
    temp[inds] = (temp[inds] + 1) % 2        # Fill with something
    temp = temp.reshape(shape)                     # Restore original shape
    if len(shape) == 3:
        # check for 0-slices
        s01 = np.sum(temp, axis = (0,1))
        s02 = np.sum(temp, axis = (0,2))
        s12 = np.sum(temp, axis = (1,2))
        check = np.sum(s01 == 0) + np.sum(s02 == 0) + np.sum(s12 == 0)
    elif len(shape) == 2:
        s0 = np.sum(temp, axis = 0)
        s1 = np.sum(temp, axis = 1)
        check = np.sum(s0 ==0) + np.sum(s1 == 0)
    else:
        check = 0
        #warn ('Controllo della mancanza di sezioni tutte a 0 non effettuato', UserWarning)
    return temp, check

def replaceRandom(arr, noise, rng, max_attempts = 1):
    for i in range(max_attempts):
        temp, check = replaceRandomBase(arr, noise, rng)
        if check == 0:
            break
    if check > 0:        
        warn ('Almeno una sezione del tensore è formata da soli zeri', UserWarning)
    return temp

def CreateTensor3(nrows = None, ncols = None, nz = None, rowclust = None, colclust = None, zclust = None, noise = 0, v_rowclust = None, v_colclust = None, v_zclust = None, max_attempts = 1, random_state=None):

    """
    Scrivere qualcosa

    """
    rng = np.random.default_rng(seed = random_state)

    if (rowclust == None or nrows == None) and v_rowclust == None:
        raise ValueError('Valorizzare o nrows e rowclust oppure v_rowclust')

    if v_rowclust == None:
        v_rowclust = f(nrows, rowclust)
    else:
        rowclust = len(v_rowclust)
        nrows = sum(v_rowclust)

    if (colclust == None or ncols == None) and v_colclust == None:
        raise ValueError('Valorizzare o ncols e colclust oppure v_colclust')

    if v_colclust == None:
        v_colclust = f(ncols, colclust)
    else:
        colclust = len(v_colclust)
        ncols = sum(v_rowclust)

    if (zclust == None or nz == None) and v_zclust == None:
        raise ValueError('Valorizzare o nz e zclust oppure v_zclust')

    if v_zclust == None:
        v_zclust = f(nz, zclust)
    else:
        zclust = len(v_zclust)
        nz = sum(v_zclust)
    
    if (rowclust > nrows) or (colclust > ncols) or (zclust > nz):
        raise ValueError('Il numero di cluster non può essere maggiore del numero di elementi da partizionare')

    c = [rowclust, colclust, zclust]
    d = [(i, x) for i, x in enumerate(c)]
    v_d = [v_rowclust, v_colclust, v_zclust]
    c.sort()
    d.sort(key=lambda tup: tup[1])
    e = [(i, x[0]) for i,x in enumerate(d)]
    e.sort(key=lambda tup: tup[1])
    v_c = [v_d[x[0]] for x in d]


    V1, x1, y1 = CreateMatrix(c[1], c[2], c[1], c[2])
    Vlist = [V1]
    Vn = np.copy(V1)
    for i in range(c[0] - 1):
        Vn = np.roll(Vn, 1, axis = 1)
        yn = np.roll(y1, 1)
        Vlist.append(Vn)

    #per evitare che ci siano righe tutte a 0
    V = np.array(Vlist)

##    S = np.sum(V, axis = 0)
##    V[c[0] -1][S == 0] = 1
    
    V = np.repeat(V, v_c[1], axis = 1)
    V = np.repeat(V, v_c[2], axis = 2)
    V = np.repeat(V, v_c[0], axis = 0)

    y = np.repeat(x1, v_c[1])
    z = np.repeat(y1, v_c[2])
    x = np.arange(c[0]).repeat(v_c[0])
    target = [x, y, z]

    V = np.transpose(V, [x[0] for x in e])
    t = [target[i] for i in [x[0] for x in e]]

    V = replaceRandom(V, noise, rng, max_attempts = max_attempts)
    
    #return V, x, y, z
    return V, t[0], t[1], t[2]


def CreateTensor(n= None, nclust = None, noise = 0, max_attempts = 1, last_iteration = True, random_state=None):
    """
    n = list containing the dimnesion on each mode
    nclust = list containing the number of clusters on each mode
    noise = float, level of noise
    max_attempts: parametro per la funzione replaceRandom.
                Indica il numero massimo di volte in cui si prova a generare una matrice con rumore se nei tentativi precedenti si erano ottenute matrici con intere righe o colonne interamente a zero.


    """
    rng = np.random.default_rng(seed = random_state)

    n_modes = len(n)
    if n_modes != len(nclust):
        raise ValueError('n e nclust devono avere la stssa lunghezza')

    if n_modes == 2:
        result = CreateMatrix(n[0],n[1],nclust[0],nclust[1], noise = noise, max_attempts = max_attempts)
        V = result[0]
        t = [result[i] for i in range(1,3)] 
    elif n_modes == 3:
        result = CreateTensor3(n[0],n[1],n[2],nclust[0],nclust[1],nclust[2], noise = noise, max_attempts = max_attempts)
        V = result[0]
        t = [result[i] for i in range(1,4)] 
    else:
        v_clust=[]
        for i in range(n_modes):
            v_clust.append(f(n[i], nclust[i]))
        
        c = nclust.copy()
        d = [(i, x) for i, x in enumerate(c)]
        v_d = v_clust.copy()
        c.sort()
        d.sort(key=lambda tup: tup[1])
        e = [(i, x[0]) for i,x in enumerate(d)]
        
        e.sort(key=lambda tup: tup[1])
        v_c = [v_d[x[0]] for x in d]



            
        sol = [0] * n_modes
        if n_modes == 4:
            sol1 = [0,0,0]
            V1, sol1[0], sol1[1], sol1[2] = CreateTensor3(c[n_modes -3], c[n_modes -2], c[n_modes -1], c[n_modes -3], c[n_modes -2], c[n_modes -1])
            V1 = V1.astype('int8')
        else:
            result = CreateTensor(c[1:],c[1:], noise = 0, last_iteration = False)
            V1 = result[0]
            sol1 = result[1:][0]
            
            
        Vlist = [np.copy(V1)]
        for i in range(c[0] - 1):
            V1 = np.roll(V1, 1, axis = 1)
            Vlist.append(V1)


        V = np.array(Vlist, dtype = 'int8')
        del(V1, Vlist)



        for i in range(1,n_modes):
        
            V = np.repeat(V, v_c[i], axis = i)
            sol[i] = np.repeat(sol1[i-1], v_c[i])
        V = np.repeat(V, v_c[0], axis = 0)
        sol[0] = np.arange(c[0]).repeat(v_c[0])


        V = np.transpose(V, [x[0] for x in e])
        t = [sol[i] for i in [x[0] for x in e]]

        V = replaceRandom(V, noise, rng, max_attempts = max_attempts)




    return V, t




def modifySparsityBase(arr, sparsity, rng):
    temp = np.asarray(arr)   # Cast to numpy array
    shape = temp.shape       # Store original shape
    temp = temp.flatten()    # Flatten to 1D
    zeros = temp.size - np.sum(temp)
    new_zeros = round(sparsity * temp.size) - zeros
    actual_sparsity = zeros / temp.size
    max_sparsity = 1 - (np.max(shape)/ temp.size)
    
    if new_zeros <= 0:
        raise ValueError(f"Inserire una sparsità maggiore di quella di partenza ({actual_sparsity})")
    if sparsity > max_sparsity:
        raise ValueError(f"sparsità troppo elevata (massimo consentito: {max_sparsity}")
    ones_ind = temp * np.arange(temp.size)
    t01 = False
    if temp[0] == 1:
        ones_ind[0] = 1
        t01 = True
    ones_ind = ones_ind[ones_ind != 0]
    if t01 == True:
        ones_ind[0] = 0
        
    inds = rng.choice(ones_ind, size = new_zeros, replace = False)   # Get random indices
    temp[inds] = 0        # Fill with something
    temp = temp.reshape(shape)                     # Restore original shape

    if len(shape) == 3:
        # check for 0-slices
        s01 = np.sum(temp, axis = (0,1))
        s02 = np.sum(temp, axis = (0,2))
        s12 = np.sum(temp, axis = (1,2))
        check = np.sum(s01 == 0) + np.sum(s02 == 0) + np.sum(s12 == 0)
    elif len(shape) == 2:
        s0 = np.sum(temp, axis = 0)
        s1 = np.sum(temp, axis = 1)
        check = np.sum(s0 ==0) + np.sum(s1 == 0)
    else:
        check = 0
        warn ('Controllo della mancanza di sezioni tutte a 0 non effettuato', UserWarning)

    return temp, check

def modifySparsity(arr, sparsity, max_attempts = 1):
    for i in range(max_attempts):
        temp, check = modifySparsityBase(arr, sparsity)
        if check == 0:
            break
    if check > 0:        
        warn ('Almeno una sezione del tensore è formata da soli zeri', UserWarning)
    return temp

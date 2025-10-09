from time import time
import numpy as np
from scipy.sparse import issparse
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder


class CoClust():
    """ Fast Tensor Co-clustering with denormalized Goodman-Kruskal's Tau (Battaglia et al., 2023).

    CoStar is an algorithm created to deal with multi-view data.
    It finds automatically the best number of row / column clusters.

    Parameters
    ------------

    n_iterations : int, optional, default: 500
        The maximum number of iterations to be performed.
    
    n_iter_per_mode : int, optional, default: 1
        The maximum number of iterations per mode

    init : {'discrete', 'extract_centroids'}, optional, default: 'random'
        The initialization methods.

    k: array of int, optional (default: [20,20,20])
        The initial number of clusters per mode ([0,0,0] = discrete partition)
    
    verbose: bool, optional (default: False)
        The verbosity of the algorithm
    
    random_state: int, opional (default: None)
        The seed for the random numbers generator


    Attributes
    -----------

    labels_ : ndarray, length items per mode
        Results of the clustering on rows. `labels_[m][i]` is `c` if
        item `i` in mode m is assigned to cluster `c`. Available only after calling ``fit``.

    execution_time_ : float
        The execution time.

    References
    ----------

    * Battaglia E., et al., 2023. `Fast parameterless prototype-based co-clustering`
        Machine Learning, 2023

    """

    def __init__(self, n_iterations=500, n_iter_per_mode = 1, initialization= 'extract_centroids', k = [20,20,20], verbose = False, random_state=None):
        """
        Create the model object and initialize the required parameters.

        :type n_iterations: int
        :param n_iterations: the max number of iterations to perform
        :type n_iter_per_mode: int
        :param n_iter_per_mode: the max number of iterations per rows
        :type initialization: string
        :param initialization: the initialization method, default = 'extract_centroids'
        :type k: array of int
        :param k: number of initial clusters on all modes. 
        :type verbose: boolean
        :param verbose: if True, it prints details of the computations
        :type random_state: int | None
        :param random_state: random seed
        
        """
        self._rng = np.random.default_rng(seed=random_state)
        self.n_iterations = n_iterations
        self.n_iter_per_mode = n_iter_per_mode
        self.initialization = initialization
        self.labelencoder_ = LabelEncoder()
        self.k = np.array(k)
        self.verbose = verbose

        np.seterr(all='ignore')

    def _init_all(self, V):
        """
        Initialize all variables needed by the model.

        :param V: the dataset
        :return:
        """
        # verify that all matrices are correctly represented
        # check_array is a sklearn utility method
        self._dataset = None
        self._tmp_dataset = None

        self._dataset = check_array(V, accept_sparse='csr', ensure_2d = False, allow_nd = True, dtype=[np.int32, np.int8, np.float64, np.float32])

        self._csc_dataset = None
        if issparse(self._dataset):
            # transform also to csc
            self._csc_dataset = self._dataset.tocsc()

        # the number of modes and dimensions on each mode
        self._n = np.array(self._dataset.shape)
        self._n_modes = len(self._dataset.shape)

        # the number of row/ column clusters
        self._n_clusters = np.zeros(self._n_modes, dtype = 'int')

        # a list of n_documents (n_features) elements
        # for each document (feature) d contains the row cluster index d is associated to
        self._assignment = [np.zeros(self._n[i], 'int') for i in range(self._n_modes)]
        self._tmp_assignment = [np.zeros(self._n[i], 'int') for i in range(self._n_modes)]
        self._incidence = [np.zeros((self._n[i],self.k[i]), 'int') for i in range(self._n_modes)]

        # computation time
        self.execution_time_ = 0

        self._tot = np.sum(self._dataset)
        self._dataset = self._dataset/self._tot
        self.tau = list()
        

        if (self.initialization == 'discrete'):
            self._discrete_initialization()
        elif self.initialization == 'extract_centroids':
            self._extract_centroids_initialization()
        else:
            raise ValueError("The only valid initialization methods are: discrete, extract_centroids")

    def fit(self, V, y=None):
        """
        Fit CoClust to the provided data.

        Parameters
        -----------

        V : array-like or sparse matrix;
            shape of the matrix = (n_documents, n_features)

        y : unused parameter

        Returns
        --------

        self

        """

        # Initialization phase
        self._init_all(V)

        self._T = self._init_contingency_tensor(0)[1]
        self.tau.append(self._compute_taus())

        start_time = time()

        # Execution phase
        self._actual_n_iterations = 0 #conta il numero totale di iterazioni
        actual_n_iterations = 0 #conta come una sola iterazione un intero giro, spostamenti delle x + spostamenti delle y
        
        while actual_n_iterations < self.n_iterations:
            actual_iteration = np.zeros(self._n_modes, dtype = int)
            for m in range(self._n_modes):
                #actual_iteration_x = 0    # all'interno di ogni iterazione vengono fatte più iterazioni consecutive su ogni modo
                cont = True
                while cont:
                    # each iterations performs a move on rows

                    iter_start_time = time()

                    # perform a move within the rows partition
                    cont = self._perform_move(m)
                    #print( '############################' )
                    #self._perform_col_move()
                    #print( '############################' )

                    actual_iteration[m] += 1
                    self._actual_n_iterations +=1 
                    iter_end_time = time()

                    if actual_iteration[m] > self.n_iter_per_mode:
                        cont = False
                    if self.verbose:
                        self._T = self._init_contingency_tensor(0)[1]
                        tau = self._compute_taus()
                        self.tau.append(tau)
                        print(f'Values of tau: {tau}, for {self._n_clusters}-sized T at iteration: {actual_n_iterations} (on mode {m}).')

            
               
            if np.sum(actual_iteration) == self._n_modes:
                actual_n_iterations = self.n_iterations
            else:
                actual_n_iterations += 1
       
        end_time = time()
        execution_time = end_time - start_time    

        
        if not self.verbose:
            self._T = self._init_contingency_tensor(0)[1]
            self.tau.append(self._compute_taus())

        # clone cluster assignments and transform in lists
        self.execution_time_ = execution_time
        self.labels_ = [np.copy(self._assignment[i]) for i in range(self._n_modes)]

        if self.verbose:
            print(f'Final values of tau: {self.tau[-1]}, for {self._n_clusters}-sized T.')
            print(f'Runtime: {self.execution_time_:0.4f} seconds.')      

        return self



    def _discrete_initialization(self):

        for m in range(self._n_modes):
            # simple assign each row to a row cluster and each column of a view to a column cluster
            self._n_clusters[m] = self._n[m]
            # assign each row to a row cluster
            self._assignment[m] = np.arange(self._n[m])
            self._row_incidence = np.identity(self._n_clusters[m])
        if self.verbose:
            print(f'Initialization step for {self._n}-sized input matrix.')


    def _extract_centroids_initialization(self):
        if len(self.k) == 0 :
            raise ValueError("Parameter k is needed when initialization = 'extract_centroids'")


        if np.sum(self.k>self._n) >0:
            raise ValueError("The number of clusters must be <= the number of objects, on all dimensions")

        self._n_clusters = np.copy(self.k)
        #self._tmp_dataset = np.copy(self._dataset)
        self._tmp_dataset = self._dataset
        

        for d in range(self._n_modes):
        
            a = self._rng.choice(self._n[d], self._n_clusters[d], replace = False)
            T = self._tmp_dataset[a]
            T = T/np.sum(T)
            #S = np.repeat(np.sum(self._tmp_dataset, axis=tuple(range(1,self._tmp_dataset.ndim))).reshape(-1,1), repeats = np.prod(self._tmp_dataset.shape[1:]), axis = 1)
            #B = np.nan_to_num(self._tmp_dataset.reshape(self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:]))/np.sum(self._tmp_dataset.reshape(self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:])), axis = 0) - S)
            #B = np.nan_to_num(self._tmp_dataset.reshape(self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:]))/np.sum(self._tmp_dataset.reshape(self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:])), axis = 0) - np.repeat(np.sum(self._tmp_dataset, axis=tuple(range(1,self._tmp_dataset.ndim))).reshape(-1,1), repeats = np.prod(self._tmp_dataset.shape[1:]), axis = 1))
            
            all_tau = np.dot(np.nan_to_num(self._tmp_dataset.reshape(
                self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:]))/
                np.sum(self._tmp_dataset.reshape(self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:])), axis = 0) - 
                np.repeat(np.sum(self._tmp_dataset, axis=tuple(range(1,self._tmp_dataset.ndim))).reshape(-1,1), 
                          repeats = np.prod(self._tmp_dataset.shape[1:]), axis = 1)),T.reshape(T.shape[0],np.prod(T.shape[1:])).T)
            max_tau = np.max(np.dot(np.nan_to_num(self._tmp_dataset.reshape(
                self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:]))/
                np.sum(self._tmp_dataset.reshape(self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:])), axis = 0) - 
                np.repeat(np.sum(self._tmp_dataset, axis=tuple(range(1,self._tmp_dataset.ndim))).reshape(-1,1), 
                          repeats = np.prod(self._tmp_dataset.shape[1:]), axis = 1)),T.reshape(T.shape[0],np.prod(T.shape[1:])).T), axis = 1)
            #e_max = np.where(np.max(np.dot(np.nan_to_num(self._tmp_dataset.reshape(self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:]))/np.sum(self._tmp_dataset.reshape(self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:])), axis = 0) - np.repeat(np.sum(self._tmp_dataset, axis=tuple(range(1,self._tmp_dataset.ndim))).reshape(-1,1), repeats = np.prod(self._tmp_dataset.shape[1:]), axis = 1)),T.reshape(T.shape[0],np.prod(T.shape[1:])).T), axis = 1) == np.dot(np.nan_to_num(self._tmp_dataset.reshape(self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:]))/np.sum(self._tmp_dataset.reshape(self._tmp_dataset.shape[0],np.prod(self._tmp_dataset.shape[1:])), axis = 0) - np.repeat(np.sum(self._tmp_dataset, axis=tuple(range(1,self._tmp_dataset.ndim))).reshape(-1,1), repeats = np.prod(self._tmp_dataset.shape[1:]), axis = 1)),T.reshape(T.shape[0],np.prod(T.shape[1:])).T).T)
            #all_tau = np.dot(B,T.reshape(T.shape[0],np.prod(T.shape[1:])).T)
            #max_tau = np.max(all_tau, axis = 1)
            e_max = np.where(max_tau == all_tau.T)
            self._assignment[d][e_max[1][:self._n[d]]] = e_max[0][:self._n[d]]
            #idx = np.where(max_tau <= 0)[0]
            #self._row_assignment[idx] = np.arange(self._n_row_clusters,self._n_row_clusters+len(idx))
            if d < (self._n_modes - 1):
                self._tmp_dataset = self._tmp_dataset.transpose(tuple(np.arange(1,self._n_modes)) + tuple([0]))
            self._check_clustering(d)
        
        if self.verbose:
            print(f'Initialization step for {self._n}-sized input matrix.')



    def _check_clustering(self, dimension):
        
        self._assignment[dimension] = self.labelencoder_.fit_transform(self._assignment[dimension].astype(int))
        self._n_clusters[dimension] = len(np.unique(self._assignment[dimension]))
        self._incidence[dimension] = np.zeros((self._n[dimension], self._n_clusters[dimension]))
        self._incidence[dimension][np.arange(0,self._n[dimension],dtype='int'), self._assignment[dimension].astype(int)] = 1 
        
            
    
    def _init_contingency_tensor(self, dimension):  
        """
        Initialize the T contingency tensor
        :return:
            - the dataset with the other modes aggregated according to the current co-clustering
            - the contingency tensor
        """

        dataset = self._update_dataset(dimension)

        t = tuple([self._n_clusters[i] for i in range(self._n_modes) if i != dimension])
        #new_t = np.zeros(tuple([self._n_clusters[dimension]]) + t)
        new_t = np.dot(self._incidence[dimension].T, dataset.reshape(dataset.shape[0],np.prod(dataset.shape[1:]))).reshape(tuple([self._n_clusters[dimension]]) + t)
        
        return dataset, new_t



    def _update_dataset(self, dimension): 

        n = tuple([i for i in range(self._n_modes) if i != dimension])
        t = tuple([self._n_clusters[i] for i in range(self._n_modes) if i != dimension])
        d = tuple([self._n[i] for i in range(self._n_modes) if i != dimension])

        dataset = np.transpose(self._dataset, tuple([dimension]) + n)
        new_t = np.zeros(dataset.shape[:-1] + tuple([t[-1]]))

        for m in range(self._n_modes -1):
            dataset = np.transpose(dataset, tuple([-1]) + tuple(np.arange(self._n_modes -1)))
            new_t = np.transpose(new_t, tuple([-1]) + tuple(np.arange(self._n_modes -1)))
            new_t = np.dot(self._incidence[n[-1-m]].T, dataset.reshape(dataset.shape[0],np.prod(dataset.shape[1:]))).reshape(new_t.shape)
            #for i in range(t[-m -1]):
            #    new_t[i] = np.sum(dataset[self._assignment[n[-1-m]] == i], axis = 0)

            if m < self._n_modes -2:
                dataset = np.copy(new_t)
                new_t = np.zeros(dataset.shape[:-1] + tuple([t[-2-m]]))
                #print("dataset:", dataset.shape)
                #print("new_t:", new_t.shape)
        new_t = np.transpose(new_t, tuple([-1]) + tuple(np.arange(self._n_modes -1)))
        #print("new_t:", new_t.shape)

        return new_t




    def _perform_move(self, dimension):
        """
        Perform a single move to improve the partition on rows.

        :return:
        """
        #dataset = self._update_dataset(0)
        dataset, T = self._init_contingency_tensor(dimension)
        moves = 0

        S = np.repeat(np.sum(dataset, axis=tuple(range(1,dataset.ndim))).reshape(-1,1), repeats = np.prod(dataset.shape[1:]), axis = 1)
        B = np.nan_to_num(dataset.reshape(dataset.shape[0],np.prod(dataset.shape[1:]))/np.sum(dataset.reshape(dataset.shape[0],np.prod(dataset.shape[1:])), axis = 0) - S)
        all_tau = np.dot(B,T.reshape(T.shape[0],np.prod(T.shape[1:])).T)
        max_tau = np.max(all_tau, axis = 1)
        e_max = np.where(max_tau == all_tau.T)
        self._tmp_assignment[dimension][e_max[1][:self._n[dimension]]] = e_max[0][:self._n[dimension]]
        moves = np.sum(self._tmp_assignment[dimension] != self._assignment[dimension])
        if moves > 0:
            self._assignment[dimension] = self._tmp_assignment[dimension]
            self._check_clustering(dimension)       
        if self.verbose:
            print(f"iteration {self._actual_n_iterations}, moving mode {dimension}, n_clusters: ({self._n_clusters}), n_moves: {moves}")
        if moves ==0:
            return False
        else:
            return True


    def _compute_taus(self):
        """
        Compute the value of tau_x, tau_y and tau_z

        :return: a tuple (tau_x, tau_y, tau_z)
        """
        
        tau = np.zeros(self._n_modes)
        for j in range(self._n_modes):
            d = tuple([i for i in range(self._n_modes) if i != j])
            a = np.sum(np.nan_to_num(np.true_divide(np.sum(np.power(self._T, 2), axis = d), np.sum(self._T, axis = d)))) # scalar
            b = np.sum(np.power(np.sum(self._T, axis = j), 2)) #scalar
            tau[j] = np.nan_to_num(np.true_divide(a - b, 1 - b))

        
        #logging.debug("[INFO] a_x, a_y, a_z, b_x, b_y, b_z: {0},{1}, {2}, {3}, {4}, {5}".format(a_x, a_y, a_z, b_x, b_y, b_z))

        return tau




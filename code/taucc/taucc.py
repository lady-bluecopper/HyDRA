from time import time
import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder


class CoClust():
    """ Fast Co-clustering with denormalized Goodman-Kruskal's Tau (Battaglia et al., 2023).

    CoStar is an algorithm created to deal with multi-view data.
    It finds automatically the best number of row / column clusters.

    Parameters
    ------------

    n_iterations : int, optional, default: 500
        The maximum number of iterations to be performed.
    
    n_iter_per_mode : int, optional, default: 100
        The maximum number of iterations per mode

    init : {'random', 'discrete', 'extract_centroids'}, optional, default: 'random'
        The initialization methods.

    k: int, optional (default: 0)
        The initial number of row clusters (0 = discrete partition)
    
    l: int, optional (default: 0)
        The initial number of column clusters (0 = discrete partition)
    
    verbose: bool, optional (default: False)
        The verbosity of the algorithm
    
    random_state: int, opional (default: None)
        The seed for the random numbers generator


    Attributes
    -----------

    row_labels_ : array, length n_rows
        Results of the clustering on rows. `row_labels_[i]` is `c` if
        row `i` is assigned to cluster `c`. Available only after calling ``fit``.

    column_labels : array, length n_columns
        Results of the clustering on columns. `column_labels_[i]` is `c` if
        column `i` is assigned to cluster `c`. Available only after calling ``fit``.

    execution_time_ : float
        The execution time.

    References
    ----------

    * Battaglia E., et al., 2023. `Fast parameterless prototype-based co-clustering`
        Machine Learning, 2023

    """

    def __init__(self,
                 n_iterations=500,
                 n_iter_per_mode = 100,
                 initialization= 'random',
                 k = 30,
                 l = 30,
                 row_clusters = np.zeros(1),
                 col_clusters = np.zeros(1),
                 initial_prototypes = np.zeros(1),
                 verbose = False,
                 random_state=None):

        """
        Create the model object and initialize the required parameters.

        :type n_iterations: int
        :param n_iterations: the max number of iterations to perform
        :type n_iter_per_mode: int
        :param n_iter_per_mode: the max number of iterations per rows
        :type initialization: string
        :param initialization: the initialization method, default = 'random'
        :type k: int
        :param k: number of initial clusters on rows. 
        :type l: int
        :param l: number of initial clusters on columns. 
        :type verbose: boolean
        :param verbose: if True, it prints details of the computations
        :type random_state: int | None
        :param random_state: random seed
        
        """

        self.n_iterations = n_iterations
        self.n_iter_per_mode = n_iter_per_mode
        self.initialization = initialization
        self.k = k
        self.l = l
        self.row_clusters = row_clusters
        self.col_clusters = col_clusters
        self.initial_prototypes = initial_prototypes
        self.verbose = verbose
        self.labelencoder_ = LabelEncoder()
        self.rng = np.random.default_rng(seed = random_state)
        # these fields will be available after calling fit
        self.row_labels_ = None
        self.column_labels_ = None
        self.execution_time_ = None

        np.seterr(all='ignore')

    def _init_all(self, V):

        """
        Initialize all variables needed by the model.

        :param V: the dataset
        :return:
        """
        # verify that all matrices are correctly represented
        # check_array is a sklearn utility method
        self._dataset = check_array(V, accept_sparse='csr', dtype=[np.float64, np.float32, np.int32])
        self._csc_dataset = None
        if issparse(self._dataset):
            # transform also to csc
            self._csc_dataset = self._dataset.tocsc()
            
        # the number of documents and the number of features in the data (n_rows and n_columns)
        self._n_documents = self._dataset.shape[0]
        self._n_features = self._dataset.shape[1]

        # the number of row/ column clusters
        self._n_row_clusters = 0
        self._n_col_clusters = 0

        # a list of n_documents (n_features) elements
        # for each document (feature) d contains the row cluster index d is associated to
        self._row_assignment = np.zeros(self._n_documents)
        self._col_assignment = np.zeros(self._n_features)
        self._tmp_row_assignment = np.zeros(self._n_documents)
        self._tmp_col_assignment = np.zeros(self._n_features)

        self._row_incidence = np.zeros((self._n_documents, self.k))
        self._col_incidence = np.zeros((self._n_features, self.l))

        self._tot = np.sum(self._dataset)
        self._dataset = self._dataset/self._tot
        self.tau_x = []
        self.tau_y = []
        
        if (self.initialization == 'discrete') or (self.initialization == 'random_optimal'):
            self._discrete_initialization()
        elif self.initialization == 'random':
            self._random_initialization()
        elif self.initialization == 'extract_centroids':
            self._extract_centroids_initialization()
        else:
            raise ValueError("The only valid initialization methods are: random, discrete, extract_centroids")
        if self.verbose:
            print(f'Initialization step for ({self._n_documents},{self._n_features})-siezed input matrix.')


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

        self._T = self._init_contingency_matrix(0)[1]
        tau_x, tau_y = self.compute_taus()
        
        self.tau_x.append(tau_x)
        self.tau_y.append(tau_y)

        start_time = time()

        # Execution phase
        self._actual_n_iterations = 0
        actual_n_iterations = 0 
        
        while actual_n_iterations < self.n_iterations:
            actual_iteration_x = 0    
            cont = True
            while cont:
                # perform a move within the rows partition
                cont = self._perform_row_move()

                actual_iteration_x += 1
                self._actual_n_iterations +=1 
                
                if actual_iteration_x > self.n_iter_per_mode:
                    cont = False
                if self.verbose:
                    self._T = self._init_contingency_matrix(0)[1]
                    tau_x, tau_y = self.compute_taus()
                    self.tau_x.append(tau_x)
                    self.tau_y.append(tau_y)
                    print(f'Values of tau_x: {tau_x:0.4f} and tau_y: {tau_y:0.4f}, for ({self._n_row_clusters},{self._n_col_clusters})-sized T at iteration: {actual_n_iterations} (on rows).')
            
            
            actual_iteration_y = 0
            cont = True
            while cont:
                # perform a move within the rows partition
                cont = self._perform_col_move()

                actual_iteration_y += 1
                self._actual_n_iterations +=1 

                if actual_iteration_y > self.n_iter_per_mode:
                    cont = False
                if self.verbose:
                    self._T = self._init_contingency_matrix(1)[1]
                    tau_x, tau_y = self.compute_taus()
                    self.tau_x.append(tau_x)
                    self.tau_y.append(tau_y)
                    print(f'Values of tau_x: {tau_x:0.4f} and tau_y: {tau_y:0.4f}, for ({self._n_row_clusters},{self._n_col_clusters})-sized T at iteration: {actual_n_iterations} (on columns).')
                
                
            if (actual_iteration_x == 1) and (actual_iteration_y == 1):
                actual_n_iterations = self.n_iterations
            else:
                actual_n_iterations += 1
            

        end_time = time()

        if not self.verbose:
            self._T = self._init_contingency_matrix(1)[1]
            tau_x, tau_y = self.compute_taus()
            self.tau_x.append(tau_x)
            self.tau_y.append(tau_y)

        execution_time = end_time - start_time
        # clone cluster assignments and transform in lists
        self.row_labels_ = np.copy(self._row_assignment).tolist()
        self.column_labels_ = np.copy(self._col_assignment).tolist()
        self.execution_time_ = execution_time

        if self.verbose:
            print(f'Final values of tau_x: {self.tau_x[-1]:0.4f} and tau_y: {self.tau_y[-1]:0.4f}, for ({self._n_row_clusters},{self._n_col_clusters})-sized T.')
            print(f'Runtime: {self.execution_time_:0.4f} seconds.')        

        return self


    def _discrete_initialization(self):
        
        # simply assign each row to a row cluster and each column of a view to a column cluster
        self._n_row_clusters = self._n_documents
        self._n_col_clusters = self._n_features

        # assign each row to a row cluster
        self._row_assignment = np.arange(self._n_documents)
        self._row_incidence = np.identity(self._n_documents)

        # assign each column to a cluster
        self._col_assignment = np.arange(self._n_features)
        self._row_incidence = np.identity(self._n_documents)


    def _random_initialization(self):

        if (self.k > self._n_documents) or (self.l > self._n_features):
            raise ValueError("The number of clusters must be <= the number of objects, on both dimensions")
        if self.k == 0 :
            self._n_row_clusters = self.rng.choice(self._n_documents)
        else:
            self._n_row_clusters = self.k
        if self.l == 0:
            self._n_col_clusters = self.rng.choice(self._n_features)
        else:
            self._n_col_clusters = self.l

        # assign each row to a row cluster
        if self._n_row_clusters < self._n_documents:
            self._row_assignment = self.rng.choice(self._n_row_clusters, size = self._n_documents)
        else:
            self._row_assignment = np.arange(self._n_row_clusters).astype(int)
        # assign each column to a cluster
        
        if self._n_col_clusters < self._n_features:
            self._col_assignment = self.rng.choice(self._n_col_clusters, size = self._n_features)
        else:
            self._col_assignment = np.arange(self._n_col_clusters).astype(int)

            
        self._check_clustering(0)
        self._check_clustering(1)


    def _extract_centroids_initialization(self):
        if (self.k > self._n_documents) or (self.l > self._n_features) or (self.k <= 0) or (self.l <= 0):
            raise ValueError("The number of clusters must be <= the number of objects and greater than 0, on both dimensions")

        self._n_row_clusters = self.k
        self._n_col_clusters = self.l

        if self.k == self._n_documents:
            self._row_assignment = np.arange(self._n_documents)
            self._row_incidence = np.identity(self._n_documents)
        if self.l == self._n_features:
            self._col_assignment = np.arange(self._n_features)
            self._col_incidence = np.identity(self._n_features)
        if (self.k < self._n_documents) or (self.l < self._n_features):
            a = self.rng.choice(self._n_documents, self._n_row_clusters, replace = False)
            T = self._dataset[a]
            S = np.repeat(np.sum(self._dataset, axis = 1).reshape((-1,1)), repeats = self._dataset.shape[1], axis = 1)
            B = np.nan_to_num(self._dataset/np.sum(self._dataset, axis = 0) - S)
            all_tau = np.dot(B,T.T)
            max_tau = np.max(all_tau, axis = 1)
            e_max = np.where(max_tau == all_tau.T)
            self._row_assignment[e_max[1][:self._n_documents]] = e_max[0][:self._n_documents] # type: ignore
            idx = np.where(max_tau <= 0)[0]
            self._row_assignment[idx] = np.arange(self._n_row_clusters,self._n_row_clusters+len(idx)) # type: ignore
            self._check_clustering(0)
            
            dataset, T = self._init_contingency_matrix(1)
            
            b = self.rng.choice(self._n_features, self._n_col_clusters, replace = False)        
            T = dataset[:,b].T
            dataset = dataset.T
            S = np.repeat(np.sum(dataset, axis = 1).reshape((-1,1)), repeats = dataset.shape[1], axis = 1)
            B = np.nan_to_num(dataset/np.sum(dataset, axis = 0) - S)
            all_tau = np.dot(B,T.T)
            max_tau = np.max(all_tau, axis = 1)
            e_max = np.where(max_tau == all_tau.T)
            self._col_assignment[e_max[1][:self._n_features]] = e_max[0][:self._n_features] # type: ignore
            idx = np.where(max_tau <= 0)[0]
            self._col_assignment[idx] = np.arange(self._n_col_clusters,self._n_col_clusters+len(idx)) # type: ignore
            self._check_clustering(1)

        self._init_k = self._n_row_clusters
        self._init_l = self._n_col_clusters
        
    def _check_clustering(self, dimension):
        if dimension == 1:
            self._col_assignment = self.labelencoder_.fit_transform(self._col_assignment.astype(int)) # type: ignore
            self._n_col_clusters = len(np.unique(self._col_assignment))
            self._col_incidence = np.zeros((self._n_features, self._n_col_clusters))      
            self._col_incidence[np.arange(0,self._n_features,dtype='int'), self._col_assignment.astype(int)] = 1 # type: ignore 
        elif dimension == 0:
            self._row_assignment = self.labelencoder_.fit_transform(self._row_assignment.astype(int)) # type: ignore
            self._n_row_clusters = len(np.unique(self._row_assignment))
            self._row_incidence = np.zeros((self._n_documents, self._n_row_clusters))
            self._row_incidence[np.arange(0,self._n_documents,dtype='int'), self._row_assignment.astype(int)] = 1 # type: ignore
            
    def _init_contingency_matrix(self, dimension):
        dataset = self._update_dataset(dimension)
        #new_t = np.zeros((self._n_row_clusters, self._n_col_clusters), dtype=float)
        if dimension == 0:
            new_t = np.dot(self._row_incidence.T, dataset)   
        else:
            new_t = np.dot(dataset, self._col_incidence)   
        return dataset, new_t

    def _update_dataset(self, dimension):
        if dimension == 0:
            #new_t = np.zeros((self._n_documents, self._n_col_clusters), dtype = float)
            new_t = np.dot(self._dataset, self._col_incidence)             
        else:
            #new_t = np.zeros((self._n_row_clusters, self._n_features), dtype = float)
            new_t = np.dot(self._row_incidence.T, self._dataset)
        return new_t


    def _perform_row_move(self):

        dataset, T = self._init_contingency_matrix(0)
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S
        moves = 0
        all_tau = np.dot(dataset,B.T)
        max_tau = np.max(all_tau, axis = 1)
        e_max = np.where(max_tau == all_tau.T)
        self._tmp_row_assignment[e_max[1][:self._n_documents]] = e_max[0][:self._n_documents]
        moves = np.sum(self._tmp_row_assignment != self._row_assignment)
        if moves > 0:
            self._row_assignment = self._tmp_row_assignment
            self._check_clustering(0)
        if self.verbose:
            print(f"iteration {self._actual_n_iterations}, moving rows, n_clusters: ({self._n_row_clusters}, {self._n_col_clusters}), n_moves: {moves}")
        if moves:
            return True
        else:
            return False


    def _perform_col_move(self):

        dataset, T = self._init_contingency_matrix(1)
        T = T.T
        dataset = dataset.T
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S
        moves = 0

        all_tau = np.dot(dataset,B.T)
        max_tau = np.max(all_tau, axis = 1)
        e_max = np.where(max_tau == all_tau.T)
        self._tmp_col_assignment[e_max[1][:self._n_features]] = e_max[0][:self._n_features]
        moves = np.sum(self._tmp_col_assignment != self._col_assignment)
        if moves > 0:
            self._col_assignment = self._tmp_col_assignment
            self._check_clustering(1)
        if self.verbose:
            print(f"iteration {self._actual_n_iterations}, moving columns, n_clusters: ({self._n_row_clusters}, {self._n_col_clusters}), n_moves: {moves}")
        if moves:
            return True
        else:
            return False

    def compute_taus(self):
        tot_per_x = np.sum(self._T, 1)
        tot_per_y = np.sum(self._T, 0)
        t_square = np.power(self._T, 2)

        a_x = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 0), tot_per_y)))
        b_x = np.sum(np.power(tot_per_x, 2))
        

        a_y = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 1), tot_per_x)))
        b_y = np.sum(np.power(tot_per_y, 2))

        tau_x = np.nan_to_num(np.true_divide(a_x - b_x, 1 - b_x))
        tau_y = np.nan_to_num(np.true_divide(a_y - b_y, 1 - b_y))

        return tau_x, tau_y#, a_x, b_x

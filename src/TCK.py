from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture

from utils import DataUtils


class TCK(TransformerMixin):
    """
    implementation of the TCK algorithm by <article quote>

    :param Q: number of initializations
    :param C: maximal number of mixture components
    """

    def __init__(self, Q: int, C: int, mixture_model_params: Dict = None):
        # Model parameters
        self.Q = Q
        self.C = C
        self.q_params = {}  # Dictionary to track at each iteration q the results

        if not mixture_model_params:
            self.mixture_model_params = {'covariance_type': 'diag',
                                         'max_iter': 100,
                                         'n_init': 1,
                                         'init_params': 'kmeans',
                                         'verbose': 2}

        self.K = None  # Kernel matrix (the actual TCK)
        self.N = 0  # Amount of MTS (initialize during the fit method)

        # region randomization params

        # Time segments params
        self.T_max = -1
        self.T_min = -1

        # attributes params
        self.V_max = -1
        self.V_min = -1

        # number of data points params
        self.N_max = -1
        self.N_min = -1

        # endregion randomization params

    """
        Algorithm 2 from the article
        :param X: a 3d matrix represent MTS (num_of_mts, max_time_window, attributes)
        :param R: a 3d matrix represent the missing values of the MTS
    """

    def fit(self, X: np.ndarray, R: np.ndarray =None):
        self.initialize_kernel_matrix(X)
        self.set_randomization_fields(X)

        for q in range(self.Q):
            hyperparameters = None  # TODO: initialize according to GMM.
            time_segments_indices = self.get_iter_time_segment_indices()
            attributes_indices = self.get_iter_attributes_indices()
            mts_indices = self.get_iter_mts_indices()
            gmm_mixture_params = self.get_iter_num_of_mixtures()
            gmm_model = GaussianMixture(n_components=gmm_mixture_params,
                                        **self.mixture_model_params)

            current_subset_data = DataUtils.get_3d_array_subset(X, mts_indices,
                                                                time_segments_indices,
                                                                attributes_indices)
            current_subset_mask = DataUtils.get_3d_array_subset(R, mts_indices,
                                                                time_segments_indices,
                                                                attributes_indices)

            current_data_for_train = current_subset_data[current_subset_mask == 1]
            gmm_model.fit(current_data_for_train)
            posterior_probabilities = gmm_model.predict_proba(X)
            # Theta params
            means = gmm_model.means_
            covariances = gmm_model.covariances_

            self.update_q_params(q, hyperparameters, time_segments_indices, attributes_indices,
                                 mts_indices, gmm_mixture_params, gmm_model)

            # TODO: this is the right way to update?
            self.K += posterior_probabilities.dot(posterior_probabilities.T)

        return self

    # region initialization

    def initialize_kernel_matrix(self, X: np.ndarray):
        self.N = X.shape[0]
        kernel_matrix_shape = (self.N, self.N)
        self.K = np.zeros(kernel_matrix_shape)

    def set_randomization_fields(self, X: np.ndarray):
        # TODO: consult with Tamir about the initialization
        self.T_min = 2  # We want at least 2
        # TODO: more robust way, at most time setmen
        self.T_max = 10

        self.V_max = X.shape[2]  # Number of attributes
        self.V_min = 1

        # This just for safety, doesn't suppose to happen
        if not self.N:
            self.N = X.shape[0]
        self.N_min = int(0.1 * self.N)  # At least 10% of the data
        self.N_max = self.N  # At most all the data

    # endregion initialization

    def get_iter_time_segment_indices(self):
        # TODO: check if this is what I want
        T = np.random.randint(self.T_min, self.T_max)
        time_window = np.arange(T)

        return time_window

    def get_iter_attributes_indices(self):
        V = np.random.randint(self.V_min, self.V_max)
        attributes_subset_indices = np.random.choice(np.arange(self.V_max), V, replace=False)

        return attributes_subset_indices

    def get_iter_mts_indices(self):
        N = np.random.randint(self.N_min, self.N_max)
        mts_subset_indices = np.random.choice(np.arange(self.N_max), N, replace=False)

        return mts_subset_indices

    def get_iter_num_of_mixtures(self):
        return np.random.randint(2, self.C)

    """
    :param q: current iteration
    """
    def update_q_params(self, q: int,
                        hyperparameters,
                        time_segments_indices,
                        attributes_indices,
                        mts_indices,
                        gmm_mixture_params,
                        gmm_model):
        self.q_params[q]['hyperparameters'] = hyperparameters
        self.q_params[q]['time_segments_indices'] = time_segments_indices
        self.q_params[q]['attributes_indices'] = attributes_indices
        self.q_params[q]['mts_indices'] = mts_indices
        self.q_params[q]['gmm_mixture_params'] = gmm_mixture_params
        self.q_params[q]['gmm_model'] = gmm_model

    """
    Algorithm 3 from the article
    """

    def transform(self, X):
        return self.K


from scipy.stats import norm


class MAP_EM_GMM(TransformerMixin):
    """
    :param a0: Hyperparameter for the kernel based mean prior
    :param b0 : Hyperparameter for the kernel based mean prior
    :param N0: Hyperparameter for the inverse gamma distribution on the std prior
    :param C: Number of GMMs
    :param num_iter: number of maximum iteration (default: 20)
    """
    def __init__(self, a0: float,
                 b0: float,
                 N0: int,
                 C: int,
                 num_iter: int = 20):
        self.a0 = a0
        self.b0 = b0
        self.N0 = N0
        self.C = C
        self.num_iter = num_iter
        self.N = None  # Number of MTS instances
        self.V = None  # Number of attributes
        self.T = None  # Number of time segments
        self.posteriors = None  # shape CxNxT
        self.theta = None  # Shape Cx1
        self.mu = None  # Shape CxTxV
        self.s2 = None  # Shape CxV

    """
        Algorithm 1 from the article
        :param X: a 3d matrix represent MTS (NxTxV)
        :param R: a 3d matrix represent the missing values of the MTS
    """
    def fit(self, X: np.ndarray,
            R: np.ndarray = None):
        # TODO: change to more indicative names, for now follow the matlab code notations
        self.N = X.shape[0]
        self.T = X.shape[1]
        self.V = X.shape[2]
        # TODO: initialize correctly
        self.posteriors = np.zeros((self.C, self.N, self.T))
        self.theta = self.init_cluster_priors()
        self.mu = self.init_cluster_means()
        self.s2 = self.init_cluster_variance()

        if not R:
            R = np.ones_like(X)

        for i in range(self.num_iter):
            self.expectation_step(X, R)
            self.maximization_step()

        return self

    def init_cluster_priors(self):
        return np.ones(self.C) / self.C

    def init_cluster_means(self):
        # TODO: implement the right way e.g. according to the algorithm
        return np.zeros((self.C, self.T, self.V))  # Cluster means

    def init_cluster_variance(self):
        # TODO: implement the right way, e.g. according to the algorithm
        return np.ones((self.C, self.V))  # cluster variances

    def expectation_step(self, X: np.ndarray, R: np.ndarray):
        new_posterior = np.zeros_like(self.posteriors)

        for c in range(self.C):
            for n in range(self.N):
                current_MTS = X[n]
                current_missing_indication = R[n]
                mu = self.mu[c]
                s2 = self.s2[c]
                theta = self.theta[c]

                new_posterior[c, n] = self.evalute_posterior()

    def maximization_step(self):
        pass



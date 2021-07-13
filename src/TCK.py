from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture


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
        :param X: a 3d matrix represent MTS
        :param R: a 3d matrix reoresent the missing values of the MTS
    """

    def fit(self, X: np.ndarray, R=None):
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

            current_data = X[mts_indices, attributes_indices, time_segments_indices]

            gmm_model.fit(current_data)
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

        self.V_max = X.shape[1]  # Number of attributes
        self.V_min = 1

        self.N_max = -1

        # This just for safety, doesn't suppose to happen
        if not self.N:
            self.N = X.shape[0]
        self.N_min = 0.1 * self.N  # At least 10% of the data
        self.N_max = self.N  # At most all the data

    # endregion initialization

    def get_iter_time_segment_indices(self):
        # TODO: check if this is what I want
        T = np.random.randint(self.T_min, self.T_max)
        time_window = np.arange(T)

        return time_window

    def get_iter_attributes_indices(self):
        V = np.random.randint(self.V_min, self.V_max)
        attributes_subset_indices = np.random.choice(np.arange(self.V_max), V)

        return attributes_subset_indices

    def get_iter_mts_indices(self):
        V = np.random.randint(self.N_min, self.N_max)
        mts_subset_indices = np.random.choice(np.arange(self.V_max), V)

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

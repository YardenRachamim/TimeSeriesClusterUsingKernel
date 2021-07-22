from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from GMM_MAP_EM import GMM_MAP_EM
from utils import DataUtils


class TCK(TransformerMixin):
    """
    implementation of the TCK algorithm by <article quote>

    :param Q: number of initializations
    :param C: maximal number of mixture components
    """

    def __init__(self, Q: int, C: int):
        # Model parameters
        self.Q = Q
        self.C = C

        np.random.seed (7)   # only once for TCK life to ensure that randoms or permutations do not repeat.

        self.q_params = {}  # Dictionary to track at each iteration q the results

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

    def fit(self, X: np.ndarray, R: np.ndarray = None):
        self.N = X.shape[0]
        self.K = DataUtils.initialize_empty_kernel_matrix(X)
        self.set_randomization_fields(X)

        if R is None:
            R = np.ones_like(X)

        for q in range(self.Q):
            # TODO: can be pararalized for performance
            hyperparameters = self.get_iter_hyper()
            time_segments_indices = self.get_iter_time_segment_indices()
            attributes_indices = self.get_iter_attributes_indices()
            mts_indices = self.get_iter_mts_indices()
            C = self.get_iter_num_of_mixtures()
            gmm_map_em_model = GMM_MAP_EM(a0=hyperparameters['a0'],
                                          b0=hyperparameters['b0'],
                                          N0=hyperparameters['N0'],
                                          C=C)
            gmm_model = SubsetGmmMapEm(gmm_map_em_model,
                                       mts_indices,
                                       time_segments_indices,
                                       attributes_indices)

            gmm_model.fit(X, R)
            posterior_probabilities = gmm_model.transform(X, R)

            # # Theta params
            # means = gmm_model.mu
            # covariances = gmm_model.s2

            self.update_q_params(q, hyperparameters, time_segments_indices, attributes_indices,
                                 mts_indices, C, gmm_model)

            self.K += posterior_probabilities.T @ posterior_probabilities

        return self

    # region initialization
    def set_randomization_fields(self, X: np.ndarray):
        # The parameters are initialized according to section 4.2 in the article
        self.T_min = 6
        self.T_max = X.shape[1]  # Number of time segments

        self.V_max = X.shape[2]  # Number of attributes
        self.V_min = 2

        # This just for safety, doesn't suppose to happen
        if not self.N:
            self.N = X.shape[0]
        self.N_min = int(0.8 * self.N)  # At least 80% of the data
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

    def get_iter_hyper(self):
        # The parameters are initialized according to section 4.2 in the article
        a0 = np.random.uniform(0.001, 1)
        b0 = np.random.uniform(0.005, 0.2)
        N0 = np.random.uniform(0.001, 0.2)

        return {'a0': a0, 'b0': b0, 'N0': N0}



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
        self.q_params[q] = {
            'hyperparameters': hyperparameters,
            'time_segments_indices': time_segments_indices,
            'attributes_indices': attributes_indices,
            'mts_indices': mts_indices,
            'gmm_mixture_params': gmm_mixture_params,
            'gmm_model': gmm_model
        }

    """
    Algorithm 3 from the article
    """
    def transform(self, X: np.ndarray,
                  R: np.ndarray = None) -> np.ndarray:
        if R is None:
            R = np.ones_like(X)

        K = DataUtils.initialize_empty_kernel_matrix(X)

        for q in range(self.Q):
            q_params = self.q_params[q]
            gmm_model = q_params['gmm_model']
            posterior_probabilities = gmm_model.transform(X, R)

            K += posterior_probabilities.T @ posterior_probabilities

        return K


class SubsetGmmMapEm(TransformerMixin):
    def __init__(self, gmm_map_em_model: GMM_MAP_EM,
                 mts_indices: np.ndarray,
                 time_segments_indices: np.ndarray,
                 attributes_indices: np.ndarray):
        self.gmm_map_em_model = gmm_map_em_model
        self.mts_indices = mts_indices
        self.time_segments_indices = time_segments_indices
        self.attributes_indices = attributes_indices

        self.X_shape = None

    def fit(self, X, R):
        self.X_shape = X.shape
        if X.shape != R.shape:
            error_msg = f"X and R are not of same shape"

            raise Exception(error_msg)

        current_subset_data = DataUtils.get_3d_array_subset(X,
                                                            self.mts_indices,
                                                            self.time_segments_indices,
                                                            self.attributes_indices)
        current_subset_mask = DataUtils.get_3d_array_subset(R,
                                                            self.mts_indices,
                                                            self.time_segments_indices,
                                                            self.attributes_indices)

        self.gmm_map_em_model.fit(current_subset_data, current_subset_mask)

    def transform(self, X, R) -> np.ndarray:
        if X.ndim < 3:
            X = X[None, :, :]
        if R is None:
            R = np.ones_like(X)
        if R.ndim < 3:
            R = R[None, :, :]
        is_valid_shape = (X.shape[1] == self.X_shape[1]) and (X.shape[2] == self.X_shape[2])
        if not is_valid_shape:
            error_msg = "X shape is not valid"

            raise Exception(error_msg)

        current_subset_data = DataUtils.get_3d_array_subset(X,
                                                            np.arange(X.shape[0]),
                                                            self.time_segments_indices,
                                                            self.attributes_indices)
        current_subset_mask = DataUtils.get_3d_array_subset(R,
                                                            np.arange(R.shape[0]),
                                                            self.time_segments_indices,
                                                            self.attributes_indices)

        return self.gmm_map_em_model.transform(current_subset_data, current_subset_mask)

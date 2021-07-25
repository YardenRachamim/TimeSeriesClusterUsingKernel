from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

from GMM_MAP_EM import GMM_MAP_EM
from utils import DataUtils


class TCK(TransformerMixin):
    """
    implementation of the TCK algorithm by <article quote>

    :param Q: number of initializations
    :param C: maximal number of mixture components
    """

    def __init__(self, Q: int, C: int, verbose=1):
        # Model parameters
        log_level = logging.INFO
        if verbose == 1:
            log_level = logging.DEBUG
        self.logger = self.set_logger(log_level)

        self.Q = Q
        self.C = C

        # TODO: delete
        np.random.seed(8)   # only once for TCK life to ensure that randoms or permutations do not repeat.

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

    def set_logger(self, log_level: int) -> logging.Logger:
        logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(funcName)s :: %(lineno)d '
                                   ':: %(message)s', level=log_level)

        return logging.getLogger("TCK")

    """
        Algorithm 2 from the article
        :param X: a 3d matrix represent MTS (num_of_mts, max_time_window, attributes)
        :param R: a 3d matrix represent the missing values of the MTS
    """
    def fit(self, X: np.ndarray, R: np.ndarray = None):
        self.N = X.shape[0]
        self.K = np.zeros((self.N, self.N))
        # TODO: delete
        test_K = self.K.copy()

        self.set_randomization_fields(X)

        if R is None:
            R = np.ones_like(X)

        params = []
        for q in range(self.Q):
            self.logger.info(f"q={q+1}/{self.Q}")

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


            self.logger.info(f"q params are: C={C}, a0={hyperparameters['a0']:.3f}, b0={hyperparameters['b0']:.3f},"
                             f" N0={hyperparameters['N0']:.3f},"
                             f" X.shape={mts_indices.shape[0], time_segments_indices.shape[0], attributes_indices.shape[0]}")
            gmm_model.fit(X, R)
            posterior_probabilities = gmm_model.transform(X, R)
            self.update_q_params(q, hyperparameters, time_segments_indices, attributes_indices,
                                 mts_indices, C, gmm_model, posterior_probabilities)

            # # Theta params
            # means = gmm_model.mu
            # covariances = gmm_model.s2
            # B = np.zeros_like(self.K)
            # for c in range(posterior_probabilities.shape[0]):
            #     A = np.tile(posterior_probabilities[c], (posterior_probabilities[c].shape[0], 1))
            #     B = B + (A.T @ A)
            #
            # self.K += B
            self.K += (posterior_probabilities.T @ posterior_probabilities)

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
        T = np.random.randint(self.T_min, self.T_max + 1)
        time_window = np.arange(T)

        return time_window

    def get_iter_attributes_indices(self):
        V = np.random.randint(self.V_min, self.V_max + 1)
        attributes_subset_indices = np.random.choice(np.arange(self.V_max), V, replace=False)

        return attributes_subset_indices

    def get_iter_mts_indices(self):
        N = np.random.randint(self.N_min, self.N_max + 1)
        mts_subset_indices = np.random.choice(np.arange(self.N_max), N, replace=False)

        return mts_subset_indices

    def get_iter_num_of_mixtures(self):
        return np.random.randint(2, self.C+1)

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
                        gmm_model,
                        posterior_probabilities):
        self.q_params[q] = {
            'hyperparameters': hyperparameters,
            'time_segments_indices': time_segments_indices,
            'attributes_indices': attributes_indices,
            'mts_indices': mts_indices,
            'gmm_mixture_params': gmm_mixture_params,
            'gmm_model': gmm_model,
            'posterior_probabilities': posterior_probabilities
        }

    """
    Algorithm 3 from the article
    """
    def transform(self, X: np.ndarray,
                  R: np.ndarray = None) -> (np.ndarray, np.ndarray):
        if R is None:
            R = np.ones_like(X)

        K_star = np.zeros((self.N, X.shape[0]))
        K_test = np.zeros((X.shape[0], X.shape[0]))

        for q in range(self.Q):
            q_params = self.q_params[q]
            gmm_model = q_params['gmm_model']
            q_posterior = q_params['posterior_probabilities']
            current_posterior = gmm_model.transform(X, R)

            K_star += (q_posterior.T @ current_posterior)
            K_test += (current_posterior.T @ current_posterior)

        return K_star, K_test


class SubsetGmmMapEm(TransformerMixin):
    """
    helper class for the TCK algorithm to represent GMM_MAP_EMM for subsets of data
    """
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

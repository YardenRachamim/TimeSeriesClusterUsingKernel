import multiprocessing
from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import logging.handlers

from GMM_MAP_EM import GMM_MAP_EM
from utils import DataUtils, TCKUtils
import multiprocessing as mp
from itertools import product


class TCK(TransformerMixin):
    """
    implementation of the TCK algorithm by <article quote>

    :param Q: number of initializations
    :param C: maximal number of mixture components
    """

    logger = None

    def __init__(self, Q: int, C: int, verbose=1, n_jobs=1):
        # Model parameters
        log_level = logging.INFO
        if verbose == 1:
            log_level = logging.DEBUG
        logger = TCKUtils.set_logger(TCK.__name__, log_level)

        self.n_jobs = n_jobs
        self.Q = Q
        self.C = C

        # TODO: delete
        np.random.seed(8)  # only once for TCK life to ensure that randoms or permutations do not repeat.

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
        self.K = np.zeros((self.N, self.N))

        self.set_randomization_fields(X)

        if R is None:
            R = np.ones_like(X)

        params = []
        ranges = list(product(range(self.Q), range(2, self.C + 1)))
        for i, (q, c) in enumerate(ranges):
            # TODO: can be pararalized for performance
            hyperparameters = TCKUtils.get_random_gmm_hyperparameters()
            time_segments_indices = self.get_iter_time_segment_indices()
            attributes_indices = self.get_iter_attributes_indices()
            mts_indices = self.get_iter_mts_indices()
            C = c
            gmm_map_em_model = GMM_MAP_EM(a0=hyperparameters['a0'],
                                          b0=hyperparameters['b0'],
                                          N0=hyperparameters['N0'],
                                          C=C)
            gmm_model = SubsetGmmMapEm(gmm_map_em_model,
                                       mts_indices,
                                       time_segments_indices,
                                       attributes_indices)

            log_msg = (f"({i + 1}/{len(ranges)}): q params are: q={q}, C={C}, a0={hyperparameters['a0']:.3f},"
                       f" b0={hyperparameters['b0']:.3f},"
                       f" N0={hyperparameters['N0']:.3f},"
                       f" X.shape={mts_indices.shape[0], time_segments_indices.shape[0], attributes_indices.shape[0]}")

            params.append((i, gmm_model, X, R, log_msg))

            self.update_q_params(i, hyperparameters, time_segments_indices, attributes_indices,
                                 mts_indices, C)

        processes = self.n_jobs
        if self.n_jobs == -1:
            processes = multiprocessing.cpu_count() - 1

        with mp.Pool(processes=processes) as pool:
            res = pool.starmap(TCKUtils.single_fit, params)

        for i, trained_model in res:
            posterior_probabilities = trained_model.transform(X, R)
            self.q_params[i]['posterior_probabilities'] = posterior_probabilities
            self.q_params[i]['gmm_model'] = trained_model
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

    def get_iter_num_of_mixtures(self, current_iter_num: int):
        return max(self.C - current_iter_num, 2)

    """
    :param q: current iteration
    """

    def update_q_params(self, q: int,
                        hyperparameters,
                        time_segments_indices,
                        attributes_indices,
                        mts_indices,
                        gmm_mixture_params):
        self.q_params[q] = {
            'hyperparameters': hyperparameters,
            'time_segments_indices': time_segments_indices,
            'attributes_indices': attributes_indices,
            'mts_indices': mts_indices,
            'gmm_mixture_params': gmm_mixture_params
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

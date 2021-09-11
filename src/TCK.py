import multiprocessing
from typing import Union, Callable, List, Iterable, Set, Dict
from collections.abc import Iterable as Iter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances

import logging
import logging.handlers

from GMM_MAP_EM import GMM_MAP_EM, HMM_GMM
from utils import DataUtils, TCKUtils
import multiprocessing as mp
from itertools import product
from functools import reduce, partial
from scipy.spatial.distance import jensenshannon, cdist


class TCK(TransformerMixin):
    """
    implementation of the TCK algorithm by <article quote>

    :param Q: number of initializations
    :param C: maximal number of mixture components
    """

    logger = TCKUtils.set_logger("TCK", logging.INFO)
    VALID_MAX_FEATURE_VALS = {'all', 'sqrt', 'log2'}
    VALID_SIMILARITY_FUNCTION_VALS = {'jensenshannon',
                                      'additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian',
                                      'sigmoid', 'cosine'}
    VALID_MODELS = {'HMM', 'GMM'}

    def __init__(self, Q: int, C: int,
                 max_features: str = 'all',
                 similarity_function: Union[str, Iterable[str]] = 'linear',
                 verbose=1, n_jobs=1, single_model_num_iter: int = 20, model='GMM'):
        # Model parameters

        if verbose == 1:
            TCK.logger.setLevel(logging.INFO)
        else:
            TCK.logger.setLevel(logging.WARN)

        self.n_jobs = n_jobs
        self.Q = Q
        self.C = C
        self.max_features = None
        self.set_max_features(max_features)
        self.similarity_function = self.set_similarity_function(similarity_function)
        self.q_params = {}  # Dictionary to track at each iteration q the results
        self.ranges = None
        self.iter_ranges = None
        self.total_iters = None
        self.iter = None

        self.K = None  # Kernel matrix (the actual TCK)
        self.N = 0  # Amount of MTS (initialize during the fit method)
        self.T = 0  # Longest time segment (initialize during the fit method)
        self.V = 0  # Number of attributes(initialize during the fit method)
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

        # region model params
        self.single_num_iter = single_model_num_iter
        # TODO: add validation
        self.model = model
        # endregion model params

    """
        Algorithm 2 from the article
        :param X: a 3d matrix represent MTS (num_of_mts, max_time_window, attributes)
        :param R: a 3d matrix represent the missing values of the MTS
    """

    def fit(self, X: Union[np.ndarray, List[np.ndarray]],
            R: np.ndarray = None,
            warm_start: bool = False):
        if not warm_start:
            self.N = len(X)
            self.T = max([x.shape[0] for x in X])
            self.V = X[0].shape[1]
            self.K = np.zeros((self.N, self.N))
            self.set_randomization_fields(X)

            if R is None:
                R = np.ones_like(X)

            self.ranges = set(product(range(self.Q), range(2, self.C + 1)))
            self.iter_ranges = self.ranges
            self.iter = 0
        else:
            # Else warm start is on
            #  #(all iteration) - #(already run)
            self.iter_ranges = set(product(range(self.Q), range(2, self.C + 1))) - self.ranges
            # Stay with only the new ranges
            self.ranges = self.iter_ranges.union(self.ranges)

            if (self.N != X.shape[0]) or (self.T != X.shape[1]) or (self.V != X.shape[2]):
                raise Exception(f"expected shape {(self.N, self.T, self.V)} but got {X.shape} instead")

        self.total_iters = len(self.ranges)
        processes = self.set_processes()

        with mp.pool.ThreadPool(processes=processes) as pool:
            sequence_lengths = [x.shape[0] for x in X]
            single_gmm_fit_results = []

            for i, (q, c) in enumerate(self.iter_ranges):
                self.iter += 1
                hyperparameters = TCKUtils.get_random_gmm_hyperparameters()
                time_segments_indices = self.get_iter_time_segment_indices()
                attributes_indices = self.get_iter_attributes_indices()
                mts_indices = self.get_iter_mts_indices()
                C = c
                if self.model == 'GMM':
                    base_model = GMM_MAP_EM(a0=hyperparameters['a0'],
                                            b0=hyperparameters['b0'],
                                            N0=hyperparameters['N0'],
                                            C=C)

                elif self.model == 'HMM':
                    base_model = HMM_GMM(C=C,
                                         num_iter=self.single_num_iter)
                    time_segments_indices = np.array(sequence_lengths)

                model = SubsetDataModel(base_model,
                                        mts_indices,
                                        time_segments_indices,
                                        attributes_indices)

                self.update_q_params(self.iter, hyperparameters, time_segments_indices, attributes_indices,
                                     mts_indices, C)

                args = (self.iter, model, X, R)
                single_gmm_fit_results.append(pool.apply_async(TCK.single_fit, args))

                # TODO: this is a quick and dirty workaround, dont leave it this way!
                time_segments_indices = np.array([None] * max([x.shape[0] for x in X]))
                TCK.logger.info(
                    f"({self.iter}/{self.total_iters}): q params are: q={q}, C={C}, a0={hyperparameters['a0']:.3f},"
                    f" b0={hyperparameters['b0']:.3f},"
                    f" N0={hyperparameters['N0']:.3f},"
                    f" X.shape={mts_indices.shape[0], time_segments_indices.shape[0], attributes_indices.shape[0]}")

            for r in single_gmm_fit_results:
                i, trained_gmm_model = r.get()
                TCK.logger.info(f"fiinished training GMM number ({i}/{self.total_iters})")
                current_posterior = trained_gmm_model.transform(X, R)
                self.q_params[i]['posterior_probabilities'] = current_posterior
                self.q_params[i]['gmm_model'] = trained_gmm_model

            self.K = self.transform(X, R, use_n_jobs=False)

        return self

    def set_params(self, **kwargs):
        # Must be called before fit while warm_Start=True
        if 'Q' in kwargs.keys():
            self.Q = kwargs['Q']
        if 'C' in kwargs.keys():
            self.C = kwargs['C']
        if 'n_jobs' in kwargs.keys():
            self.n_jobs = kwargs['n_jobs']

    @staticmethod
    # TODO: add similarity_function params
    def calculate_similarity(i: int,
                             p: Union[np.ndarray, Callable],
                             q: Union[np.ndarray, Callable],
                             similarity_functions: Set[str],
                             gamma: float = None):
        # First init arguments of the distributions
        if callable(p):
            p = p()
        if callable(q):
            q = q()

        # Second init matrix
        K = np.zeros((p.shape[0], q.shape[0]))

        if gamma is None:
            gamma = 1 / p.shape[1]

        # Calculate kernel with many methods(ensemble of kernels) or one
        for similarity_function in similarity_functions:
            if similarity_function == 'jensenshannon':
                #  distance measure to kernel -> exp(-D * gamma)
                K += np.exp(-gamma * (cdist(p, q, 'jensenshannon')))

                # Handle null values using the minimum value of the kernel matrix
                K[np.isnan(K)] = 0
            else:
                K += pairwise_kernels(p, q, metric=similarity_function)

        # TODO: check that K is a kernel.

        return i, K

    @staticmethod
    def single_fit(i: int,
                   gmm_model,
                   X: np.ndarray,
                   R: np.ndarray):
        gmm_model.fit(X, R)

        return i, gmm_model

    # region initialization
    def set_max_features(self, max_features):
        if max_features not in TCK.VALID_MAX_FEATURE_VALS:
            raise Exception(f"'{max_features}' is not valid for max_features please use {TCK.VALID_MAX_FEATURE_VALS}")

        self.max_features = max_features

    def set_similarity_function(self, similarity_function: Union[str, Iterable[str]]):
        f = None
        if type(similarity_function) is str:
            f = {similarity_function}
        elif isinstance(similarity_function, Iterable):
            f = set(similarity_function)

        if not f.issubset(TCK.VALID_SIMILARITY_FUNCTION_VALS):
            raise Exception(f"'{similarity_function}' is not valid for similarity_function "
                            f"please use {TCK.VALID_SIMILARITY_FUNCTION_VALS}")

        return f

    def set_processes(self):
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count() - 1

        return self.n_jobs

    def set_randomization_fields(self, X: np.ndarray):
        # The parameters are initialized according to section 4.2 in the article
        self.T_min = 6
        self.T_max = min(self.T, 25)  # Number of time segments

        self.V_min = 2
        if self.max_features == 'sqrt':
            self.V_max = max(2, np.ceil(np.sqrt(self.V)))
        elif self.max_features == 'log2':
            self.V_max = max(2, np.ceil(np.log2(self.max_features)))
        elif self.max_features == 'all':
            self.V_max = self.V

        # This just for safety, doesn't suppose to happen
        if not self.N:
            self.N = X.shape[0]
        self.N_min = int(0.8 * self.N)  # At least 80% of the data
        self.N_max = self.N  # At most all the data

        TCK.logger.info(f"randomization fields are:"
                        f"\n(N_min, N_max)=({self.N_min}, {self.N_max})"
                        f"\n(T_min, T_max)=({self.T_min}, {self.T_max})"
                        f"\n(V_min, V_max)=({self.V_min}, {self.V_max})")

    # endregion initialization

    # region iteration arguments
    def get_iter_time_segment_indices(self):
        # # TODO: check if this is what I want
        # T = np.ceil(self.T_max / np.ceil(self.T_max / 25))
        # t1 = 0
        # t2 = np.finfo('float64').max
        #
        # while (t2 - t1 > T) | (t2 - t1 == 0):
        #     t1 = np.random.randint(1, self.T_max - self.T_min + 1)
        #     t2 = np.random.randint(t1 + self.T_min - 1, min(self.T_max, (t1 + self.T_max - 1)))
        T_size = np.random.randint(self.T_min, self.T_max + 1)
        T_start = np.random.randint(0, self.T - T_size + 1)

        return np.arange(T_start, T_start + T_size)

    def get_iter_attributes_indices(self):
        V_size = np.random.randint(self.V_min, self.V_max + 1)
        attributes_subset_indices = np.random.choice(np.arange(self.V), V_size, replace=False)

        return attributes_subset_indices

    def get_iter_mts_indices(self):
        N_size = np.random.randint(self.N_min, self.N_max + 1)
        mts_subset_indices = np.random.choice(np.arange(self.N), N_size, replace=False)

        return mts_subset_indices

    def update_q_params(self, i: int,
                        hyperparameters,
                        time_segments_indices,
                        attributes_indices,
                        mts_indices,
                        gmm_mixture_params):
        self.q_params[i] = {
            'hyperparameters': hyperparameters,
            'time_segments_indices': time_segments_indices,
            'attributes_indices': attributes_indices,
            'mts_indices': mts_indices,
            'gmm_mixture_params': gmm_mixture_params
        }

    # endregion iteration arguments

    """
    Algorithm 3 from the article
    """

    def transform(self, X: np.ndarray,
                  R: np.ndarray = None,
                  use_n_jobs: bool = False) -> np.ndarray:
        if R is None:
            R = np.ones_like(X)

        K_star = np.zeros((self.N, X.shape[0]))

        if use_n_jobs:
            processes = self.set_processes()
        else:
            processes = 1

        with mp.pool.ThreadPool(processes=processes) as pool:
            results = []

            for i in self.q_params.keys():
                q_params = self.q_params[i]
                gmm_model = q_params['gmm_model']
                q_posterior = q_params['posterior_probabilities']
                current_posterior = partial(gmm_model.transform, X, R)
                args = (i, q_posterior, current_posterior, self.similarity_function)
                results.append(partial(TCK.calculate_similarity, *args))

                TCK.logger.info(f"({i}/{self.total_iters}) start transforming")

            for i, K in pool.imap(partial.__call__, results):
                K_star += K
                TCK.logger.info(f"(finished transforming {i}/{len(self.ranges)})")

        return K_star

    def change_similarity_calculation(self, similarity_function):
        # TODO: implement
        pass


class SubsetDataModel(TransformerMixin):
    """
    helper class for the TCK algorithm to represent GMM_MAP_EMM for subsets of data
    """

    def __init__(self, base_model: Union[HMM_GMM, GMM_MAP_EM],
                 mts_indices: np.ndarray,
                 time_segments_indices: np.ndarray,
                 attributes_indices: np.ndarray):
        self.base_model = base_model
        self.mts_indices = mts_indices
        self.time_segments_indices = time_segments_indices
        self.attributes_indices = attributes_indices

        self.X_shape = None

    def fit(self, X, R):
        self.X_shape = X.shape
        if X.shape != R.shape and len(X.shape) > 1:
            error_msg = f"X and R are not of same shape"

            raise Exception(error_msg)

        if isinstance(self.base_model, GMM_MAP_EM):
            current_subset_data = DataUtils.get_3d_array_subset(X,
                                                                self.mts_indices,
                                                                self.time_segments_indices,
                                                                self.attributes_indices)
            current_subset_mask = DataUtils.get_3d_array_subset(R,
                                                                self.mts_indices,
                                                                self.time_segments_indices,
                                                                self.attributes_indices)
        elif isinstance(self.base_model, HMM_GMM):
            current_X = [X[index] for index in self.mts_indices]
            current_subset_data = [x[:, self.attributes_indices] for x in current_X]

            current_subset_mask = None
        else:
            raise Exception(f"base_model is of type {type(self.base_model)} which is not valid in this context")

        self.base_model.fit(current_subset_data, current_subset_mask)

    def transform(self, X, R) -> np.ndarray:
        if X.ndim == 2:
            X = X[None, :, :]
        if R is None:
            R = np.ones_like(X)
        if R.ndim == 2:
            R = R[None, :, :]
        is_valid_shape = (X.ndim != 3) or (X.shape[1] == self.X_shape[1]) and (X.shape[2] == self.X_shape[2])
        if not is_valid_shape:
            error_msg = "X shape is not valid"

            raise Exception(error_msg)

        if isinstance(self.base_model, GMM_MAP_EM):
            current_subset_data = DataUtils.get_3d_array_subset(X,
                                                                np.arange(X.shape[0]),
                                                                self.time_segments_indices,
                                                                self.attributes_indices)
            current_subset_mask = DataUtils.get_3d_array_subset(R,
                                                                np.arange(R.shape[0]),
                                                                self.time_segments_indices,
                                                                self.attributes_indices)
        elif isinstance(self.base_model, HMM_GMM):
            current_subset_data = [x[:, self.attributes_indices] for x in X]

            current_subset_mask = None
        else:
            raise Exception(f"base_model is of type {type(self.base_model)} which is not valid in this context")

        return self.base_model.transform(current_subset_data, current_subset_mask)


# For back compatibility
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
import numpy as np
import logging
import threading
from datetime import datetime
import sys


class DataUtils:
    @staticmethod
    def get_3d_array_subset(arr: np.ndarray,
                            first_dim_indices: np.ndarray,
                            second_dim_indices: np.ndarray,
                            third_dim_indices: np.ndarray,):
        subset = arr[first_dim_indices][:, second_dim_indices][:, :, third_dim_indices]

        return subset

    @staticmethod
    def initialize_empty_kernel_matrix(X: np.ndarray):
        N = X.shape[0]
        kernel_matrix_shape = (N, N)

        return np.zeros(kernel_matrix_shape)


class TCKUtils:
    lock = threading.Lock()

    @staticmethod
    def single_fit(q, gmm_model, X, R, log_msg):
        logger = TCKUtils.set_logger("TCK", logging.INFO)

        with TCKUtils.lock:
            logger.info(log_msg)

        gmm_model.fit(X, R)

        with TCKUtils.lock:
            logger.info(f"Finnished q={q+1} ")

        return q, gmm_model

    @staticmethod
    def get_random_gmm_hyperparameters():
        # The parameters are initialized according to section 4.2 in the article
        a0 = np.random.uniform(0.001, 1)
        b0 = np.random.uniform(0.005, 0.2)
        N0 = np.random.uniform(0.001, 0.2)

        return {'a0': a0, 'b0': b0, 'N0': N0}

    @staticmethod
    def set_logger(logger_name: str, log_level: int) -> logging.Logger:
        logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(funcName)s :: %(lineno)d '
                                   ':: %(message)s', level=log_level)
        logger = logging.getLogger(logger_name)

        return logger

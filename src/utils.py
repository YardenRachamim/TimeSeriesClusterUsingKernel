import logging

from scipy import interpolate
import threading
import joblib
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import io
from scipy.spatial.distance import jensenshannon
from itertools import product


class DataUtils:
    @staticmethod
    def get_article_data_set():
        # This is the example run of the MATLAB code
        X = io \
            .loadmat(r"C:\Users\Yarden\Computer Science\Masters\1\Advance Machine Learning\final "
                     r"project\Time-series-cluster-kernel-TCK--master\x_VAR.mat") \
            ['x'].reshape((200, 50, 2))
        Z = np.ones_like(X)
        R = np.random.random(X.shape)
        Z[R > 0.5] = 0
        X[Z == 0] = np.nan

        X_test = io \
            .loadmat(r"C:\Users\Yarden\Computer Science\Masters\1\Advance Machine Learning\final "
                     r"project\Time-series-cluster-kernel-TCK--master\xte_VAR.mat") \
            ['xte'].reshape((200, 50, 2))
        Z_test = np.ones_like(X_test)
        R_Test = np.random.random(X_test.shape)
        Z_test[R_Test > 0.5] = 0
        X_test[Z_test == 0] = np.nan

        y_test = np.zeros(X_test.shape[0], dtype=int)
        y_test[0:y_test.shape[0] // 2] = 1
        y_test[y_test.shape[0] // 2:] = 2

        return X, X_test, y_test, y_test

    @staticmethod
    def get_3d_array_subset(arr: np.ndarray,
                            first_dim_indices: np.ndarray,
                            second_dim_indices: np.ndarray,
                            third_dim_indices: np.ndarray, ):
        subset = arr[first_dim_indices][:, second_dim_indices][:, :, third_dim_indices]

        return subset

    @staticmethod
    def get_train_test_indices(data_array, labels_array):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25)
        return next(sss.split(data_array, labels_array))

    @staticmethod
    def get_wisdom_data():
        data_array = joblib.load(
            r"C:\Users\Yarden\Computer Science\Masters\Research\WISDOM-20210725T165727Z-001\WISDOM\raw"
            r"\data_10sec_20Hz_compress_3.gz")
        labels_array = joblib.load(
            r"C:\Users\Yarden\Computer Science\Masters\Research\WISDOM-20210725T165727Z-001\WISDOM\raw\label_10sec_20Hz")
        scaler = MinMaxScaler()
        train_indices, test_indices = DataUtils.get_train_test_indices(data_array, labels_array)
        train_x, test_x = data_array[train_indices], data_array[test_indices]
        train_y, test_y = labels_array[train_indices].reshape(-1), labels_array[test_indices].reshape(-1)

        temp_train_x = scaler.fit_transform(train_x.reshape(train_x.shape[1], -1).T)
        train_x = temp_train_x.T.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2])
        temp_test_x = scaler.transform(test_x.reshape(test_x.shape[1], -1).T)
        test_x = temp_test_x.T.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2])

        return np.transpose(train_x, (0, 2, 1)), \
               np.transpose(test_x, (0, 2, 1)), \
               train_y, \
               test_y


class TCKUtils:
    lock = threading.Lock()

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

    @staticmethod
    def interp_data(X: np.ndarray,
                    X_len: list = None,
                    restore: bool = False,
                    interp_kind: str = 'linear'):
        """
        Interpolate data to match the same maximum length in X_len
        If restore is True, data are interpolated back to their original length
        data are assumed to be time-major
        interp_kind: can be 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        """
        if X_len is None:
            X_len = [X.shape[1] for _ in range(X.shape[0])]
        [N, T, V] = X.shape
        X_new = np.zeros((X.shape[0], X_len[0], X.shape[2]))

        # restore original lengths
        if not restore:
            for n in range(N):
                t = np.linspace(start=0, stop=X_len[n], num=T)
                t_new = np.linspace(start=0, stop=X_len[n], num=X_len[n])
                for v in range(V):
                    x_n_v = X[n, :, v]
                    f = interpolate.interp1d(t, x_n_v, kind=interp_kind)
                    X_new[n, :X_len[n], v] = f(t_new)

        # interpolate all data to length T
        else:
            for n in range(N):
                t = np.linspace(start=0, stop=X_len[n], num=X_len[n])
                t_new = np.linspace(start=0, stop=X_len[n], num=T)
                for v in range(V):
                    x_n_v = X[n, :X_len[n], v]
                    f = interpolate.interp1d(t, x_n_v, kind=interp_kind)
                    X_new[n, :, v] = f(t_new)

        return X_new


class LinearAlgebraUtils:
    @staticmethod
    def is_invertible(a: np.ndarray) -> bool:
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

    @staticmethod
    def check_symmetric(a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @staticmethod
    def is_pd(K):
        try:
            np.linalg.cholesky(K)
            return True
        except np.linalg.linalg.LinAlgError as err:
            if 'Matrix is not positive definite' in str(err):
                return False
            else:
                raise

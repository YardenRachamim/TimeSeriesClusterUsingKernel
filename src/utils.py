import numpy as np


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

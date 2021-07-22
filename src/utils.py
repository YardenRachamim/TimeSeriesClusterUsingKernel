import numpy as np


class DataUtils:
    @staticmethod
    def get_3d_array_subset(arr: np.ndarray,
                            first_dim_indices: np.ndarray,
                            second_dim_indices: np.ndarray,
                            third_dim_indices: np.ndarray,):
        subset = arr[first_dim_indices][:, second_dim_indices][:, :, third_dim_indices]

        return subset

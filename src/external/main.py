import joblib
import numpy as np
import scipy.io
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from GMM_MAP_EM import GMM_MAP_EM
from utils import TCKUtils
from TCK import TCK
from scipy import io
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_train_test_indices(data_array, labels_array):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25)
    return next(sss.split(data_array, labels_array))


def get_wisdom_data():
    data_array = joblib.load(
        r"C:\Users\Yarden\Computer Science\Masters\Research\WISDOM-20210725T165727Z-001\WISDOM\raw"
        r"\data_10sec_20Hz_compress_3.gz")
    labels_array = joblib.load(
        r"C:\Users\Yarden\Computer Science\Masters\Research\WISDOM-20210725T165727Z-001\WISDOM\raw\label_10sec_20Hz")
    scaler = MinMaxScaler()
    train_indices, test_indices = get_train_test_indices(data_array, labels_array)
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

    return X, X_test, None, y_test, Z, Z_test


if __name__ == '__main__':
    np.random.seed(0)  # only once for TCK life to ensure that randoms or permutations do not repeat.

    art_X_train, art_X_test, art_y_train, art_y_test, art_R_teain, art_R_test = get_article_data_set()
    wis_X_train, wis_X_test, wis_y_train, wis_y_test = get_wisdom_data()

    tck_model = TCK(Q=30, C=40, n_jobs=-1)

    tck_model.fit(wis_X_train)

    Kte, _ = tck_model.transform(wis_X_test)
    y_pred = wis_y_train[Kte.argmax(axis=0)].astype(int)
    accuracy = accuracy_score(wis_y_test, y_pred)
    print(accuracy)
    X_tsne = TSNE(n_components=2,
                  n_jobs=-1,
                  verbose=1,
                  init="pca").fit_transform(Kte)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=wis_y_test)
    plt.show()

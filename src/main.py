import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import io

from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import KernelPCA

from GMM_MAP_EM import GMM_MAP_EM
from utils import TCKUtils
from TCK import TCK

from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import cdist_ctw
from tslearn.utils import to_time_series_dataset
from sklearn.neighbors import KNeighborsClassifier

from pathlib import Path


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

    return X, X_test, y_test, y_test, Z, Z_test


def pickle_tck_model(tck_model: TCK,
                     X_train: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray,
                     R_train: np.ndarray = None, R_test: np.ndarray = None
                     ):
    with open(r"/models/tck_model2", 'wb') as fos:
        pickle.dump(
            (tck_model, X_train, X_test, y_train, y_test, R_train, R_test), fos
        )


def load_tck_model():
    with open(r"/models/tck_model2", 'rb') as fis:
        model_params = pickle.load(fis)

    return model_params


def load_ucr_dataset(dataset_name: str):
    uea_ucr = UCR_UEA_datasets()

    dataset = uea_ucr.load_dataset(dataset_name)

    return dataset[0], dataset[2], dataset[1], dataset[3]


def get_blood_test_data(is_data_norm: bool = True):
    x_tr = io.loadmat(r"C:\Users\Yarden\Desktop\Temp\TCK_AE-master\TCK_AE-master\Data\x.mat")
    y_tr = io.loadmat(r"C:\Users\Yarden\Desktop\Temp\TCK_AE-master\TCK_AE-master\Data\Y.mat")
    x_te = io.loadmat(r"C:\Users\Yarden\Desktop\Temp\TCK_AE-master\TCK_AE-master\Data\xte.mat")
    y_te = io.loadmat(r"C:\Users\Yarden\Desktop\Temp\TCK_AE-master\TCK_AE-master\Data\Yte.mat")

    Ntr = x_tr['x'].shape[0]
    Nts = x_te['xte'].shape[0]
    T = 20
    V = 10

    X = np.reshape(x_tr['x'], (Ntr, T, V))
    Xte = np.reshape(x_te['xte'], (Nts, T, V))
    Y = y_tr['Y']
    Yte = y_te['Yte']

    if is_data_norm:
        for v in range(1, V):
            X_v = X[:, :, v]
            Xte_v = Xte[:, :, v]
            Xv_m = np.nanmean(X_v[:])
            Xv_s = np.nanstd(X_v[:])

            X_v = (X_v - Xv_m) / Xv_s
            X[:, :, v] = X_v
            Xte_v = (Xte_v - Xv_m) / Xv_s
            Xte[:, :, v] = Xte_v

    return X, Xte, Y, Yte


if __name__ == '__main__':
    np.random.seed(0)  # only once for TCK life to ensure that randoms or permutations do not repeat.

    # Single GMM test
    test = io.loadmat(r"C:\Users\Yarden\Computer Science\Masters\1\Advance Machine Learning\final project\src\test\test.mat")
    X = test['sX']
    R = test['R']
    a0 = test['a0'][0][0]
    b0 = test['b0'][0][0]
    N0 = test['n0'][0][0]
    C = test['C'][0][0]
    gmm_model = GMM_MAP_EM(a0, b0, N0, C)
    gmm_model.fit(X, R)

    # # Syntetic data
    # X_train, X_test, y_train, y_test, _, _ = get_article_data_set()
    # X_train = to_time_series_dataset(X_train)

    # # Blood test
    # X_train, X_test, y_train, y_test = get_blood_test_data()

    # # Wisdom data set
    # X_train, X_test, y_train, y_test = get_wisdom_data()

    #  # Pen digits
    # X_train, X_test, y_train, y_test = load_ucr_dataset('PenDigits')
    # train_instance_num = 300
    # X_test = np.array(list(X_train[train_instance_num:]) + list(X_test))
    # y_test = np.array(list(y_train[train_instance_num:]) + list(y_test)).astype(int)
    # X_train = X_train[: train_instance_num]
    # y_train = y_train[: train_instance_num].astype(int)

    # # Arabic digits
    # X_train, X_test, y_train, y_test = load_ucr_dataset('SpokenArabicDigits')
    #
    # # Preprocess
    # scaler = TimeSeriesScalerMeanVariance()
    # X_train = scaler.fit_transform(X_train)
    # # X_test = scaler.transform(X_test)
    # X_train_len = [24 for _ in range(X_train.shape[0])]
    # X_train = TCKUtils.interp_data(X_train, X_train_len)

    # # Training the model
    # R_train = (~(np.isnan(X_train))).astype(int)
    # R_test = (~(np.isnan(X_test))).astype(int)
    # tck_model = TCK(Q=30, C=40, n_jobs=6, max_features='all')
    # tck_model.fit(X_train, R_train)

    # # Saving the model
    # pickle_tck_model(tck_model, X_train, X_test, y_train, y_test)

    # # Reading saved model
    # tck_model, X_train, X_test, y_train, y_test, R_train, R_test = load_tck_model()

    # dtw_similarity = cdist_ctw(X_train,
    #           n_jobs=-1,
    #           verbose=1)
    # tck_similarity = tck_model.K
    # from sklearn import svm

    # clf = svm.SVC(kernel='precomputed')
    # neigh = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
    # neigh.fit(tck_model.K, y_train)

    # Predict
    K_star, K_test = tck_model.transform(X_test, R_test)

    # y_pred = neigh.predict(K_star.T)
    tck_y_pred = y_train[K_star.T.argmax(axis=1)].astype(int)
    accuracy = accuracy_score(y_test, tck_y_pred)
    print(accuracy)

    # Visualization
    X_pca = KernelPCA(n_components=2, kernel='precomputed').fit_transform(K_star)
    # tck_X_tsne = TSNE(n_components=2,
    #               n_jobs=-1,
    #               verbose=1,
    #               init="pca").fit_transform(tck_model.K)
    #
    # # dtw_X_tsne = TSNE(n_components=2,
    # #               n_jobs=-1,
    # #               verbose=1,
    # #               init="pca").fit_transform(dtw_similarity)
    #
    # # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # #
    # # axes[0].scatter(tck_X_tsne[:, 0], tck_X_tsne[:, 1], c=y_train)
    # # axes[0].set_title("TCK")
    # #
    # # axes[0].scatter(dtw_X_tsne[:, 0], dtw_X_tsne[:, 1], c=y_train)
    # # axes[0].set_title("DTW")
    # # plt.show()
    #
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test)
    plt.show()

import numpy as np
from sklearn.base import TransformerMixin
from scipy.stats import norm
from scipy.stats import multivariate_normal
from typing import Tuple


class GMM_MAP_EM(TransformerMixin):
    """
    :param a0: Hyperparameter for the kernel based mean prior
    :param b0 : Hyperparameter for the kernel based mean prior
    :param N0: Hyperparameter for the inverse gamma distribution on the std prior
    :param C: Number of gaussians
    :param num_iter: number of maximum iteration (default: 20)
    """

    def __init__(self, a0: float,
                 b0: float,
                 N0: int,
                 C: int,
                 num_iter: int = 20):
        self.a0 = a0
        self.b0 = b0
        self.N0 = N0
        self.C = C
        self.num_iter = num_iter
        self.N = None  # Number of MTS instances
        self.V = None  # Number of attributes
        self.T = None  # Number of time segments
        self.posteriors = None  # shape CxN
        self.theta = None  # Shape Cx1
        self.mu = None  # Shape CxTxV
        self.s2 = None  # Shape CxV

        self.empirical_mean = None  # Shape NxT
        self.empirical_variance = None  # Shape Vx1
        self.S_0 = None  # Shape TxTxV
        self.invS_0 = None

        # # TODO: check if its ok to add this. for smothness
        # self.EPSILON = 1e-6
        self.EPSILON = 0
        self.v_multivariate_normal_pdf = np.vectorize(multivariate_normal.pdf)

    """
        Algorithm 1 from the article
        :param X: a 3d matrix represent MTS (NxTxV)
        :param R: a 3d matrix represent the missing values of the MTS
    """
    def fit(self, X: np.ndarray,
            R: np.ndarray = None):
        # TODO: change to more indicative names, for now follow the matlab code notations
        self.N = X.shape[0]
        self.T = X.shape[1]
        self.V = X.shape[2]
        # TODO: initialize correctly
        self.posteriors = self.init_cluster_posteriors()  # Big pi in the algorithm
        self.theta = self.init_cluster_theta()  # Small theta
        self.mu = self.init_cluster_means()  # mu of big theta
        self.s2 = self.init_cluster_variance()  # s2 of big theta
        self.empirical_mean = np.nanmean(X, axis=0)  # m_v
        self.empirical_variance = np.nanstd(X.reshape(self.N*self.T, self.V), axis=0) ** 2
        self.S_0, self.invS_0 = self.init_s0()

        if R is None:
            R = np.ones_like(X)

        for i in range(self.num_iter):
            # Here we assumed we have a random posterior initialization,
            # hence we will start with the maximization step
            is_first_iter = i == 0
            if not is_first_iter:
                self.expectation_step(X, R)
            self.maximization_step(X, R)

        return self

    def init_s0(self) -> Tuple[np.ndarray, np.ndarray]:
        S_0 = np.zeros((self.T, self.T, self.V))
        invS_0 = np.zeros_like(S_0)
        T_1 = np.arange(self.T)
        T_2 = np.arange(self.T).reshape((self.T, 1))

        for v in range(self.V):
            S_0[:, :, v] = self.empirical_variance[v] * self.b0 * np.exp((-self.a0 * (T_1 - T_2)**2))
            invS_0[:, :, v] = np.linalg.inv(S_0[:, :, v])
        # TODO: add a condition check og invertable

        return S_0, invS_0

    def init_cluster_posteriors(self) -> np.ndarray:
        raw = np.random.rand(self.C, self.N)
        sums = raw.sum(axis=0)   # sum for each MTS
        posteriors = raw / sums[np.newaxis, :]  # each MTS total probability is 1
        
        return posteriors

    def init_cluster_theta(self) -> np.ndarray:
        return np.ones(self.C) / self.C

    def init_cluster_means(self) -> np.ndarray:
        return np.random.normal(loc=0, scale=1, size=(self.C, self.T, self.V))

    def init_cluster_variance(self) -> np.ndarray:
        return np.ones((self.C, self.V))  # cluster variances

    def expectation_step(self, X: np.ndarray, R: np.ndarray):
        self.posteriors = self.evaluate_posterior(X, R)

    def evaluate_posterior(self, X: np.ndarray, R: np.ndarray) -> np.ndarray:
        N, T, V = X.shape[0], X.shape[1], X.shape[2]
        posterior = np.zeros((self.C, N))

        for c in range(self.C):
            mean = np.tile(self.mu[c], (N, 1, 1))
            cov = np.tile(np.sqrt(self.s2[c]), (N, T, 1))
            prob = self.v_multivariate_normal_pdf(X, mean=mean, cov=cov) ** R

            posterior[c] = prob.prod(axis=1).prod(axis=1)

        return posterior / posterior.sum(axis=0)

    def maximization_step(self, X: np.ndarray, R: np.ndarray):
        # Update theta
        self.theta = self.posteriors.sum(axis=1) / self.N

        # Update sigma
        for c in range(self.C):
            for v in range(self.V):
                var2 = np.matmul(R[:, :, v].sum(axis=1).T, self.posteriors[c])
                temp = (X[:, :, v] - np.tile(self.mu[c, :, v].T, (self.N, 1))) ** 2
                var1 = np.matmul(self.posteriors[c].T, (R[:, :, v] * temp).sum(axis=1))
                self.s2[c, v] = (self.N0 * self.empirical_variance[v] + var1) / (self.N0 + var2)

                A = self.invS_0[:, :, v] + np.diag(np.matmul(R[:, :, v].T, self.posteriors[c]) / self.s2[c, v])
                b = np.matmul(self.invS_0[:, :, v], self.empirical_mean[:, v]) + \
                    np.matmul((R[:, :, v] * X[:, :, v]).T, self.posteriors[c]) / self.s2[c, v]
                self.mu[c, :, v] = np.linalg.lstsq(A, b)[0]

    def transform(self, X: np.ndarray, R: np.ndarray = None) -> np.ndarray:
        is_dim_added = False
        if X.ndim < 3:
            X = X[None, :, :]
        if R is None:
            R = np.ones_like(X)
        if R.ndim < 3:
            R = R[None, :, :]

        return self.evaluate_posterior(X, R)
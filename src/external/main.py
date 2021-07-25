from GMM_MAP_EM import GMM_MAP_EM
from TCK import TCK
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

model = TCK(Q=2, C=3)
X1 = np.random.normal(loc=0, scale=1, size=(5, 100, 3))
X2 = np.random.normal(loc=1, scale=1, size=(5, 100, 3))
X3 = np.random.normal(loc=2, scale=1, size=(5, 100, 3))

X = np.array(list(X1) + list(X2) + list(X3))

model.fit(X)
Y1 = np.random.normal(loc=0, scale=1, size=(5, 100, 3))
Y2 = np.random.normal(loc=1, scale=1, size=(5, 100, 3))
Y3 = np.random.normal(loc=2, scale=1, size=(5, 100, 3))

K = model.transform(Y1)
model.get_pairwise_dist()
model.get_distance_matrix_from_kernel_matrix(K)
print(model.transform(Y1))

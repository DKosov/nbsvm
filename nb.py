import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse


class NBFeaturer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha):
        self.alpha = alpha

    def transform(self, x):
        return x.multiply(self._r)

    def fit(self, x, y):
        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) / self.pr(x, 0, y)))
        return self

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(axis=0)
        return (p + self.alpha) / (p.sum() + self.alpha)

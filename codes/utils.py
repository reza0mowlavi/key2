from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RBF(BaseEstimator, TransformerMixin):
    def __init__(self, gamma, dtype="float64", transform_on_distance=None):
        super().__init__()
        self.gamma = gamma
        self.dtype = dtype
        self.transform_on_distance = transform_on_distance

    def fit(self, X, y=None):
        if self.gamma == "scale":
            self._gamma = 1 / (X.shape[-1] * X.var())
        elif self.gamma == "auto":
            self._gamma = 1 / (X.shape[-1])
        else:
            self._gamma = self.gamma

        return self

    def transform(self, X):
        og_X = X
        X = X if self.transform_on_distance is None else self.transform_on_distance(X)
        X = X.astype(self.dtype)
        if X is og_X:
            X = np.copy(X)
        np.multiply(X, -self._gamma, X)
        np.exp(X, X)
        return X

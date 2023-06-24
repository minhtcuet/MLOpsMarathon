import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from optbinning import BinningProcess, OptimalBinning


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, _feature_names):
        self._feature_names = _feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._feature_names]


class WOE(BaseEstimator, TransformerMixin):
    def __init__(self, cats, nums):
        self.cats = cats
        self.nums = nums
        self.res = {}

    def fit(self, X, y):
        for col in self.cats:
            optb = OptimalBinning(name=col, dtype="categorical", solver="cp")
            optb.fit(X[col].values, y.values)
            self.res[col] = optb

        for col in self.nums:
            optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
            optb.fit(X[col].values, y.values)
            self.res[col] = optb
        return self

    def _round(self, number):
        return int(number * 10 ** 8) / 10 ** 8

    def transform(self, X, y=None):
        for col in X.columns:
            if col in self.nums + self.cats:
                X[col] = self.res[col].transform(X[col], metric='woe')
                X[col] = X[col].apply(self._round)
        return X
"""Importable, serializable train-only preprocessing for Adult."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']


class CleanFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, derived=False):
        self.derived = derived

    def _missing(self, X):
        return X.astype(object).replace('?', np.nan)

    def fit(self, X, y=None):
        X = self._missing(X)
        self.fill_ = {c: float(pd.to_numeric(X[c]).median()) for c in NUMERIC}
        self.fill_.update({c: X[c].mode().iloc[0] for c in CATEGORICAL})
        numeric = X[NUMERIC].astype(float).fillna(self.fill_)
        self.lower_ = numeric.quantile(.01)
        self.upper_ = numeric.quantile(.99)
        return self

    def transform(self, X):
        X = self._missing(X).fillna(self.fill_)
        X[NUMERIC] = X[NUMERIC].astype(float).clip(self.lower_, self.upper_, axis=1)
        if self.derived:
            X['capital_net'] = X['capital-gain'] - X['capital-loss']
            X['hours_per_age'] = X['hours-per-week'] / (X['age'] - 16)
        return X

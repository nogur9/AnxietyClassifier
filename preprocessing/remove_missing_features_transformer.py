from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class RemoveMissingFeaturesTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        copy_x = X.dropna(axis=1, inplace=False)
        return copy_x

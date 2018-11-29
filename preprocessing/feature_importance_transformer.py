from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


class FeatureImportanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_of_feature=15):
        self.num_of_feature = num_of_feature

    def fit(self, X, Y=None):
        model = RandomForestClassifier()
        model.fit(X, Y)
        self.feature_importance = model.feature_importances_
        return self

    def transform(self, X, Y=None):
        df = pd.DataFrame(X)
        importance_zip = sorted(zip(self.feature_importance, df.T), key=lambda x: x[0], reverse=True)
        columns = [importance_zip[i][1] for i in range(self.num_of_feature)]
        highest_features = df.iloc[:, columns]
        return highest_features.values


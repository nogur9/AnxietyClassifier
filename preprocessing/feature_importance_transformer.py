from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import datetime


def get_feature_importance(X, Y):
    model = RandomForestClassifier()
    model.fit(X, Y)
    return model.feature_importances_

class FeatureImportanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_of_feature=15, feature_importance_file_name="feature_importance_{}.xlsx".format(
        datetime.datetime.now().strftime('%Y-%m-%d')), full_feature_importance_analysis=True):

        self.num_of_feature = num_of_feature
        self.feature_importance_file_name = feature_importance_file_name
        self.full_feature_importance_analysis = full_feature_importance_analysis


    def fit(self, X, Y=None):
        if self.full_feature_importance_analysis:
            df = pd.DataFrame(X)
            columns_dict = {column: [] for column in df.columns}
            iterations = 10000
            for k in range(iterations):
                feature_importance = get_feature_importance(df, Y)
                importance_zip = sorted(zip(feature_importance, list(df)), key=lambda x: x[0], reverse=True)

                for i, item in enumerate(importance_zip):
                    columns_dict[item[1]].append(i)
            pd.DataFrame(columns_dict).mean().sort_values(ascending=True).to_excel(self.feature_importance_file_name)

        self.feature_importance = pd.read_excel(self.feature_importance_file_name)[:self.num_of_feature]
        return self

    def transform(self, X, Y=None):
        df = pd.DataFrame(X)
        columns = self.feature_importance.index
        highest_features = df.loc[:, columns.tolist()]
        return highest_features

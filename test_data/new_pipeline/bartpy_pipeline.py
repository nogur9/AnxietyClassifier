from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pandas as pd
import os
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from imblearn.over_sampling import SMOTE, ADASYN
#from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import random
from bartpy.sklearnmodel import SklearnModel



class RemoveCorrelationTransformer2(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold=0.7):
        self.correlation_threshold = correlation_threshold


    def fit(self, X, Y=None):
        df = pd.DataFrame(X)
        df_corr = df.corr(method='pearson', min_periods=1)
        df_not_correlated = ~(df_corr.mask(
            np.tril(np.ones([len(df_corr)] * 2, dtype=bool))).abs() > self.correlation_threshold).any()
        self.un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
        return self

    def transform(self, X, Y=None):
        df = pd.DataFrame(X)
        df = df[self.un_corr_idx]
        return df.values




class RemoveCorrelationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold=0.7, pca_components_ratio=3):
        self.correlation_threshold = correlation_threshold
        self.pca_components_ratio = pca_components_ratio


    def fit(self, X, Y=None):
        df = pd.DataFrame(X)
        df_corr = df.corr(method='pearson')
        df_corr = df_corr - np.eye(df.shape[1])
        outliares_corr = df_corr[np.abs(df_corr) > self.correlation_threshold]
        self.outliares_corr = outliares_corr.dropna(axis=1, how='all')

        correlated_df = df[self.outliares_corr.columns]

        n_components = len(self.outliares_corr.columns) // self.pca_components_ratio
        pca = PCA(n_components=n_components)

        correlated_df = pca.fit_transform(correlated_df)
        self.correlated_df = pd.DataFrame(correlated_df, columns=["pca_{}".format(i) for i in range(n_components)])

        return self

    def transform(self, X, Y=None):
        df = pd.DataFrame(X)
        df = df.drop((self.outliares_corr.columns), axis=1)
        df = df.join(self.correlated_df)
        return df



class RemoveMissingFeaturesTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, Y=None):
        self.is_missing = X.isnull().values.any(axis=0)
        return self

    def transform(self, X, Y=None):
        copy_x = pd.DataFrame(X)
        self.is_missing += copy_x.isnull().values.any(axis=0)

        copy_x = copy_x.iloc[:, ~self.is_missing]

        return copy_x.values




def refactor_labels(df):
    return df.replace({'low': 0 ,'high': 1, 'clinical': 1 })


def get_data(file_name, LSAS_threshold=None):
    group_column = 'group'
    sub_num_col = 'Subject_Number'
    lsas_col = 'LSAS'
    df = pd.read_excel(file_name, sheet_name='Sheet1')
    if LSAS_threshold is None:
        X = df.drop([group_column, sub_num_col, lsas_col], 1)
        Y = refactor_labels(df[group_column])
        return X, Y
    else:
        X = df.drop([group_column], 1)
        Y = pd.Series(np.where(X[lsas_col] > LSAS_threshold, 1, 0))
        X = X.drop([sub_num_col, lsas_col], 1)
        return X, Y



file_name = "training_set_100.xlsx"
X_train, y_train = get_data(file_name, LSAS_threshold = 50)
random.seed(217828)
columns_shuffled = list(X_train.columns)
random.shuffle(columns_shuffled)
X_train = X_train[columns_shuffled]
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_train, y_train, test_size = 0.1, stratify=y_train)
#
# pipe =  Pipeline([
#     ("rnf", RemoveMissingFeaturesTransformer()),
#     ('correlation_threshold', RemoveCorrelationTransformer2()),
#     ('rfc', RFE(RandomForestClassifier(n_estimators = 100))),
#     ('classifier', SklearnModel())])
#
# params_grid = [
#     {
#         'correlation_threshold__correlation_threshold' : [0.8, 0.7, 0.9],
#         'rfc__n_features_to_select': [8,6,10,14]
#     }]


model = SklearnModel(sigma_a=0.1, sigma_b=0.1, n_samples =200)
model.fit(X_train_2, y_train_2)
y_pred = model.predict(X_test_2) > 0.5
print(accuracy_score(y_test_2, y_pred))

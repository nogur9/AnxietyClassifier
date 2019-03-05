from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from preprocessing.lsas_cutoff import LSASCutoff
from preprocessing.remove_high_correlation import RemoveCorrelationTransformer
from preprocessing.remove_invalid_columns_transformer import RemoveInvalidColumns
from preprocessing.remove_missing_features_transformer import RemoveMissingFeaturesTransformer
from sklearn.neighbors import KNeighborsClassifier
from preprocessing.feature_importance_transformer import FeatureImportanceTransformer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from preprocessing.Corr import RemoveCorrelationTransformer2
from sklearn.feature_selection import RFE

def build_full_pipeline():
    rfc = RandomForestClassifier()

    prepro =Pipeline([
        ('missing_values', RemoveMissingFeaturesTransformer()),
        ('scaling', MinMaxScaler()),
        ('variance_threshold', VarianceThreshold()),
        ('correlation_threshold', RemoveCorrelationTransformer2()),
        ('rfc', RFE(RandomForestClassifier(n_estimators = 100))),
    ])

    clf = Pipeline([
        ('classifier', GradientBoostingClassifier(learning_rate= 0.05))
         ])
    return prepro, clf

def get_full_params_grid():
    prepro_params = [

        #{'variance_threshold__threshold': [0]},

        #{'correlation_threshold__pca_components_ratio': [3]},
        {'rfc__n_features_to_select': [11]},

        #{'rcf__num_of_iterations': [1000, 100, 10000]},
        #{'pca__n_components': [7]}
    ]

    params_grid = [
        {
 'classifier__max_depth': [6],

 'classifier__n_estimators': [400]}]

    return prepro_params, params_grid




def build_pipeline():
    rfc = RandomForestClassifier()

    pipeline = Pipeline([
        ('classifier', rfc)
         ])
    return pipeline

def get_params_grid():
    params_grid = [
        {'classifier': [GradientBoostingClassifier()],
                          "classifier__learning_rate": [0.01, 0.2],
                          "classifier__min_samples_split": np.linspace(0.1, 0.5, 2),
                          "classifier__min_samples_leaf": np.linspace(0.1, 0.5, 2),
                          "classifier__subsample": [0.5, 0.75]},
        {'classifier': [RandomForestClassifier()],
                          'classifier__n_estimators': [200, 500],
                          'classifier__max_features': ['log2'],
                          'classifier__max_depth': [4, 10],
                          }]

    return params_grid
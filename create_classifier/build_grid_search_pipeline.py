from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

def build_full_pipeline():
    rfc = RandomForestClassifier()

    prepro =Pipeline([
        ('missing_values', RemoveMissingFeaturesTransformer()),
        ('scaling', MinMaxScaler()),
        ('variance_threshold', VarianceThreshold()),
        ('correlation_threshold', RemoveCorrelationTransformer2()),
        ('rfc', FeatureImportanceTransformer()),
        ('pca', PCA())
    ])

    clf = Pipeline([
        ('classifier', rfc)
         ])
    return prepro, clf

def get_full_params_grid():
    prepro_params = [

        {'variance_threshold__threshold': [0]},
        {'correlation_threshold__correlation_threshold': [0.9]},
        #{'correlation_threshold__pca_components_ratio': [3]},
        {'rfc__threshold': [15]},
        #{'rcf__num_of_iterations': [1000, 100, 10000]},
        {'pca__n_components': [7, 6, 5]}
    ]

    params_grid = [

         {'classifier': [SVC()],
          'classifier__C':[1, 3, 5, 0.75, 0.5],
          'classifier__kernel':['sigmoid', 'rbf'],
          'classifier__gamma': [1, 3, 5, 0.75, 0.5]
          }]

        #]
         # {'classifier': [GradientBoostingClassifier()],
          #                  "classifier__learning_rate": [0.2, 0.4],
           #                 "classifier__n_estimators":[100, 200],
            #                "classifier__min_samples_split": [0.5],
             #               "classifier__min_samples_leaf": [0.1],
              #              "classifier__subsample": [0.5],
               #             "classifier__max_depth":[5]
 #           },
  #        {'classifier': [RandomForestClassifier()],
   #                          'classifier__n_estimators': [200, 100],
    #                         'classifier__max_features': ['log2', None],
     #                        'classifier__max_depth': [4, 7],
      #                       'classifier__min_samples_split':[3]
       #                      }]

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
                          'classifier__criterion': ['gini', 'entropy']
                          }]

    return params_grid
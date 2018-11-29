import matplotlib.pyplot as plt
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from DataImporting import ImportData

# feature selection
from sklearn.feature_selection import RFECV

# model selection
from sklearn import model_selection

# models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.preprocessing import MinMaxScaler

#defining enviroment variables
from dim_reduction.feature_selection_pipeline import interactive_pipeline

SEED = 7
SCORING = 'accuracy'
#SCORING = 'f1'
PATH =r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\data_features_10_3.csv"
SHEET_NAME = "Sheet1"

group_column = 'group'
subject_number_column = 'Subject_Number'
subject_number_present = 0

def get_data(dataset):
    if subject_number_present:
        X_train = dataset.drop([group_column,subject_number_column], 1)
    else:
        X_train = dataset.drop([group_column], 1)
    Y_train = dataset[group_column]
    return X_train,Y_train


def grid_search(pipe, search_space, X, Y):
    clf = GridSearchCV(pipe, search_space, scoring=SCORING)
    best_model = clf.fit(X, Y)
    #print(clf.best_score_,"\nclf.best_score_")
    print("best classifier", best_model.best_estimator_)
    cv_results = clf.best_score_
    #cv_results = model_selection.cross_val_score(best_model, X, Y, cv=model_selection.LeaveOneOut(), scoring=SCORING)
    print("best score", cv_results,"\n")
    print("best_params_", best_model.best_params_)
    print("grid_scores_", best_model.grid_scores_)
    return best_model

def get_pipeline():
    # Create a pipeline
    pipe = Pipeline([('classifier', RandomForestClassifier())])
    # Create space of candidate learning algorithms and their hyperparameters
    search_space = [{'classifier': [GradientBoostingClassifier()],
                     "classifier__learning_rate": [0.01, 0.1, 0.2],
                     "classifier__min_samples_split": np.linspace(0.1, 0.5, 5),
                     "classifier__min_samples_leaf": np.linspace(0.1, 0.5, 5),
                     "classifier__subsample": [0.5, 0.75, 1.0],
                     },
                    # {'classifier': [SVC(kernel='linear')],
                    #  'classifier__C': [0.001, 1, 10],
                    # {'classifier': [LogisticRegression()],
                    #  'classifier__C': [10 ** -i for i in [-5, -3, 0, 3, 5]]}
                    # {'classifier': [KNeighborsClassifier()],
                    #  'classifier__metric': ['euclidean', 'manhattan'],
                    #  'classifier__n_neighbors': [5, 10]
                    #  },
                    {'classifier': [RandomForestClassifier()],
                     'classifier__n_estimators': [200, 500, 1000],
                     'classifier__max_features': ['auto', 'log2'],
                     'classifier__max_depth': [4, 6, 8, 10],
                     'classifier__criterion': ['gini', 'entropy']
                     }
                    ]
    return pipe, search_space


def get_RFE_pipeline(X_train, Y_train):
    lda = SVC(kernel='linear')
    rfecv = RFECV(estimator=lda, step=1, scoring=SCORING)
    rfecv.fit_transform(X_train, Y_train)
    cv_results = model_selection.cross_val_score(rfecv, X_train, Y_train, scoring=SCORING)
    print(cv_results)
    return rfecv



def runner(path,sheet_name):
    for pca_n in [3,5,8,9]:
        for rf_k in [3,6,9,12]:
            print("pca n", pca_n+2, "RF k",pca_n+rf_k+2)
            X_train, Y_train = interactive_pipeline(pca_n+2,pca_n+rf_k+2)
            pipe, search_space = get_pipeline()
            grid_search(pipe, search_space, X_train, Y_train)
            print("naive classifier score {}".format(Y_train[Y_train == 0].count()/ Y_train[Y_train == 1].count()))






runner(PATH,SHEET_NAME)
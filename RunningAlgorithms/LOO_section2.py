import xlsxwriter
from sklearn.model_selection import LeaveOneOut
from CalculatingFeaturesFromExcel.PCA import PCA_transforme
# Libraries
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from DataImporting import DataFromExcel

# Missing values
from DataImporting.Data_Imputation import imputing_median
from DataImporting.Data_Imputation import imputing_avarage
from DataImporting.Data_Imputation import imputing_knn
from DataImporting.Data_Imputation import imputing_most_frequent

# feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFECV

# model selection
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

# models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

#defining enviroment variables
SEED = 7
SCORING = 'accuracy'
COMB = 105
PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\Alls_data_NO.xlsx"
#PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\Alls_data.xlsx"
SHEET_NAME = "Sheet1"

def get_data (dateset):
    X_train = PCA_transforme(dateset,10)
    Y_train = dateset["group"]
    return X_train,Y_train

def get_RFE_models():
    '''

    :return: List of classification models that can be evaluated by RFE.
    '''

    models = []
    models.append(('LogisticRegression', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('RandomForest', RandomForestClassifier()))
    models.append(('AdaBoost', AdaBoostClassifier()))
    models.append(('GradientBoosting', GradientBoostingClassifier()))
    models.append(('SVM_linear', SVC(kernel='linear')))
    return models


def get_best_result_RFE(results_list):
    '''
    :param results_list: list of results given from RFE CV functions.

    Prints the classifier with the best accuracy,
    creates two plots:
    1. plot that compares accuracy as function of model type.
    2. plot that compare accuracy as function of imputing method.
    '''

    #cv_results.mean(), cv_results.std(), model_name, rfecv.n_features_

    # best  means
    best_algo = [i[0] for i in results_list]
    index = best_algo.index(max(best_algo))
    print("The best CV accuracy on RFE found with:\n {0} model,  # of features- {1}\n"
           "with accuracy {2} and std {3}".format(results_list[index][2],
                                                  results_list[index][3], results_list[index][0],
                                                  results_list[index][1]))


def RFE_cross_validation(models,X_train,Y_train):
    """

    :param models:List of classification models and their names.
    :param imputing_algo: List of imputing missing data algorithms and their names.
    :param dataset: DataFrame dataset
    :param prints: boolean indicator, determines whether to create printings during the CV.
    create plots of the results from this cross validation
    """
    results = []



    for model_name, model in models:
        if model_name == "SVM_linear":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
        loo = model_selection.LeaveOneOut()
        rfecv = RFECV(estimator=model, step=1, cv=loo, scoring=SCORING)
        X_train_scaled = rfecv.fit_transform(X_train, Y_train)
        cv_results = model_selection.cross_val_score(model, X_train_scaled, Y_train, cv=loo, scoring=SCORING)
        results.append([cv_results.mean(), cv_results.std(), model_name, rfecv.n_features_])
    get_best_result_RFE(results)


def runner(X_train, Y_train):
    RFE_models = get_RFE_models()
    RFE_cross_validation(RFE_models,X_train, Y_train)

    #test_set(dataset)

def looper (path_wise,path_corr,path_other,sheet_name):
    '''

    :param dataset: pandas DataFrame dataset.
    :return: Two np.array sets of Training ang Testsng features and labels.
    slitted psuedo randomly by a 20:80 ratio.
    '''

    dataset_wise = DataFromExcel.refactor_labels(DataFromExcel.get_data(path_wise, sheet_name), "group")
    dataset_wise = imputing_avarage(dataset_wise)

    dataset_corr = DataFromExcel.get_data(path_corr, sheet_name)
    dataset_corr = imputing_avarage(dataset_corr)

    dataset_other = DataFromExcel.get_data(path_other, sheet_name)
    dataset_other = imputing_avarage(dataset_other)
    cols_names = [i for i in dataset_other]
    iter_df = combinations(cols_names,COMB)
    Y_train = dataset_wise["group"]
    X_train_wise = dataset_wise.drop(['Age','group','PHQ9','Subject_Number'],1)
    X_train_corr = PCA_transforme(dataset_corr,3)

    for dropping_titles in iter_df:
        tmp_df = dataset_other.drop([i for i in dropping_titles],1)
        print([i for i in tmp_df])
        X_train_other = PCA_transforme(tmp_df,3,header=0)
        X_train = np.concatenate((X_train_wise.values, X_train_corr,X_train_other), axis=1)
        runner(X_train,Y_train)




path_wise = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\Alls_data_NO_specific_vars_wise.xlsx"
path_corr = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\Alls_data_NO_specific_vars_corr.xlsx"
path_other = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\Alls_data_NO_specific_vars_others.xlsx"
looper(path_wise,path_corr,path_other,SHEET_NAME)
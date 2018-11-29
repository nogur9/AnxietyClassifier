import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.preprocessing import MinMaxScaler

#defining enviroment variables
SEED = 7
SCORING = 'accuracy'
PATH =r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\data_features_for_each_matrix_v1.csv"
SHEET_NAME = "Sheet1"


def get_data (dateset):
    X_train = dateset.drop('group',1)
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



def plot_results_RFE(results_list):
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

    models_name = set([results_list[i][2] for i in range(len(results_list))])
    mean_acc_algos = [np.mean([i[0] for i in results_list if i[2] == j]) for j in models_name]
    mean_std_algos = [np.mean([i[1] for i in results_list if i[2] == j]) for j in models_name]
    N = len(mean_std_algos)
    ind = np.arange(N)
    width = 0.3
    fig, ax = plt.subplots()
    ax.bar(ind, mean_acc_algos, width, color='r', yerr=mean_std_algos)

    ax.set_ylabel('Accuracy')
    ax.set_title('Classifier')
    ax.set_ylim(0.3, 1.5)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(models_name)
    plt.title('RFE -  Classification model')
    plt.show()
    plt.close()


def RFE_cross_validation(models, dataset, prints=1):
    """

    :param models:List of classification models and their names.
    :param imputing_algo: List of imputing missing data algorithms and their names.
    :param dataset: DataFrame dataset
    :param prints: boolean indicator, determines whether to create printings during the CV.
    create plots of the results from this cross validation
    """
    results = []


    X_train, Y_train = get_data(dataset)
    for model_name, model in models:
        #if model_name == "SVM_linear":
            #scaler = MinMaxScaler()
            #X_train = scaler.fit_transform(X_train)
        loo = model_selection.LeaveOneOut()
        rfecv = RFECV(estimator=model, step=1, cv=loo, scoring=SCORING)
        X_train_scaled = rfecv.fit_transform(X_train, Y_train)
        cv_results = model_selection.cross_val_score(model, X_train_scaled, Y_train, cv=loo, scoring=SCORING)
        results.append([cv_results.mean(), cv_results.std(), model_name, rfecv.n_features_])
        if prints:
            print(cv_results)
            print("Optimal number of features : {0}, {1}, acc - {2}".format(rfecv.n_features_, model_name, rfecv.grid_scores_[rfecv.n_features_ - 1]))
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (nb of correct classifications)")
            plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
            plt.show()

    plot_results_RFE(results)


def runner(path,sheet_name):

    dataset = ImportData.refactor_labels(ImportData.get_data(path, sheet_name, csv=1), "group")
    RFE_models = get_RFE_models()
    RFE_cross_validation(RFE_models, dataset)



runner(PATH,SHEET_NAME)
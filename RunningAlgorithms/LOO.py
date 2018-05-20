import xlsxwriter
from sklearn.model_selection import LeaveOneOut
from CalculatingFeaturesFromExcel.PCA import PCA_transforme, meow
# Libraries
import matplotlib.pyplot as plt
import numpy as np

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
PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\Alls_data_NO_specific_vars_wise.xlsx"
#PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\Alls_data_NO_specific_vars_corr.xlsx"
#PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\Alls_data.xlsx"
SHEET_NAME = "Sheet1"

def get_data_no_pca (dataset):
    '''

    :param dataset: pandas DataFrame dataset.
    :return: Two np.array sets of Training ang Testsng features and labels. 
    slitted psuedo randomly by a 20:80 ratio.
    '''
    dataset = imputing_avarage(dataset)
    X_train = dataset.drop(['Age','group','PHQ9','Subject_Number'],1)
    x2 = meow()
    X_train = np.concatenate((X_train.values, x2), axis=1)

    Y_train = dataset["group"]

    return X_train,Y_train

def get_data (dateset):
    X_train = PCA_transforme(dateset,9)
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


def get_select_K_best_models():
    '''

    :return: List of nine classification models and their names.
    '''

    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM_rbf', SVC(kernel='rbf')))
    models.append(('SVM_linear', SVC(kernel='linear')))
    models.append(('GradientBoosting', GradientBoostingClassifier()))
    models.append(('QDA', QuadraticDiscriminantAnalysis()))
    models.append(('RandomForest', RandomForestClassifier()))
    models.append(('AdaBoost', AdaBoostClassifier()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('LogisticRegression', LogisticRegression()))
    return models


def get_imputing_algo():
    '''

    :return: List of four imputing missing data algorithms and their names.
    '''

    inputing_algo = []
    inputing_algo.append(('imputing_median', imputing_median))
    inputing_algo.append(('imputing_avarage', imputing_avarage))
    inputing_algo.append(('imputing_most_frequent', imputing_most_frequent))
    inputing_algo.append(('imputing_knn', imputing_knn))

    return inputing_algo


def get_seleck_K_best_scoring():
    '''

    :return: List of three scoring methods and their names.
    '''

    scoring_models = []
    #scoring_models.append(("chi2", chi2))
    scoring_models.append(("mutual_info", mutual_info_classif))
    scoring_models.append(("f_classif", f_classif))
    return scoring_models


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
    # print to xlsx
    workbook = xlsxwriter.Workbook('C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\LOO_RFE_all_0.05 new.xlsx')
    #workbook = xlsxwriter.Workbook('C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\LOO_RFE_C&L_0.1.xlsx')
    # workbook = xlsxwriter.Workbook('C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\LOO_RFE_C&L_0.1_age/PHQ.xlsx')
    worksheet = workbook.add_worksheet()

    col = 1
    for result in results_list:
        row = 0
        for item in result:
            worksheet.write(row, col, item)
            row += 1
        col += 1

    workbook.close()

    # compare model types
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


def RFE_cross_validation(models, dataset, prints=0):
    """

    :param models:List of classification models and their names.  
    :param imputing_algo: List of imputing missing data algorithms and their names.
    :param dataset: DataFrame dataset
    :param prints: boolean indicator, determines whether to create printings during the CV.
    create plots of the results from this cross validation
    """
    results = []


    #X_train, Y_train = get_data(dataset)
    X_train, Y_train = get_data_no_pca(dataset)
    for model_name, model in models:
        if model_name == "SVM_linear":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
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


def single_model_CV(model_name, model, k, score_name, X_train, Y_train):
    """

    :param model_name: the name of the classification model 
    :param model: the classification model
    :param k: number of features
    :param impute_algo_name: The name of imputing missing data algorithm. 
    :param score_name: the name of the scoring method. 
    :param X_train: np.array set of Training features.
    :param Y_train: np.array set of Training labels.
    :return: List of results.
    """
    if model_name == "SVM_linear" or model_name == "SVM_rbf":
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        kfold = model_selection.KFold(n_splits=3)
        cv_results = model_selection.cross_val_score(model, X_train_scaled, Y_train, cv=kfold, scoring=SCORING)
        return [cv_results.mean(), cv_results.std(), model_name, k, score_name]

    if model_name == "KNN":
        results = []
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        for K in range(3, 15):
            kfold = model_selection.KFold(n_splits=3)
            cv_results = model_selection.cross_val_score(KNeighborsClassifier(n_neighbors=K), X_train_scaled, Y_train,
                                                         cv=kfold, scoring=SCORING)
            results.append([cv_results.mean(), cv_results.std(), K])
        best_K = [i[0] for i in results]
        index = best_K.index(max(best_K))
        return [results[index][0], results[index][1], model_name, k, score_name, results[index][2]]

    kfold = model_selection.KFold(n_splits=3)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=SCORING)
    return [cv_results.mean(), cv_results.std(), model_name, k, score_name]


def plot_select_K_best_results(results_list):
    """
    :param results_list: list of results given from select K best CV function. 

    Prints the classifier with the best accuracy, 
    creates four plots:
    1. plot that compares accuracy as function of model type. 
    2. plot that compare accuracy as function of imputing method.
    3. plot that compares accuraty as function of number of features (k).
    4. plot that compares accuraty as function of select k best's scoring method.
    """
    best_algo = [i[0] for i in results_list]
    index = best_algo.index(max(best_algo))
    #[cv_results.mean(), cv_results.std(), model_name, k, score_name]
    print("The best CV accuracy on select k best found with:\n {0} model,  # of features- {1}\n"
           "scoring method - {2}, with accuracy {3} and std {4}".format(results_list[index][2],
                                                                        results_list[index][3], results_list[index][4],
                                                                        results_list[index][0], results_list[index][1]))

    # print to xlsx
    workbook = xlsxwriter.Workbook('C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\LOO_SKB_all_0.05 new.xlsx')
    #workbook = xlsxwriter.Workbook('C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\LOO_SKB_C&L_0.1.xlsx')
    # workbook = xlsxwriter.Workbook('C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\LOO_SKB_C&L_0.1_age_PHQ.xlsx')
    worksheet = workbook.add_worksheet()

    col = 1
    for result in results_list:
        row = 0
        for item in result:
            worksheet.write(row, col, item)
            row += 1
        col += 1

    workbook.close()


    # comparing model types.
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
    ax.xaxis.set_label_position("bottom")
    plt.title('Select K Best - Classification model')
    plt.show()
    plt.close()

    # comparing number of features.
    ks = range(4, 9)
    mean_acc_ks = [np.mean([i[0] for i in results_list if i[3] == j]) for j in ks]
    std_acc_ks = [np.std([i[0] for i in results_list if i[3] == j]) for j in ks]
    N = len(mean_acc_ks)
    ind = np.arange(N)
    width = 0.4
    fig, ax = plt.subplots()
    ax.bar(ind, mean_acc_ks, width, color='r', yerr=std_acc_ks)
    ax.set_ylabel('Accuracy')
    ax.set_title('Number of features used')
    ax.set_ylim(0.3, 1.5)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(ks)
    ax.xaxis.set_label_position("bottom")
    plt.title('Select K Best - Number of features')
    plt.show()
    plt.close()

    # comparing scoring methods
    scoring_name = set([results_list[i][4] for i in range(len(results_list))])
    mean_acc_scoring = [np.mean([i[0] for i in results_list if i[4] == j]) for j in scoring_name]
    std_acc_scoring = [np.std([i[0] for i in results_list if i[4] == j]) for j in scoring_name]
    N = len(mean_acc_scoring)
    ind = np.arange(N)
    width = 0.4
    fig, ax = plt.subplots()
    ax.bar(ind, mean_acc_scoring, width, color='r', yerr=std_acc_scoring)
    ax.set_ylabel('Accuracy')
    ax.set_title('Scoring method')
    ax.set_ylim(0.3, 1.5)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(scoring_name)
    ax.xaxis.set_label_position("bottom")
    plt.title('Select K Best - Scoring method')
    plt.show()
    plt.close()


def select_K_best_CV(models, scoring_models, dataset):
    """

    :param models: List of classification models and their names.
    :param imputing_algo:List of imputing missing data algorithms and their names.  
    :param scoring_models: List of scoring methods and their names.
    :param dataset:Pandas DataFrame dataset  
    creates plots of the CV results. 
    """
    results_list = []

    #X_train_all, Y_train_all = get_data(dataset)
    X_train_all, Y_train_all = get_data_no_pca(dataset)
    for k in range(4, X_train_all.shape[1]):

        for score_name, scoring_model in scoring_models:
            print("\nk={}".format(k))
            select_feature = SelectKBest(scoring_model, k=k).fit(X_train_all, Y_train_all)
            X_train = select_feature.transform(X_train_all)
            for model_name, model in models:
                #print("\nmodel={}".format(model_name))
                results_list.append(single_model_CV(model_name, model, k, score_name, X_train, Y_train_all))
    plot_select_K_best_results(results_list)


def test_set(dataset):
    """
    :param dataset: Pandas DataFrame dataset    

    Uses - linear SVM, median imputation and RFE feature selection. 
    sclaes the training and the test data by minMax scaler and prints the followings:
    1. features importance. where 1 means XXX and the rest means XXX.
    2. Accuracy on test set
    3. Confusion Matrix
    4. Precision and recall
    """

#    tmp_dataset = imputing_median(dataset)
    X_train, Y_train = get_data(dataset)

#    scaler = MinMaxScaler()
#    X_train_scaled = scaler.fit_transform(X_train)
#    X_test_scaled = scaler.fit_transform(X_test)
    loo = model_selection.LeaveOneOut()
    AdaBoost= AdaBoostClassifier()
    #GB = GradientBoostingClassifier()
    rfecv_ada_boost = RFECV(estimator=AdaBoost, step=1, cv=loo, scoring=SCORING)
    #rfecv_gb = RFECV(estimator=GB, step=1, cv=loo, scoring=SCORING)

    rfecv_ada_boost.fit(X_train, Y_train)
    #rfecv_gb.fit(X_train, Y_train)
    names = ["PHQ9","sum_fixation_length_Disgusted","sum_fixation_length_Neutral","sum_fixation_length_White_Space","mean_top_decile_std_White_Space","mean_top_decile_std_Disgusted",
    "mean_top_decile_std_Neutral",
    "average_fixation_length_Disgusted",
    "average_fixation_length_Neutral",
    "average_fixation_length_White_Space",
    "amount_fixation_Disgusted",
    "amount_fixation_Neutral",
    "amount_fixation_White_Space",
    "STD_fixation_length_Disgusted",
    "STD_fixation_length_Neutral",
    "STD_fixation_length_White_Space",
    "Disgusted_Neutral_ratio"]
    print(len(names))
    svm_feature_importance = sorted([(rfecv_ada_boost.ranking_[i], names[i]) for i in range(len(names))])
    #svm_feature_importance = sorted([(rfecv_gb.ranking_[i], names[i]) for i in range(len(names))])
    print("ada_boost_feature_importance {}".format(svm_feature_importance))
#    print("GB_feature_importance {}".format(svm_feature_importance))

    #
    # # Make predictions on test set
    # SVM_predictions = rfecv_SVM.predict(X_test_scaled)
    # print("Accuracy on test set - {}".format(accuracy_score(Y_test, SVM_predictions)))
    # print("Confusion Matrix (predicted class X actual class)\n {}".format(confusion_matrix(Y_test, SVM_predictions)))
    # print("Classification Report\n {}".format(classification_report(Y_test, SVM_predictions)))
    #

def runner(path,sheet_name):

    dataset = DataFromExcel.refactor_labels(DataFromExcel.get_data(path, sheet_name),"group")
    SKB_models = get_select_K_best_models()
    RFE_models = get_RFE_models()
    SKB_scoring = get_seleck_K_best_scoring()
    RFE_cross_validation(RFE_models, dataset)

    select_K_best_CV(SKB_models, SKB_scoring, dataset)

    #test_set(dataset)



runner(PATH,SHEET_NAME)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from DataImporting import ImportData
from sklearn.neural_network import MLPClassifier
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
SEED = 7
#SCORING = 'accuracy'
SCORING = 'f1'
PATH =r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\data_features_for_each_matrix_features_processed_v2.csv"
SHEET_NAME = "Sheet1"

group_column = 'group'
subject_number_column = 'Subject_Number'


def get_data(dateset):
    X_train = dateset.drop([group_column,subject_number_column], 1)
    Y_train = dateset[group_column]

    return X_train,Y_train



def grid_search(pipe, search_space, X, Y):
    clf = GridSearchCV(pipe, search_space, cv=10, scoring='f1')
    best_model = clf.fit(X, Y)
    #print(clf.best_score_,"\nclf.best_score_")
    #print(best_model.best_estimator_.get_params()['classifier'], "\nbest classifier")
    return best_model

def get_pipeline():
    # Create a pipeline
    pipe = Pipeline([('classifier', RandomForestClassifier())])
    # Create space of candidate learning algorithms and their hyperparameters
    search_space = [#{'classifier': [RandomForestClassifier()]},
                    # {'classifier': [AdaBoostClassifier()]},
                    # {'classifier': [GradientBoostingClassifier()]},
                    # {'classifier': [LinearDiscriminantAnalysis()]},
                    # {'classifier': [KNeighborsClassifier()]},
                    # {'classifier': [SVC(kernel='rbf')]},
                    # {'classifier': [SVC(kernel='linear')]},
                    # {'classifier': [QuadraticDiscriminantAnalysis()]},
                    # {'classifier': [LogisticRegression()]},
                    {'classifier': [MLPClassifier(solver='adam', hidden_layer_sizes=(50, 50, 50, 1))]}
                    ]
    return pipe, search_space


def get_RFE_pipeline(X_train, Y_train):
    lda = SVC(kernel='linear')
    rfecv = RFECV(estimator=lda, step=1, scoring=SCORING)
    rfecv.fit_transform(X_train, Y_train)
    cv_results = model_selection.cross_val_score(rfecv, X_train, Y_train, scoring=SCORING)
    print(cv_results)
    return rfecv


def choose_label(best_model, X_test):
    label_dict = {0:0, 1:0}
    for index, X in X_test.iterrows():
        y = best_model.predict(X.reshape(1,24))
        label_dict[int(y[0])] += 1
    key_max = max(label_dict.keys(), key=(lambda k: label_dict[k]))
    print("label_dict\n", label_dict)
    return key_max


def LOO(df):
    acc_list = []

    i = 0
    for sub in df[subject_number_column].unique():
        i+=1
        train_df = df[df[subject_number_column] != sub]
        X_train = train_df.drop([group_column, subject_number_column], 1)
        Y_train = train_df[group_column]
        #print("i", i)
        test_df = df[df[subject_number_column] == sub]
        X_test = test_df.drop([group_column,subject_number_column], 1)
        Y_test =test_df[group_column][test_df.index[0]]

        pipe, search_space = get_pipeline()
        best_model = grid_search(pipe, search_space, X_train, Y_train)
        Y_hat = choose_label(best_model, X_test)
        #print("y hat", Y_hat, "y test", Y_test)
        acc_list.append(Y_test == Y_hat)
        #print("tmp acc", np.array(acc_list).mean())
    return np.array(acc_list).mean()

def rfecv_LOO(df):
    acc_list = []

    i = 0
    for sub in df[subject_number_column].unique():
        i+=1
        train_df = df[df[subject_number_column] != sub]
        X_train = train_df.drop([group_column, subject_number_column], 1)
        Y_train = train_df[group_column]
        #print("i", i)
        test_df = df[df[subject_number_column] == sub]
        X_test = test_df.drop([group_column,subject_number_column], 1)
        Y_test =test_df[group_column][test_df.index[0]]

        best_model = get_RFE_pipeline(X_train, Y_train)
        Y_hat = choose_label(best_model, X_test)
        #print("y hat", Y_hat, "y test", Y_test)
        acc_list.append(Y_test == Y_hat)
        #print("tmp acc", np.array(acc_list).mean())
    return np.array(acc_list).mean()


def runner(path,sheet_name):

    dataset = ImportData.refactor_labels(ImportData.get_data(path, sheet_name, csv=1), "group")
    dataset = shuffle(dataset)
    print("LOO score", LOO(dataset))






runner(PATH,SHEET_NAME)
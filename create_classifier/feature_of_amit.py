import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from create_classifier.build_grid_search_pipeline import build_pipeline, get_params_grid
from create_classifier.get_grid_search_results import get_reasults_only_clf_pipeline
from dim_reduction.random_forest_selection import get_feature_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.patches as mpatches
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut


import string


def refactor_labels(dataset):

    return  dataset.replace({'low': 0 ,'high': 1, 'clinical': 1 })

def format_filename(s):
    """Take a string and return a valid filename constructed from the string.
Uses a whitelist approach: any characters not present in valid_chars are
removed. Also spaces are replaced with underscores.

Note: this method may produce invalid filenames such as ``, `.` or `..`
When I use this method I prepend a date string like '2009_01_15_19_46_32_'
and append a file extension like '.txt', so I avoid the potential of using
an invalid filename.

"""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')  # I don't like spaces in filenames.
    return filename

#plot everyones features with amit
def plot_amit_scatter(path=""):
    saving_path = r"C:\‏‏PycharmProjects\AnxietyClassifier\visualizations\Amits_plots_4,11"
    sheet_name = 'Sheet1'
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.dropna(axis=1)
    df['group'] = refactor_labels(df['group'])
    high = df[df['group'] == 0]
    low = df[df['group'] == 1]

    feature_names = list(df)
    for feature in feature_names:
        red_patch = mpatches.Patch(color='red', label='Non SAD')
        blue_patch = mpatches.Patch(color='blue', label='SAD')
        plt.legend(handles=[red_patch, blue_patch])
        plt.xlabel(feature)
        plt.ylabel("Amits")
        plt.scatter(high[feature], high["Amits"], c = 'b')
        plt.scatter(low[feature], low["Amits"], c = 'r')
        plt.savefig(os.path.join(saving_path, "Amits-{}.png".format(format_filename(feature))))
        plt.close()

#plot amit high and low

def create_two_hists_by_group(path=""):
    saving_path = r"C:\‏‏PycharmProjects\AnxietyClassifier\visualizations\Amits_plots_4,11"
    sheet_name = 'Sheet1'
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.dropna(axis=1)
    df['group'] = refactor_labels(df['group'])
    high = df[df["group"] == 1]["Amits"]
    #blue
    low = df[df["group"] == 0]["Amits"]
    #red

    bins = np.linspace(min(df["Amits"]),max(df["Amits"]), 100)
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Amit", fontsize=16)
    axs[0].hist(high, bins=bins, color='b')
    axs[0].set_title("high")
    axs[1].hist(low, bins=bins, color='c')
    axs[1].set_title("low")
    plt.savefig(os.path.join(saving_path,"subplot_hist_Amit.png"))
    plt.close()




# get mean, mean by group and std
def get_featue_description(path=""):
    sheet_name = 'Sheet1'
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df[["Amits","group"]]
    df['group'] = refactor_labels(df['group'])
    print(df.describe())
    print("group by")
    print(df.groupby("group").describe())
    print("info")
    print(df.info(verbose=True))
    print("\n\n\n\n")

# outliers
def get_outliers(path=""):
    sheet_name = 'Sheet1'
    df = pd.read_excel(path, sheet_name=sheet_name)
    print("Outliers")
    print(df[(np.abs(df["Amits"] - df["Amits"].mean()) > (3 * df["Amits"].std()))])
    print("\n\n\n\n")

# feature importance
def feature_importance(path=""):
    sheet_name = 'Sheet1'
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.dropna(axis=1)
    Y = refactor_labels(df['group'])
    X = df.drop(['LSAS', 'Subject_Number', 'group'], 1)
    feature_importance = get_feature_importance(X, Y)
    importance_zip = sorted(zip(feature_importance,list(X)), key=lambda x: x[0], reverse=True)
    print("Feature importance")
    print(importance_zip,sep="\n")
    for i, item in enumerate(importance_zip):
        if item[1] == "Amits":
            print("amits feature is in the {} position".format(i))
    print("\n\n\n\n")



# classiff
def classify(path=""):
    sheet_name = 'Sheet1'
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.dropna(axis=1)

    print("only Amits feature")

    Y = refactor_labels(df['group'])
    X = df['Amits']
    pipe = Pipeline([('scaling', MinMaxScaler()), ('classifier', GradientBoostingClassifier())])
    search_space = [{'classifier': [GradientBoostingClassifier()],
                     "classifier__learning_rate": [0.01, 0.2],
                     "classifier__min_samples_split": np.linspace(0.1, 0.5, 2),
                     "classifier__min_samples_leaf": np.linspace(0.1, 0.5, 2),
                     "classifier__subsample": [0.5, 1.0]}
                 ]
    X = X.reshape(-1, 1)
    loo = LeaveOneOut()
    clf = GridSearchCV(pipe, search_space, cv = loo)
    best_model = clf.fit(X, Y)
    print("best classifier", best_model.best_estimator_)
    cv_results = clf.best_score_
    print("best score", cv_results, "\n")
    print("best_params_", best_model.best_params_)
    print(best_model.cv_results_['std_test_score'][best_model.best_index_])
    print("\n\n\n\n")

    print("without Amit feature")
    Y = refactor_labels(df['group'])
    X = df.drop(['LSAS', 'Subject_Number', 'group', 'Amits'], 1)
    pipeline = build_pipeline()
    params_grid = get_params_grid()
    get_reasults_only_clf_pipeline(X, Y, pipeline, params_grid)
    print("\n\n\n\n")


def regg(path=""):
    sheet_name = 'Sheet1'
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.dropna(axis=1)
    df['LSAS']= df['LSAS'].replace(to_replace={-1:np.nan})
    df = df.dropna(axis=0)
    Y = df['LSAS']
    X = df.drop(['LSAS', 'Subject_Number', 'group'], 1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg = lin_reg.fit(X, Y)
    Y_pred = lin_reg.predict(X)
    print("ridge Linear reggression")
    print(lin_reg.score(X, Y))
    print(np.sqrt(mean_squared_error(Y, Y_pred)))
    print("\n\n\n\n")


# feature importance
def STD_feature_importance(path=""):
    sheet_name = 'Sheet1'
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.dropna(axis=1)
    Y = refactor_labels(df['group'])
    X = df.drop(['LSAS', 'Subject_Number', 'group'], 1)
    feature_importance = get_feature_importance(X, Y)
    importance_zip = sorted(zip(feature_importance, list(X)), key=lambda x: x[0], reverse=True)
    print (importance_zip)
    if 0:
        #cutoff high correlation
        corr_matrix = X.corr().abs()
        high_corr = [feature for feature in range(len(corr_matrix['Amits'])) if (np.abs(corr_matrix['Amits'][feature]) > 0.7 and corr_matrix['Amits'].index[feature] != 'Amits')]
        print("high corr\n")
        print(corr_matrix['Amits'].index[high_corr])
        X.drop(X.columns[high_corr], 1, inplace=True)
        print("X.shape", X.shape)
        places_list  = []
        for k in range(1000):
            feature_importance = get_feature_importance(X, Y)
            importance_zip = sorted(zip(feature_importance,list(X)), key=lambda x: x[0], reverse=True)

            print(importance_zip,sep="\n")
            for i, item in enumerate(importance_zip):
                if item[1] == "Amits":
                    places_list.append(i)
        print("Feature importance")
        print("mean",np.mean(places_list),"std", np.std(places_list))


file_path = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeaturesFiles\extracted_features_subjects_set_Updated,with_outlier_subjects_False_with_9029,9014,2018-10-29.xlsx"
#plot_amit_scatter(file_path)
#create_two_hists_by_group(file_path)
#get_featue_description(file_path)
#get_outliers(file_path)
#feature_importance(file_path)
STD_feature_importance(file_path)
#classify(file_path)
#regg(file_path)
import numpy as np
from sklearn import preprocessing
import pandas as pd
from DataImporting.visualize_features import auto_visualize_features
from dim_reduction import random_forest_selection
from dim_reduction.PCA_Obj import PCA_Obj
from sklearn.feature_selection import VarianceThreshold
from DataImporting.ImportData import get_data,refactor_labels

#paths

path = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\subject_features_before_selection.xlsx"
pca_explained_variance_graph_path = "C:\‏‏PycharmProjects\AnxietyClassifier\DataImporting\\visualizations\\subject_features_before_selection\\pca_explained_variance_graph.jpg"
features_after_pca = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\subject_features_after_pca_v1.csv"
feature_importance_txt_path = r"C:\‏‏PycharmProjects\AnxietyClassifier\DataImporting\\visualizations\\subject_features_before_selection"
processed_dataframe_path = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\subject_features_processed_v1.csv"

group_column = 'group'
subject_number_column = 'Subject_Number'


def feature_selection_pipeline_from_file():
    #get data
    dataset = refactor_labels(get_data(path, 'Sheet1'), group_column)

    #remove missing values columns
    dataset = dataset.dropna(axis=1)

    #set X
    X = dataset.drop([group_column, subject_number_column], 1)
    Y = dataset[group_column]
    X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X))

    # all the visualizations
    #auto_visualize_features(dataset.drop([subject_number_column], 1))

    #cutoff by variance
    variance_threshold = 0.03
    variance_cutoff = VarianceThreshold(threshold=variance_threshold)
    variance_cutoff.fit_transform(X)
    print("p1", X.shape)
    #cutoff high correlation
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    X.drop(X.columns[to_drop], 1)

    print("p2",X.shape)
    #PCA
    pca = PCA_Obj(X)
    pca.explained_variance_graph(pca_explained_variance_graph_path)
    n_components = 20
    X = pca.create_pca(n_components)
    pca.save_pca_data(features_after_pca, Y=Y)
    print("p3", X.shape)
    #random forest
    k_best_features = 10
    feature_importance = random_forest_selection.get_feature_importance(X,Y)
    random_forest_selection.save_feature_importance(feature_importance_txt_path, feature_importance)
    processed_dataframe = random_forest_selection.get_k_most_important_features(X,Y,k_best_features,feature_importance)
    print("p4", processed_dataframe.shape)
    processed_dataframe.to_csv(processed_dataframe_path)


def feature_selection_pipeline(features_df, labels):
    '''

    :param features_df: dataframe of features, without subject number
    :param labels: 0-1 labels
    :return:
    '''
    # get data

    # remove missing values columns
    X = features_df.dropna(axis=1)
    Y = labels
    X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X))

    # all the visualizations
    # auto_visualize_features(dataset.drop([subject_number_column], 1))

    # cutoff by variance
    variance_threshold = 0.03
    variance_cutoff = VarianceThreshold(threshold=variance_threshold)
    variance_cutoff.fit_transform(X)

    # cutoff high correlation
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    X.drop(X.columns[to_drop], 1)

    # PCA
    pca = PCA_Obj(X)
    n_components = 20
    X = pca.create_pca(n_components)

    # random forest
    k_best_features = 10
    feature_importance = random_forest_selection.get_feature_importance(X, Y)
    random_forest_selection.save_feature_importance(feature_importance_txt_path, feature_importance)
    processed_dataframe = random_forest_selection.get_k_most_important_features(X, Y, k_best_features, feature_importance)

    return processed_dataframe

feature_selection_pipeline_from_file()
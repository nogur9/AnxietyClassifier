
import numpy as np
from DataImporting.visualize_features import auto_visualize_features
from dim_reduction import random_forest_selection
from dim_reduction.PCA_Obj import PCA_Obj
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from DataImporting.ImportData import get_data,refactor_labels
import pandas as pd
path = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\subject_features_before_selection.xlsx"
pca_explained_variance_graph_path = "C:\‏‏PycharmProjects\AnxietyClassifier\DataImporting\\visualizations\\subject_features_before_selection\\pca_explained_variance_graph.jpg"
features_after_selection = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\subject_features_after_selection_v1.csv"
feature_importance_txt_path = "C:\‏‏PycharmProjects\AnxietyClassifier\DataImporting\\visualizations\\subject_features_before_selection"
group_column = 'group'
subject_number_column = 'Subject_Number'


#get data
dataset = refactor_labels(get_data(path, 'Sheet1'), group_column)

#remove missing values columns
dataset = dataset.dropna(axis=1)

#set X
X = dataset.drop([group_column, subject_number_column], 1)
Y = dataset[group_column]

# all the visualizations
auto_visualize_features(dataset.drop([subject_number_column], 1))

#cutoff by variance
variance_threshold = 0.03
variance_cutoff = VarianceThreshold(threshold=variance_threshold)
variance_cutoff.fit_transform(X)

#cutoff high correlation
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
X.drop(X.columns[to_drop], axis=1)


#PCA
pca = PCA_Obj(X)
pca.explained_variance_graph(pca_explained_variance_graph_path)
n_components = 20
X = pca.create_pca(n_components)
pca.save_pca_data(Y=Y)

#random forest
k_best_features = 10
feature_importance = random_forest_selection.get_feature_importance(X,Y)
random_forest_selection.save_feature_importance(feature_importance_txt_path, feature_importance)
processed_dataframe = random_forest_selection.get_k_most_important_features(X,Y,n_components,feature_importance)


#rfe
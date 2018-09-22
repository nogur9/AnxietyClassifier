import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from DataImporting.ImportData import get_data,refactor_labels
import pandas as pd

class PCA_Obj:
    def __init__(self, data = None, path=None, remove_missing_values=1, group_column='group', subject_number_column= 'Subject_Number'):
        if path:
            self.dataset = refactor_labels(get_data(path, 'Sheet1'), group_column)
            features_df = self.dataset.drop([group_column, subject_number_column], 1)
        if not data is None:
            self.dataset = features_df = data


        if remove_missing_values:
            features_df = features_df.dropna(axis=1)

        X = features_df.values
        self.X = StandardScaler().fit_transform(X)

    def explained_variance_graph(self, path):
        pca = PCA().fit(self.X)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.xticks(np.linspace(0,60,13))
        plt.xlim(0,60)
        plt.grid()
        plt.savefig(path)


    def create_pca(self, k):
        pca = PCA(n_components=k)
        pca.fit(self.X)
        self.transformed_data = pca.transform(self.X)
        return self.transformed_data


    def save_pca_data(self,path, Y=None):
        self.create_new_df(predefined_info_columns= Y).to_csv(path, sep='\t')

    def create_new_df(self, predefined_info_columns=None, group_column='group', subject_number_column= 'Subject_Number'):
        if predefined_info_columns:
            info_columns = predefined_info_columns
        else:
            info_columns = pd.DataFrame(self.dataset[[group_column, subject_number_column]])

        transformed_x = pd.DataFrame(self.transformed_data)
        df = pd.concat([info_columns,transformed_x])
        return df
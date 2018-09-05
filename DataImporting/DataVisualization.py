import random
import numpy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import os
from DataImporting.ImportData import refactor_labels

class DataVisualizationObj:
    dataset = None
    group_column = ""

    def __init__(self, data_set, group_column="group"):
        self.dataset = refactor_labels(data_set, group_column)
        self.group_column = group_column


    def create_binary_hist(self, path = None):
        for feature in self.dataset:
            x = self.dataset[feature][self.dataset[self.group_column] == 1]
            #blue
            y = self.dataset[feature][self.dataset[self.group_column] == 0]
            #red
            bins = numpy.linspace(-10, 10, 100)
            red_patch = mpatches.Patch(color='red', label='low')
            blue_patch = mpatches.Patch(color='blue', label='high')
            plt.legend(handles=[red_patch, blue_patch])
            plt.hist(x, bins, alpha=0.5, label='high')
            plt.hist(y, bins, alpha=0.5, label='low')
            plt.title(feature)
            if path:
                plt.savefig(os.path.join(path,"binary_hist_{}.png".format(str(feature))))
            else:
                plt.show()


    def plot_scatters(self):
        """
    
        :param self.dataset: Pandas DataFrame self.dataset
    
        Makes scatter plot of each two features.
        """
        zero_vals = [self.dataset.values[i] for i in range(len(self.dataset.values)) if self.dataset.values[i][0] == 0.0]
        one_values = [self.dataset.values[i] for i in range(len(self.dataset.values)) if self.dataset.values[i][0] == 1.0]
    
        for i in range(1):
            for j in range(29,34):
                red_patch = mpatches.Patch(color='red', label='Non SAD')
                blue_patch = mpatches.Patch(color='blue', label='SAD')
                plt.legend(handles=[red_patch, blue_patch])
                plt.hold(True)
                plt.xlabel(list(self.dataset)[i])
                plt.ylabel(list(self.dataset)[j])
                plt.scatter([float(zero_vals[k][i]) for k in range(len(zero_vals))],
                            [float(zero_vals[k][j]) for k in range(len(zero_vals))], c='r')
                plt.scatter([float(one_values[k][i]) for k in range(len(one_values))],
                            [float(one_values[k][j]) for k in range(len(one_values))], c='b')
                plt.show()
                plt.hold(False)
                plt.close()

    def box_plot(self):
        """
    
        :param self.dataset: Pandas DataFrame self.dataset
    
        Makes box and whisker plots without the outcome variable.
        """
    
        dataset_without_output = self.dataset.drop(self.group_column, axis=1)
        dataset_without_output.plot(kind='box', subplots=True, layout=(2, 4), sharex=False, sharey=False, Columns=['Age','PHQ9'])
        plt.show()
        plt.close()


    def plot_data(self):
    
        self.dataset.hist()
        plt.show()
    
        scatter_matrix(self.dataset)
        plt.show()


    def plot_corr(self, size=10):
        '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot'''

        corr = self.dataset.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.show()

    def plot_correlation_matrix(self):
        """

        :param dataset: Pandas DataFrame dataset

        creates correlation matrix and prints the correlation that are higher then 0.5
        in comment - printing correlation that are lower then -0.5, but in this data its empty.
        """

        names = list(self.dataset)
        names.remove('group')
        correlations = sorted(self.dataset.corr())
        # plot correlation matrix
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        print("Correlations:")
        print(*[[names[i], names[j], correlations.values[i][j]] for i in range(len(names)) for j in range(len(names))
                if (not i == j) and ((correlations > 0.5).values[i][j] or (correlations < -0.5).values[i][j])], sep="\n")

        fig.colorbar(cax)
        ticks = np.arange(0, 9, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        plt.title('Correlations matrix')
        plt.show()
        plt.close()


    def print_data(self):

        #print(dataset.head(20))
        print(self.dataset.describe())
        print(self.dataset.shape)
        print(self.dataset.groupby(self.group_column).size())


    def print_missing_values(self):
        print("Missing Values")
        print(self.dataset.isnull().sum())


    def print_variance(self):
        max_abs_scaler = preprocessing.MaxAbsScaler()
        data = max_abs_scaler.fit_transform(self.dataset)
        selector = VarianceThreshold()
        selector.fit_transform(data)
        result = sorted(zip(list(self.dataset), selector.variances_), key=lambda x: x[1])
        print("Variance")
        print(*result, sep="\n")

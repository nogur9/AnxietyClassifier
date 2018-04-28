import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import matplotlib.patches as mpatches


def get_data(path, sheet_name):

    dataset = pd.read_excel(path, sheet_name=sheet_name)
    return dataset


def create_new_excel (path, sheet_name):
    df = get_data(path, sheet_name)
    previuos_trial = ""
    for index, row in df.iterrows():
        if row.Stimulus != previuos_trial and index>1:
            df.drop(index-1, inplace=True)
        previuos_trial = row.Stimulus

#    writer = pd.ExcelWriter(CONTROLS_OUTPUT_FILE)
    SAD_OUTPUT_FILE = "SAD_fixation.xlsx"
    writer = pd.ExcelWriter(SAD_OUTPUT_FILE)
    df.to_excel(writer,"Sheet1")
    writer.save()



def print_data(dataset, group_column_name):

    print (dataset.head(20))
    print (dataset.describe())
    print (dataset.shape)
    print(dataset.groupby(group_column_name).size())


def plot_data(dataset):

    dataset.hist()
    plt.show()

    scatter_matrix(dataset)
    plt.show()


def refactor_labels(dataset, group_column_name):

    dataset[[group_column_name]] = dataset[[group_column_name]].replace('low', 0)
    dataset[[group_column_name]] = dataset[[group_column_name]].replace('high', 1)
    dataset[[group_column_name]] = dataset[[group_column_name]].replace('clinical', 1)
    return dataset

def count_missing_data (dataset):

#    dataset= dataset.replace(None, np.NaN)
    print(dataset.isnull().sum())



def plot_scatters(dataset):
    """

    :param dataset: Pandas DataFrame dataset

    Makes scatter plot of each two features.
    """
    zero_vals = [dataset.values[i] for i in range(len(dataset.values)) if dataset.values[i][0] == 0.0]
    one_values = [dataset.values[i] for i in range(len(dataset.values)) if dataset.values[i][0] == 1.0]

    for i in range(2, 19):
        for j in range(2,i):
            red_patch = mpatches.Patch(color='red', label='Non SAD')
            blue_patch = mpatches.Patch(color='blue', label='SAD')
            plt.legend(handles=[red_patch, blue_patch])
            plt.hold(True)
            plt.xlabel(list(dataset)[i])
            plt.ylabel(list(dataset)[j])
            plt.scatter([float(zero_vals[k][i]) for k in range(len(zero_vals))],
                        [float(zero_vals[k][j]) for k in range(len(zero_vals))], c='r')
            plt.scatter([float(one_values[k][i]) for k in range(len(one_values))],
                        [float(one_values[k][j]) for k in range(len(one_values))], c='b')
            plt.show()
            plt.hold(False)
            plt.close()



#dataset = refactor_labels(get_data("C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\formatted_features.xlsx", "Sheet1"),"group")
#plot_scatters(dataset)
#create_new_excel('C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\AmitsData\\SAD.xlsx','fixation results')
#count_missing_data(dataset)
#print_data(dataset, 'group')
#refactor_labels(dataset, 'group')
#plot_data(dataset)
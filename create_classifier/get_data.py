import pandas as pd
import os
import numpy as np


def refactor_labels(df):
    return df.replace({'low': 0 ,'high': 1, 'clinical': 1 })


def get_data(file_name, LSAS_threshold=None):
    group_column = 'group'
    sub_num_col = 'Subject_Number'
    lsas_col = 'LSAS'
    path = r'C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeaturesFiles'
    df = pd.read_excel(os.path.join(path, file_name), sheet_name='Sheet1')
    if LSAS_threshold is None:
        X = df.drop([group_column, sub_num_col, lsas_col], 1)
        Y = refactor_labels(df[group_column])
        return X, Y
    else:
        X = df.drop([group_column], 1)
        Y = pd.Series(np.where(X[lsas_col] > LSAS_threshold, 1, 0))
        X = X.drop([sub_num_col, lsas_col], 1)
        return X, Y


def get_data_for_reg(file_name):
    group_column = 'group'
    sub_num_col = 'Subject_Number'
    lsas_col = 'LSAS'
    path = r'C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeaturesFiles'
    df = pd.read_excel(os.path.join(path, file_name), sheet_name='Sheet1')

    X = df.drop([group_column, sub_num_col, lsas_col], 1)
    Y = df[lsas_col]
    return X, Y

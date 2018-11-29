import pandas as pd
import os
import matplotlib.pyplot as plt
import string
import numpy as np

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

def plot_lsas_scatter(path = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeaturesFiles\extracted_features_subjects_set_Updated,with_outlier_subjects_False_with_9029,9014,2018-10-29.xlsx"):
    saving_path = r"C:\‏‏PycharmProjects\AnxietyClassifier\visualizations\LSAS_plots_4,11"
    sheet_name = 'Sheet1'
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.dropna(axis=1)
    df['LSAS']= df['LSAS'].replace(to_replace={-1:np.nan})
    df = df.dropna(axis=0)

    feature_names = list(df)
    for feature in feature_names:
        plt.xlabel(feature)
        plt.ylabel("LSAS_Total")
        plt.scatter(df[feature], df["LSAS"])
        plt.savefig(os.path.join(saving_path, "LSAS-{}.png".format(format_filename(str(feature)))))
        plt.close()
#plot_lsas_scatter()

import os

from DataImporting.DataVisualization import DataVisualizationObj
from DataImporting.ImportData import get_data
file_a = 'C:\‏‏PycharmProjects\\AnxietyClassifier\ExtractedFeatures\\all_trials_data_features.xlsx'
file_b = 'ExtractedFeatures\\all_trials_data_features_without_first_five_fixations.xlsx'
file_c = 'ExtractedFeatures\\data_features_for_each_matrix.xlsx'
file_d = 'C:\‏‏PycharmProjects\\AnxietyClassifier\ExtractedFeatures\\Features_with_respect_to_trials.xlsx'
file_e = 'C:\‏‏PycharmProjects\\AnxietyClassifier\ExtractedFeatures\\Book1.xlsx'
file_f = 'C:\‏‏PycharmProjects\\AnxietyClassifier\ExtractedFeatures\\subject_features_before_selection.xlsx'
processed_dataframe_path = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\subject_features_processed_v1.csv"
features_after_pca = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\subject_features_after_pca_v1.csv"

def visualize_features_interactivly():
    files = [file_a, file_b, file_c, file_d]
    file_num = input("which file's features do you wanna see?\n {}\n".format(files))
    if file_num == 'a':
        visualization_object = DataVisualizationObj(get_data(file_a, 'Sheet1'))

    elif file_num == 'b':
        visualization_object = DataVisualizationObj(get_data(file_b, 'Sheet1'))

    elif file_num == 'c':
        visualization_object = DataVisualizationObj(get_data(file_c, 'Sheet1'))

    elif file_num == 'd':
        visualization_object = DataVisualizationObj(get_data(file_d, 'Sheet1'))

    else:
        return

    while True:
        func_dict = {1: visualization_object.create_binary_hist, 2: visualization_object.print_data,
                     3: visualization_object.plot_data, 4: visualization_object.plot_corr,
                     5: visualization_object.plot_correlation_matrix}
        try:
            vis_type = int(input("which visualization func do you wanna use?\n {}\n".format(func_dict)))
        except ValueError:
            return

        func_dict[vis_type]()

def auto_visualize_features(data=None):
    if not data is None:
        visualization_object = DataVisualizationObj(data)
    else:
        visualization_object = DataVisualizationObj(get_data(processed_dataframe_path, 'Sheet1', 1))
    visualization_object.print_variance(path="C:\‏‏PycharmProjects\AnxietyClassifier\DataImporting\\visualizations\\v1")
    visualization_object.print_missing_values(path="C:\‏‏PycharmProjects\AnxietyClassifier\DataImporting\\visualizations\\v1")
    visualization_object.create_binary_hist(path = "C:\‏‏PycharmProjects\AnxietyClassifier\DataImporting\\visualizations\\v1")
    visualization_object.create_two_hists_by_group(path = "C:\‏‏PycharmProjects\AnxietyClassifier\DataImporting\\visualizations\\v1")
    visualization_object.plot_correlation_matrix(path="C:\‏‏PycharmProjects\AnxietyClassifier\DataImporting\\visualizations\\v1")


auto_visualize_features()

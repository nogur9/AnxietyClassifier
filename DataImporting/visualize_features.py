from DataImporting.DataVisualization import DataVisualizationObj
from DataImporting.ImportData import get_data
file_a = 'ExtractedFeatures/all_trials_data_features.xlsx'
file_b = 'ExtractedFeatures/all_trials_data_features_without_first_five_fixations.xlsx'
file_c = 'ExtractedFeatures/data_features_for_each_matrix.xlsx'
file_d = 'ExtractedFeatures/Features_with_respect_to_trials.xlsx'

def visualize_features():
    files = [file_a, file_b, file_c, file_d]
    file_num = input("which file's features do you wanna see?\n {}".format(files))
    if file_num == 'a':
        visualization_object = DataVisualizationObj(get_data(file_a))

    elif file_num == 'b':
        visualization_object = DataVisualizationObj(get_data(file_b))

    elif file_num == 'c':
        visualization_object = DataVisualizationObj(get_data(file_c))

    elif file_num == 'd':
        visualization_object = DataVisualizationObj(get_data(file_d))

    else:
        return

    while True:
        func_dict = {1: DataVisualizationObj.create_binary_hist, 2: DataVisualizationObj.print_data,
                     3: DataVisualizationObj.plot_data, 4: DataVisualizationObj.plot_corr,
                     5: DataVisualizationObj.plot_correlation_matrix}
        try:
            vis_type = int(input("which visualization func do you wanna use?\n {}".format(func_dict)))
        except ValueError:
            return

        visualization_object.func_dict[vis_type]()

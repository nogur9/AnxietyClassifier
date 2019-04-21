import os
import pandas as pd
from CalculatingFeaturesFromExcel.ExtractFeatures import Data
from CalculatingFeaturesFromExcel.ExctractFeaturesWRTtrials import TrialsData
import xlsxwriter
import datetime
# optional witch subjects to use

def get_outlier_subjects():
    return [313,319,332,336, 9029, 9014]

def get_only_updated_subjects():
    return [201,202,203,204,205,206,207,208,209,210,211,212]
# generate file name


#combine both options

FIXATION_DATA_SHEET = 'fixation_data'
DEMOGRAPHICS_SHEET = 'demographic'
DATA_FILE_PATH = r"C:\‏‏PycharmProjects\AnxietyClassifier\100_training_set\fixations_training_data.xlsx"


def feature_engineering(subjects_set='Updated', use_outlier_subjects=None, file_name=None, saving_path=r'C:\‏‏PycharmProjects\AnxietyClassifier\100_training_set\feature_engineering_output'):

    agg_features_extractor = Data(DATA_FILE_PATH, FIXATION_DATA_SHEET, DEMOGRAPHICS_SHEET)
    # reg_features_extractor = TrialsData(DATA_FILE_PATH, FIXATION_DATA_SHEET, DEMOGRAPHICS_SHEET)

    excluding_subjects_list = []

    if subjects_set == 'Amits':
        if use_outlier_subjects is None:
            use_outlier_subjects = False
        excluding_subjects_list.extend(get_only_updated_subjects())

    elif subjects_set == 'Updated':
        if use_outlier_subjects is None:
            use_outlier_subjects = False

    if not use_outlier_subjects:
        excluding_subjects_list.extend(get_outlier_subjects())


    agg_features_extractor.get_matrix_count_independant_features()
    # reg_features_extractor.get_matrix_count_independant_features()
    agg_features_df = pd.DataFrame(agg_features_extractor.output_data_dict)
    # reg_features_df =pd.DataFrame(reg_features_extractor.output_data_dict)
    combained_features = agg_features_df  # .merge(reg_features_df, left_on='Subject_Number', right_on='Subject_Number')


    combained_features = combained_features[combained_features['Subject_Number'].map(lambda x: x not in excluding_subjects_list)]


    if file_name is None:
        file_name = "training set cutoff30{}".format(datetime.datetime.now().strftime('%Y-%m-%d'))
    workbook = xlsxwriter.Workbook(os.path.join(saving_path,'{}.xlsx'.format(file_name)), options={'nan_inf_to_errors':True})

    worksheet = workbook.add_worksheet()
    col = 0
    for key in combained_features.keys():
        row = 0
        worksheet.write(row, col, key)
        for item in combained_features[key]:
            row += 1
            worksheet.write(row, col, item)
        col += 1

    workbook.close()

feature_engineering(subjects_set='Updated')

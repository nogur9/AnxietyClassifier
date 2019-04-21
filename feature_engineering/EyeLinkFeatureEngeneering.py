import os
import pandas as pd
from CalculatingFeaturesFromExcel.ExtractFeatures import Data
from CalculatingFeaturesFromExcel.ExctractFeaturesWRTtrials import TrialsData
import xlsxwriter
import datetime


#combine both options

FIXATION_DATA_SHEET = 'fixation_data'
DEMOGRAPHICS_SHEET = 'demographic'


def feature_engineering(data_file_path, saving_path="C:\‏‏PycharmProjects\AnxietyClassifier\\100_training_set\\feature_engineering_output"):

    agg_features_extractor = Data(data_file_path, FIXATION_DATA_SHEET, DEMOGRAPHICS_SHEET)

    agg_features_extractor.get_features_for_prediction()

    agg_features_df = pd.DataFrame(agg_features_extractor.output_data_dict)
    combained_features = agg_features_df

    file_name = "extracted_eye_link_features_gals_training_set_cutoff_30__{}.xlsx".format(datetime.datetime.now().strftime('%Y-%m-%d'))

    workbook = xlsxwriter.Workbook(os.path.join(saving_path, file_name), options={'nan_inf_to_errors': True})

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

path = r"C:\‏‏PycharmProjects\AnxietyClassifier\100_training_set\eyelink_proccessor_output\\gals training set 2019-04-17.xlsx"
feature_engineering(path)

import os
import pandas as pd
from CalculatingFeaturesFromExcel.ExtractFeatures import Data
from CalculatingFeaturesFromExcel.ExctractFeaturesWRTtrials import TrialsData
import xlsxwriter
import datetime


#combine both options

FIXATION_DATA_SHEET = 'fixation_data'
DEMOGRAPHICS_SHEET = 'demographic'


def feature_engineering(data_file_path, saving_path=r'C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeaturesFiles'):

    agg_features_extractor = Data(data_file_path, FIXATION_DATA_SHEET, DEMOGRAPHICS_SHEET)
    reg_features_extractor = TrialsData(data_file_path, FIXATION_DATA_SHEET, DEMOGRAPHICS_SHEET)

    agg_features_extractor.get_features_for_prediction()
    reg_features_extractor.get_all_good_features()
    agg_features_df = pd.DataFrame(agg_features_extractor.output_data_dict)
    reg_features_df =pd.DataFrame(reg_features_extractor.output_data_dict)
    combained_features = agg_features_df.merge(reg_features_df, left_on='Subject_Number', right_on='Subject_Number')

    file_name = "extracted_eye_link_features_subjects__{}_{}.xlsx".format(combained_features['Subject_Number'].unique(), datetime.datetime.now().strftime('%Y-%m-%d'))

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


feature_engineering("C:\‏‏PycharmProjects\AnxietyClassifier\OmersData\extracted eye link data 2018-12-02.xlsx")

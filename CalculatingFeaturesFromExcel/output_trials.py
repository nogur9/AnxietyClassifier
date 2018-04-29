import pandas as pd
from CalculatingFeaturesFromExcel.ExtractFeatures import Data
from CalculatingFeaturesFromExcel.ExctractFeaturesWRTtrials import TrialsData
from CalculatingFeaturesFromExcel.RegressionFunctions import sine
#from pandas import ExcelWriter
import xlsxwriter
import numpy as np

#נבדקים:
#311 - ציון LSAS - 48  סווג לקבוצת גבוהים בחרדה
#319 - ציון LSAS- 36 סווג לקבוצת גבוהים בחרדה
#332 - ציון LSAS - 53 סווג לקבוצת נמוכים בחרדה
#336 - ציון LSAS - 36 סווג לקבוצת נמוכים בחרדה

#Define Envairoment Variables
CONTROLS_FILE_PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\AmitsData\\Controls.xlsx"
FIXATION_DATA_SHEET = 'fixation data'
DEMOGRAPHICS_SHEET = 'Final all Results'
SAD_FILE_PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\AmitsData\\SAD.xlsx"
def create_Controls_File_Features():
    print(1)
    controls = Data(CONTROLS_FILE_PATH, FIXATION_DATA_SHEET,DEMOGRAPHICS_SHEET)
    print(2)
    controls.get_subject_number()
    print(6)
    controls.get_group()
    print(7)
    Trials_Controls = TrialsData(CONTROLS_FILE_PATH, FIXATION_DATA_SHEET)
    results_avg = Trials_Controls.get_average_fixation_length_each_trial()
    sine(results_avg[0])
    results_sums = Trials_Controls.get_sum_fixation_length()
    sine(results_sums[0])
    results_std = Trials_Controls.get_STD_fixation_length()
    sine(results_std[0])
    results_ratio = Trials_Controls.get_Ratios()
    sine(results_ratio[0])
    results_amounts = Trials_Controls.get_amount_fixation_length()
    sine(results_amounts[0])
    results_aois = Trials_Controls.get_mean_different_AOI_per_trial()
    sine(results_aois[0])

    workbook = xlsxwriter.Workbook('C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\formatted_features.xlsx')
    worksheet = workbook.add_worksheet()


    col = 0
    for key in controls.output_data_dict.keys():
        row = 0
        worksheet.write(row, col, key)
        for item in controls.output_data_dict[key]:
            row += 1
            worksheet.write(row, col, item)
        col += 1

    workbook.close()


def create_SAD_File_Features():
    print(1)
    SAD = Data(SAD_FILE_PATH, FIXATION_DATA_SHEET,DEMOGRAPHICS_SHEET)
    SAD.get_subject_number()
    print(5)
    SAD.get_group()
    print (6)

    workbook = xlsxwriter.Workbook('C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\formatted_features_SAD_updated.xlsx')
    worksheet = workbook.add_worksheet()
    col = 0
    for key in SAD.output_data_dict.keys():
        row = 0
        worksheet.write(row, col, key)
        for item in SAD.output_data_dict[key]:
            row += 1
            worksheet.write(row, col, item)
        col += 1

    workbook.close()

create_SAD_File_Features()
print("half!!!!!!")
create_Controls_File_Features()

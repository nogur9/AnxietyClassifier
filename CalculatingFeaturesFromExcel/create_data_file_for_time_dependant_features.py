from CalculatingFeaturesFromExcel.ExtractFeatures import Data
from CalculatingFeaturesFromExcel.ExctractFeaturesWRTtrials import TrialsData
from CalculatingFeaturesFromExcel.RegressionFunctions import sine
#from pandas import ExcelWriter
import xlsxwriter
import math

#נבדקים:
#311 - ציון LSAS - 48  סווג לקבוצת גבוהים בחרדה
#319 - ציון LSAS- 36 סווג לקבוצת גבוהים בחרדה
#332 - ציון LSAS - 53 סווג לקבוצת נמוכים בחרדה
#336 - ציון LSAS - 36 סווג לקבוצת נמוכים בחרדה

#Define Envairoment Variables
FIXATION_DATA_SHEET = 'fixation_data'
DEMOGRAPHICS_SHEET = 'demographic'
DATA_FILE_PATH = r"C:\‏‏PycharmProjects\AnxietyClassifier\AmitsData\data.xlsx"


def create_features_file():
    print(1)
    controls = Data(DATA_FILE_PATH, FIXATION_DATA_SHEET,DEMOGRAPHICS_SHEET)
    print(2)
    controls.get_group()
    print(7)
    controls.get_subject_number()
    print(6)
    trials_controls = TrialsData(DATA_FILE_PATH, FIXATION_DATA_SHEET)
    print(8)
    trials_controls.get_Ratios()
    print(9)
    trials_controls.get_average_fixation_length_each_trial()
    print(10)
    trials_controls.get_STD_fixation_length()
    print(11)
    trials_controls.get_amount_fixations()
    print(12)
    trials_controls.get_mean_different_AOI_per_trial()
    print(13)
    workbook = xlsxwriter.Workbook(
        'C:\\‏‏PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\Features_with_respect_to_trials.xlsx')
    worksheet = workbook.add_worksheet()
    col = 0
    for key in controls.output_data_dict.keys():
        row = 0
        worksheet.write(row, col, key)
        for item in controls.output_data_dict[key]:
            row += 1
            worksheet.write(row, col, item)
        col += 1

    for key in trials_controls.output_data_dict.keys():
        row = 0
        worksheet.write(row, col, key)
        for item in trials_controls.output_data_dict[key]:
            if math.isnan(item):
                row += 1
            else:
                row += 1
                worksheet.write(row, col, item)
        col += 1

    workbook.close()

create_features_file()
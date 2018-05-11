import pandas as pd
from CalculatingFeaturesFromExcel.ExtractFeatures import Data
#from pandas import ExcelWriter
import xlsxwriter
import numpy as np

#נבדקים:
#311 - ציון LSAS - 48  סווג לקבוצת גבוהים בחרדה
#319 - ציון LSAS- 36 סווג לקבוצת גבוהים בחרדה
#332 - ציון LSAS - 53 סווג לקבוצת נמוכים בחרדה
#336 - ציון LSAS - 36 סווג לקבוצת נמוכים בחרדה

#Define Envairoment Variables
#CONTROLS_FILE_PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\AmitsData\\Controls.xlsx"
#CONTROLS_FILE_PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\\Nisans Data\\Nisan_pre.xlsx"
CONTROLS_FILE_PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier(2)\\Testers\\NogasData.xlsx"
#FIXATION_DATA_SHEET = 'fixation data'
FIXATION_DATA_SHEET = 'Sheet1'
#DEMOGRAPHICS_SHEET = 'Final all Results'
SAD_FILE_PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\AmitsData\\SAD.xlsx"
def create_Controls_File_Features():
    print(1)
    controls = Data(CONTROLS_FILE_PATH, FIXATION_DATA_SHEET)
    print(2)
    controls.amount_of_second_fixations()
    print(3)
#    controls.get_age()
    print(4)
#    controls.get_PHQ9()
    print(5)
    controls.get_subject_number()
    print(6)
#    controls.get_group()
    print(7)
    controls.get_sum_fixation_length_Disgusted()
    print(8)
    controls.get_sum_fixation_length_Neutral()
    print(9)
    controls.get_sum_fixation_length_White_Space()
    print(10)
    controls.get_average_fixation_length_Disgusted()
    print(11)
    controls.get_average_fixation_length_Neutral()
    print(12)
    controls.get_average_fixation_length_White_Space()
    print(13)
    controls.get_amount_fixation_Disgusted()
    print(14)
    controls.get_amount_fixation_Neutral()
    print(15)
    controls.get_amount_fixation_White_Space()
    print(16)
    controls.get_STD_fixation_length_Disgusted()
    print(17)
    controls.get_STD_fixation_length_Neutral()
    print(18)
    controls.get_STD_fixation_length_White_Space()
    print(19)
    controls.get_STD_fixation_length_All()
    print(20)
    controls.get_ratio_D_DN()
    print(21)
    controls.get_ratio_N_DN()
    print(22)
    controls.get_ratio_WS_All()
    print(23)
    controls.get_ratio_D_DN_2()
    print(24)
    controls.get_ratio_N_DN_2()
    print(25)
    controls.get_amount_DN_transitions()
    print(26)
    controls.get_amount_ND_transitions()
    print(27)
    controls.get_amount_DD_transitions()
    print(28)
    controls.get_amount_NN_transitions()
    print(29)
    controls.get_amount_diff_AOI_transitions()
    print(30)
    controls.var_threat_precentage_between_trials()
    print(31)
    controls.amount_of_first_fixations()
    print(32)

    controls.get_average_pupil_size_Disgusted()
    print(33)
    controls.get_average_pupil_size_Neutral()
    print(34)
    controls.get_average_pupil_size_White_Space()
    print(35)
    controls.get_average_pupil_size_All()
    print(36)
    controls.get_STD_pupil_size_Disgusted()
    print(37)
    controls.get_STD_pupil_size_Neutral()
    print(38)
    controls.get_STD_pupil_size_White_Space()
    print(39)
    controls.get_STD_pupil_size_All()
    print(40)
    controls.get_mean_different_AOI_per_trial()
    print(41)
    #workbook = xlsxwriter.Workbook('C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\Nisan_formatted_features_pre.xlsx')
    workbook = xlsxwriter.Workbook(
        'C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\Noga_formatted_features_test.xlsx')
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


    print(2)
    SAD.get_age()
    print(3)
    SAD.get_PHQ9()
    print(4)
    SAD.get_subject_number()
    print(5)
    SAD.get_group()
    print (6)
    SAD.get_sum_fixation_length_Disgusted()
    print (7)
    SAD.get_sum_fixation_length_Neutral()
    print (8)
    SAD.get_sum_fixation_length_White_Space()
    print(9)
    print(10)
    SAD.get_average_fixation_length_Disgusted()
    print(11)
    SAD.get_average_fixation_length_Neutral()
    print(12)
    SAD.get_average_fixation_length_White_Space()
    print(13)
    print(14)
    print(10)
    SAD.get_amount_fixation_Disgusted()
    print(13)
    SAD.get_amount_fixation_Neutral()
    print(14)
    SAD.get_amount_fixation_White_Space()

    print (15)
    SAD.get_STD_fixation_length_Disgusted()
    print (16)
    SAD.get_STD_fixation_length_Neutral()
    print (17)
    SAD.get_STD_fixation_length_White_Space()
    print(197)
    SAD.get_STD_fixation_length_All()
    print (18)
    SAD.get_ratio_D_DN()
    print(20)
    SAD.get_ratio_N_DN()
    print(21)
    SAD.get_ratio_WS_All()
    print(22)
    SAD.get_ratio_D_DN_2()
    print(23)
    SAD.get_ratio_N_DN_2()
    print(24)
    SAD.get_amount_DN_transitions()
    print(25)
    SAD.get_amount_ND_transitions()
    print(26)
    SAD.get_amount_DD_transitions()
    print(27)
    SAD.get_amount_NN_transitions()
    print(28)
    SAD.get_amount_diff_AOI_transitions()
    print(29)
    SAD.var_threat_precentage_between_trials()
    print(30)
    SAD.amount_of_first_fixations()
    print(31)
    #SAD.amount_of_second_fixations()
    print(32)
    SAD.get_average_pupil_size_Disgusted()
    print(33)
    SAD.get_average_pupil_size_Neutral()
    print(34)
    SAD.get_average_pupil_size_White_Space()
    print(35)
    SAD.get_average_pupil_size_All()
    print(36)
    SAD.get_STD_pupil_size_Disgusted()
    print(37)
    SAD.get_STD_pupil_size_Neutral()
    print(38)
    SAD.get_STD_pupil_size_White_Space()
    print(39)
    SAD.get_STD_pupil_size_All()
    print(40)
    SAD.get_mean_different_AOI_per_trial()
    print (41)
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

#create_SAD_File_Features()
#print("half!!!!!!")
create_Controls_File_Features()

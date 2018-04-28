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
CONTROLS_FILE_PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\AmitsData\\Controls.xlsx"
FIXATION_DATA_SHEET = 'fixation data'
DEMOGRAPHICS_SHEET = 'Final all Results'
SAD_FILE_PATH = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\AmitsData\\SAD.xlsx"
def create_Controls_File_Features():
    print(1)
    controls = Data(CONTROLS_FILE_PATH, FIXATION_DATA_SHEET,DEMOGRAPHICS_SHEET)

    controls.get_group()
    print(2)
    controls.get_subject_number()
    print(3)

    print (4)
    controls.get_PHQ9()
    print (5)
    controls.get_age()

    print (6)
    controls.get_sum_fixation_length_Disgusted()
    print (7)
    controls.get_sum_fixation_length_Neutral()
    print (8)
    controls.get_sum_fixation_length_White_Space()

    print ("top STDs WS")
    controls.get_mean_top_decile_std_White_Space()

    print ("top STDs D")
    controls.get_mean_top_decile_std_Disgusted()

    print ("top STDs N")
    controls.get_mean_top_decile_std_Neutral()

    print (9)
    controls.get_average_fixation_length_Disgusted()
    print (10)
    controls.get_average_fixation_length_Neutral()
    print (11)
    controls.get_average_fixation_length_White_Space()

    print (12)
    controls.get_amount_fixation_Disgusted()
    print (13)
    controls.get_amount_fixation_Neutral()
    print (14)
    controls.get_amount_fixation_White_Space()

    print (15)
    controls.STD_fixation_length_Disgusted()
    print (16)
    controls.STD_fixation_length_Neutral()
    print (17)
    controls.STD_fixation_length_White_Space()

    print (18)
    controls.get_Disgusted_Neutral_ratio()



    print (22)
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


    # writer = ExcelWriter(CONTROLS_OUTPUT_FILE)
    # print (23)
    # to_excel = pd.DataFrame.from_dict(controls.output_data_dict)
    # print (24)
    # to_excel.to_excel(writer,"sheet1")
    # print (25)
    # writer.save()


def create_SAD_File_Features():
    print(1)
    SAD = Data(SAD_FILE_PATH, FIXATION_DATA_SHEET,DEMOGRAPHICS_SHEET)

    SAD.get_group()
    print(2)
    SAD.get_subject_number()
    print(3)

    print (4)
    SAD.get_PHQ9()
    print (5)
    SAD.get_age()

    print (6)
    SAD.get_sum_fixation_length_Disgusted()
    print (7)
    SAD.get_sum_fixation_length_Neutral()
    print (8)
    SAD.get_sum_fixation_length_White_Space()


    print ("top STDs WS")
    SAD.get_mean_top_decile_std_White_Space()

    print ("top STDs D")
    SAD.get_mean_top_decile_std_Disgusted()

    print ("top STDs N")
    SAD.get_mean_top_decile_std_Neutral()

    print (9)
    SAD.get_average_fixation_length_Disgusted()
    print (10)
    SAD.get_average_fixation_length_Neutral()
    print (11)
    SAD.get_average_fixation_length_White_Space()

    print (12)
    SAD.get_amount_fixation_Disgusted()
    print (13)
    SAD.get_amount_fixation_Neutral()
    print(14)
    SAD.get_amount_fixation_White_Space()
    print(15)
    SAD.STD_fixation_length_Disgusted()
    print(16)
    SAD.STD_fixation_length_Neutral()
    print(17)
    SAD.STD_fixation_length_White_Space()
    print(18)
    SAD.get_Disgusted_Neutral_ratio()



    print (22)
    workbook = xlsxwriter.Workbook('C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\formatted_features_SAD.xlsx')
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

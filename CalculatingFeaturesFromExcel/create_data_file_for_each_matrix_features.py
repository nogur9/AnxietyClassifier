from CalculatingFeaturesFromExcel.ExtractFeaturesForEachMatrix import Data
import xlsxwriter

# נבדקים שהוצאו:
# 311 - ציון LSAS - 48  סווג לקבוצת גבוהים בחרדה
# 319 - ציון LSAS- 36 סווג לקבוצת גבוהים בחרדה
# 332 - ציון LSAS - 53 סווג לקבוצת נמוכים בחרדה
# 336 - ציון LSAS - 36 סווג לקבוצת נמוכים בחרדה

# Define Envairoment Variables
FIXATION_DATA_SHEET = 'fixation_data'
DEMOGRAPHICS_SHEET = 'demographic'
DATA_FILE_PATH = "C:\‏‏PycharmProjects\AnxietyClassifier\AmitsData\data.xlsx"

def create_data_File_Features():
    print(1)
    data = Data(DATA_FILE_PATH, FIXATION_DATA_SHEET, DEMOGRAPHICS_SHEET)
    data.get_average_pupil_size_Disgusted()
    print(2)
    data.get_difference_between_medians()
    print(4)
    data.get_subject_number()
    print(5)
    data.get_group()
    print(6)
    data.get_sum_fixation_length_Disgusted()
    print(7)
    data.get_sum_fixation_length_Neutral()
    print(8)
    data.get_sum_fixation_length_White_Space()
    print(9)
    data.get_average_fixation_length_Disgusted()
    print(11)
    data.get_average_fixation_length_Neutral()
    print(12)
    data.get_average_fixation_length_White_Space()
    print(13)

    data.get_amount_fixation_Disgusted()
    print(13)
    data.get_amount_fixation_Neutral()
    print(14)
    data.get_amount_fixation_White_Space()
    print(15)

    data.get_STD_fixation_length_Disgusted()
    print(16)
    data.get_STD_fixation_length_Neutral()
    print(17)
    data.get_STD_fixation_length_White_Space()
    print(197)
    data.get_STD_fixation_length_All()
    print(18)

    data.get_ratio_D_DN()
    print(20)
    data.get_ratio_N_DN()
    print(21)
    data.get_ratio_WS_All()
    print(22)
    data.get_ratio_D_DN_2()
    print(23)
    data.get_ratio_N_DN_2()
    print(24)
    print(33)
    data.get_average_pupil_size_Neutral()
    print(34)
    data.get_average_pupil_size_White_Space()
    print(35)
    data.get_average_pupil_size_All()
    print(36)
    data.get_STD_pupil_size_Disgusted()
    print(37)
    data.get_STD_pupil_size_Neutral()
    print(38)
    data.get_STD_pupil_size_White_Space()
    print(39)
    data.get_STD_pupil_size_All()
    print(41)
    print(42)
    workbook = xlsxwriter.Workbook(
        'C:\\‏‏PycharmProjects\\AnxietyClassifier\\ExtractedFeatures\\data_features_for_each_matrix.xlsx')
    worksheet = workbook.add_worksheet()
    col = 0
    for key in data.output_data_dict.keys():
        row = 0
        worksheet.write(row, col, key)
        for item in data.output_data_dict[key]:
            row += 1
            worksheet.write(row, col, item)
        col += 1

    workbook.close()


create_data_File_Features()


DATA_FILE_PATH_WITHOUT_FIRST_FIVE_FIXATIONS = "C:\‏‏PycharmProjects\AnxietyClassifier\AmitsData\data_cut_first_five_fixations.xlsx"

def create_data_File_Features_without_first_five_fixations():
    print(1)
    data = Data(DATA_FILE_PATH_WITHOUT_FIRST_FIVE_FIXATIONS, FIXATION_DATA_SHEET, DEMOGRAPHICS_SHEET)
    print(2)
    data.get_age()
    print(3)
    data.get_PHQ9()
    print(4)
    data.get_subject_number()
    print(5)
    data.get_group()
    print(6)

    data.get_sum_fixation_length_Disgusted()
    print(7)
    data.get_sum_fixation_length_Neutral()
    print(8)
    data.get_sum_fixation_length_White_Space()
    print(9)

    data.get_average_fixation_length_Disgusted()
    print(11)
    data.get_average_fixation_length_Neutral()
    print(12)
    data.get_average_fixation_length_White_Space()
    print(13)

    data.get_amount_fixation_Disgusted()
    print(13)
    data.get_amount_fixation_Neutral()
    print(14)
    data.get_amount_fixation_White_Space()
    print(15)

    data.get_STD_fixation_length_Disgusted()
    print(16)
    data.get_STD_fixation_length_Neutral()
    print(17)
    data.get_STD_fixation_length_White_Space()
    print(197)
    data.get_STD_fixation_length_All()
    print(18)

    data.get_ratio_D_DN()
    print(20)
    data.get_ratio_N_DN()
    print(21)
    data.get_ratio_WS_All()
    print(22)
    data.get_ratio_D_DN_2()
    print(23)
    data.get_ratio_N_DN_2()
    print(24)

    data.get_average_pupil_size_Disgusted()
    print(33)
    data.get_average_pupil_size_Neutral()
    print(34)
    data.get_average_pupil_size_White_Space()
    print(35)
    data.get_average_pupil_size_All()
    print(36)
    data.get_STD_pupil_size_Disgusted()
    print(37)
    data.get_STD_pupil_size_Neutral()
    print(38)
    data.get_STD_pupil_size_White_Space()
    print(39)
    data.get_STD_pupil_size_All()
    print(41)

    data.get_difference_between_medians()
    print(42)
    workbook = xlsxwriter.Workbook(
        'C:\PycharmProjects\AnxietyClassifier\ExtractedFeatures\\data_features_for_each_matrix.xlsx')
    worksheet = workbook.add_worksheet()
    col = 0
    for key in data.output_data_dict.keys():
        row = 0
        worksheet.write(row, col, key)
        for item in data.output_data_dict[key]:
            row += 1
            worksheet.write(row, col, item)
        col += 1

    workbook.close()
create_data_File_Features()
#create_data_File_Features_without_first_five_fixations()
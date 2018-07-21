from CalculatingFeaturesFromExcel.ExtractFeatures import Data
import xlsxwriter

#נבדקים שהוצאו:
#311 - ציון LSAS - 48  סווג לקבוצת גבוהים בחרדה
#319 - ציון LSAS- 36 סווג לקבוצת גבוהים בחרדה
#332 - ציון LSAS - 53 סווג לקבוצת נמוכים בחרדה
#336 - ציון LSAS - 36 סווג לקבוצת נמוכים בחרדה

#Define Envairoment Variables
FIXATION_DATA_SHEET = 'fixation data'
DEMOGRAPHICS_SHEET = 'Final all Results'
DATA_FILE_PATH = "C:\‏‏PycharmProjects\AnxietyClassifier\AmitsData\data.xlsx"
TOMS_DATA_FILE_PATH = "C:\‏‏PycharmProjects\AnxietyClassifier\AmitsData\9020_for_Noga.xls"
CUT_FIRST_FIVE_FIX_FILE_PATH ="C:\PycharmProjects\AnxietyClassifier\AmitsData\data_cut_first_five_fixations.xlsx"
    
def create_data_File_Features():
    print(1)
    data = Data(DATA_FILE_PATH, FIXATION_DATA_SHEET, DEMOGRAPHICS_SHEET)
    print(2)
    data.get_age()
    print(3)
#    data.get_PHQ9()
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
    print(10)
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
    data.get_amount_DN_transitions()
    print(25)
    data.get_amount_ND_transitions()
    print(26)
    data.get_amount_DD_transitions()
    print(27)
    data.get_amount_NN_transitions()
    print(28)
    data.get_amount_diff_AOI_transitions()
    print(29)
    data.var_threat_precentage_between_trials()
    print(30)
    data.amount_of_first_fixations()
    print(31)
    data.amount_of_second_fixations()
    print(32)
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
    print(40)
    data.get_mean_different_AOI_per_trial()
    print(41)
    workbook = xlsxwriter.Workbook(
        'C:\PycharmProjects\AnxietyClassifier\ExtractedFeatures\exctracted_data_features.xlsx')
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


def create_only_first_five_fixations_data_File_Features():
    print(1)
    data = Data(CUT_FIRST_FIVE_FIX_FILE_PATH, FIXATION_DATA_SHEET, DEMOGRAPHICS_SHEET)
    print(4)
    data.get_subject_number()
    print(6)
    data.get_sum_fixation_length_Disgusted()
    print(7)
    data.get_sum_fixation_length_Neutral()
    print(9)
    print(10)
    data.get_average_fixation_length_Disgusted()
    print(11)
    data.get_average_fixation_length_Neutral()
    print(12)
    print(13)
    print(14)
    print(10)
    data.get_amount_fixation_Disgusted()
    print(13)
    data.get_amount_fixation_Neutral()
    print(14)
    print(15)
    data.get_STD_fixation_length_Disgusted()
    print(16)
    data.get_STD_fixation_length_Neutral()
    print(17)
    print(18)
    data.get_ratio_D_DN()
    print(20)
    data.get_ratio_N_DN()
    data.var_threat_precentage_between_trials()
    print(32)
    data.get_average_pupil_size_Disgusted()
    print(33)
    data.get_average_pupil_size_Neutral()
    print(34)
    print(35)
    data.get_mean_different_AOI_per_trial()
    print(41)
    workbook = xlsxwriter.Workbook(
        'C:\PycharmProjects\AnxietyClassifier\ExtractedFeatures\exctracted_data_features_CUT_FIRST_FIVE.xlsx')
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

def create_Toms_fixations_data_File_Features():
    print(1)
    data = Data(TOMS_DATA_FILE_PATH, "Sheet 1")
    print(4)
    #data.get_trial()
    print(6)
    data.get_DT_each_stimulus_pet_trial()
    print(7)
    workbook = xlsxwriter.Workbook(
        'Exctracted_data_features_Toms_9020.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    for key in data.output_data_dict.keys():
        col = 0
        worksheet.write(row, col, key)
        for item in data.output_data_dict[key]:
            col += 1
            worksheet.write(row, col, item)
        row += 1

    workbook.close()


#create_data_File_Features()
#print("half!!!!!!")
create_Toms_fixations_data_File_Features()
#create_data_File_Features()

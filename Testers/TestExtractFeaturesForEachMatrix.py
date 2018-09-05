import unittest
from DataImporting.ImportData import get_data
from CalculatingFeaturesFromExcel.ExtractFeaturesForEachMatrix import Data
Test_Data_path = "C:\‏‏PycharmProjects\AnxietyClassifier\Testers\TestDataForEachMatrixFeatureExtraction.xlsx"
FIXATION_DATA_SHEET = 'fixation_data'
#DEMOGRAPHICS_SHEET = 'Final all Results'


class TestFeatureExtraction(unittest.TestCase):
    Data_Object = None

    def test_get_sum_fixation_length_Disgusted(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        #act
        self.Data_Object.get_sum_fixation_length_Disgusted()
        #assert
        self.assertEqual(self.Data_Object.output_data_dict["sum_fixation_length_Disgusted"], [1450, 1912, 1529])

    def test_get_sum_fixation_length_Disgusted_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        #act
        self.Data_Object.get_sum_fixation_length_Disgusted()
        #assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_sum_fixation_length_Disgusted"], [0.29817, 0.396516, 0.428531])



    def test_sum_fixation_length_Neutral(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        #act
        self.Data_Object.get_sum_fixation_length_Neutral()
        #assert
        self.assertEqual(self.Data_Object.output_data_dict["sum_fixation_length_Neutral"], [3089, 2398, 1555])

    def test_sum_fixation_length_Neutral_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        #act
        self.Data_Object.get_sum_fixation_length_Neutral()
        #assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_sum_fixation_length_Neutral"], [0.635205, 0.497304, 0.435818])

    def test_get_sum_fixation_length_White_Space(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_sum_fixation_length_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["sum_fixation_length_White_Space"], [324, 512, 484])

    def test_get_sum_fixation_length_White_Space_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_sum_fixation_length_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_sum_fixation_length_WS"], [0.066626, 0.10618, 0.13565])

    def test_get_sum_fixation_length_All(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_sum_fixation_length_All()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["sum_fixation_length_All"], [4863, 4822, 3568])

#Averages
    def test_get_average_fixation_length_Disgusted(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_Disgusted()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_fixation_length_Disgusted"], [362.5, 239, 254.8333])

    def test_get_average_fixation_length_Disgusted_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_Disgusted()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_mean_fixation_length_Disgusted"], [1.118137, 0.743467, 1.14275])

    def test_get_average_fixation_length_Neutral(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_Neutral()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_fixation_length_Neutral"], [308.9, 479.6, 194.357])

    def test_get_average_fixation_length_Neutral_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_Neutral()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_mean_fixation_length_Neutral"], [0.952807, 1.491912, 0.871637])

    def test_get_average_fixation_length_White_Space(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_fixation_length_White_Space"], [324,256,242])


    def test_get_average_fixation_length_White_Space_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_mean_fixation_length_WS"], [0.999383, 0.79635, 1.085202])

    def test_get_average_fixation_length_ALL(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_All()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_fixation_length_All"], [324.2, 321.4667, 223])

#Amounts
    def test_get_amount_fixation_Disgusted(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_Disgusted()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_fixation_Disgusted"],[4, 8, 6])

    def test_get_amount_fixation_Disgusted_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_Disgusted()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_amount_fixation_Disgusted"],[0.2666, 0.5333, 0.375])

    def test_get_amount_fixation_Neutral(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_Neutral()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_fixation_Neutral"], [10, 5, 8])

    def test_get_amount_fixation_Neutral_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_Neutral()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_amount_fixation_Neutral"], [0.666, 0.3333, 0.5])

    def test_get_amount_fixation_White_Space(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_fixation_White_Space"], [1, 2 ,2])

    def test_get_amount_fixation_White_Space_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_amount_fixation_WS"], [0.0666, 0.13333, 0.125])

    def test_get_amount_fixation_ALL(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_All()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_fixation_All"], [15,15,16])

#STDs

    def test_STD_fixation_length_Disgusted(self):
         self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
         # act
         self.Data_Object.get_STD_fixation_length_Disgusted()
         # assert
         self.assertEqual(self.Data_Object.output_data_dict["STD_fixation_length_Disgusted"], [138.285, 110.6662, 127.0701])

    def test_STD_fixation_length_Neutral(self):
         self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
         # act
         self.Data_Object.get_STD_fixation_length_Neutral()
         # assert
         self.assertEqual(self.Data_Object.output_data_dict["STD_fixation_length_Neutral"], [194.3515, 420.9977, 88.6086])

    def test_STD_fixation_length_White_Space(self):
         self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
         # act
         self.Data_Object.get_STD_fixation_length_White_Space()
         # assert
         self.assertEqual(self.Data_Object.output_data_dict["STD_fixation_length_White_Space"], [0,44,62])

    def test_STD_fixation_length_All(self):
         self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
         # act
         self.Data_Object.get_STD_fixation_length_All()
         # assert
         self.assertEqual(self.Data_Object.output_data_dict["STD_fixation_length_All"], [175.5799, 280.0062 , 106.2832])
#Ratios - fix vals
    def test_get_ratio_D_DN(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_ratio_D_DN()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["Ratio D/D+N"], [0.319454, 0.443619, 0.495785])

    def test_get_ratio_N_DN(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_ratio_N_DN()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["Ratio N/D+N"], [0.680546, 0.556381, 0.504215])

    def test_get_ratio_WS_All(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_ratio_WS_All()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["Ratio WS/WS+N+D"], [0.066626,0.10618, 0.13565])

    def test_get_ratio_D_DN_2(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_ratio_D_DN_2()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["%threat - #2"], [0.29817, 0.396516, 0.428531])

    def test_get_ratio_N_DN_2(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_ratio_N_DN_2()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["%neutral - #2"], [0.635205, 0.497304, 0.435818])

#pupil size avg

    def test_get_average_pupil_size_Disgusted(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_pupil_size_Disgusted()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_pupil_size_Disgusted"], [4.875, 5.1, 5.03333])

    def test_get_average_pupil_size_Disgusted_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_pupil_size_Disgusted()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_mean_pupil_size_Disgusted"], [1.008621, 1.021362, 1.024597])


    def test_get_average_pupil_size_Neutral(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_pupil_size_Neutral()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_pupil_size_Neutral"], [4.85, 4.88, 4.875])

    def test_get_average_pupil_size_Neutral_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_pupil_size_Neutral()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_mean_pupil_size_Neutral"], [1.003448, 0.977303, 0.992366])

    def test_get_average_pupil_size_White_Space(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_pupil_size_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_pupil_size_White_Space"], [4.5, 4.85, 4.7])

    def test_get_average_pupil_size_White_Space_norm(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_pupil_size_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_mean_pupil_size_WS"], [0.931034, 0.971295, 0.956743])

    def test_get_average_pupil_size_All(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_pupil_size_All()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_pupil_size_All"], [4.83333, 4.993333,4.9125])

#pupil size std

    def test_get_STD_pupil_size_Disgusted(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_STD_pupil_size_Disgusted()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["STD_pupil_size_Disgusted"], [0.082916,0.05,0.188562])

    def test_get_STD_pupil_size_Neutral(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_STD_pupil_size_Neutral()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["STD_pupil_size_Neutral"], [0.15, 0.146969, 0.096825])

    def test_get_STD_pupil_size_White_Space(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_STD_pupil_size_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["STD_pupil_size_White_Space"], [0, 0.15, 0.4])

    def test_get_STD_pupil_size_All(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_STD_pupil_size_All()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["STD_pupil_size_All"], [0.157762, 0.156915, 0.223257])

# medians

    def test_get_difference_between_medians(self):
        self.Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_difference_between_medians()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["STD_Disgusted_difference_between_medians"], [1.414, 0.400016, 6.7917])
        self.assertEqual(self.Data_Object.output_data_dict["STD_Neutral_difference_between_medians"], [2.36314, 0, 0.23478])
        self.assertEqual(self.Data_Object.output_data_dict["sum_disgusted_difference_between_medians"], [0.689977, 1.15, 3.132432])
        self.assertEqual(self.Data_Object.output_data_dict["norm_sum_disgusted_difference_between_medians"], [0.514, 1.7747, 2.7499])
        self.assertEqual(self.Data_Object.output_data_dict["sum_neutral_difference_between_medians"], [1.534, 0.180118, 0.197997])
        self.assertEqual(self.Data_Object.output_data_dict["norm_sum_neutral_difference_between_medians"], [1.14365, 0.276, 0.17382])
        self.assertEqual(self.Data_Object.output_data_dict["sum_all_difference_between_medians"], [1.13413, 0.6525, 1.139089])






if __name__ == '__main__':
    unittest.main()
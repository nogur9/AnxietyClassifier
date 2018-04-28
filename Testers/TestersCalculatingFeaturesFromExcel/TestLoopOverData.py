import unittest
from DataImporting.DataFromExcel import get_data
from CalculatingFeaturesFromExcel import ExtractFeatures
Test_Data_path = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\Testers\\test data_ordered.xlsx"
FIXATION_DATA_SHEET = 'Sheet1'
#DEMOGRAPHICS_SHEET = 'Final all Results'


class TestFeatureExctraction(unittest.TestCase):
    Data_Object = None

    def test_get_sum_fixation_length_Disgusted(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        #act
        self.Data_Object.get_sum_fixation_length_Disgusted()
        #assert
        self.assertEqual(self.Data_Object.output_data_dict["sum_fixation_length_Disgusted"], [12754])

    def test_get_sum_fixation_length_Disgusted_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        #act
        self.Data_Object.get_sum_fixation_length_Disgusted()
        #assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_sum_fixation_length_Disgusted"], [0.343810653])



    def test_sum_fixation_length_Neutral(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        #act
        self.Data_Object.get_sum_fixation_length_Neutral()
        #assert
        self.assertEqual(self.Data_Object.output_data_dict["sum_fixation_length_Neutral"], [22016])

    def test_sum_fixation_length_Neutral_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        #act
        self.Data_Object.get_sum_fixation_length_Neutral()
        #assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_sum_fixation_length_Neutral"], [0.593487168])

    def test_get_sum_fixation_length_White_Space(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_sum_fixation_length_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["sum_fixation_length_White_Space"], [2326])

    def test_get_sum_fixation_length_White_Space_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_sum_fixation_length_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_sum_fixation_length_WS"], [0.062702178])

    def test_get_sum_fixation_length_All(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_sum_fixation_length_All()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["sum_fixation_length_All"], [37096])

#Averages
    def test_get_average_fixation_length_Disgusted(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_Disgusted()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_fixation_length_Disgusted"], [219.8966])

    def test_get_average_fixation_length_Disgusted_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_Disgusted()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_mean_fixation_length_Disgusted"], [0.966226492])


    def test_get_average_fixation_length_Neutral(self):

        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_Neutral()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_fixation_length_Neutral"], [241.9341])

    def test_get_average_fixation_length_Neutral_norm(self):

        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_Neutral()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_mean_fixation_length_Neutral"], [1.0630594335576005])



    def test_get_average_fixation_length_White_Space(self):

        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_fixation_length_White_Space"], [166.1429])


    def test_get_average_fixation_length_White_Space_norm(self):

        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_mean_fixation_length_WS"], [0.730032503])



    def test_get_average_fixation_length_ALL(self):

        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_average_fixation_length_All()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["average_fixation_length_All"], [227.5828])

#Amounts
    def test_get_amount_fixation_Disgusted(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_Disgusted()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_fixation_Disgusted"],[58])


    def test_get_amount_fixation_Disgusted_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_Disgusted()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_amount_fixation_Disgusted"],[0.355828221])


    def test_get_amount_fixation_Neutral(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_Neutral()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_fixation_Neutral"], [91])

    def test_get_amount_fixation_Neutral_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_Neutral()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_amount_fixation_Neutral"], [0.558282209])


    def test_get_amount_fixation_White_Space(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_fixation_White_Space"], [14])

    def test_get_amount_fixation_White_Space_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_White_Space()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["normalized_amount_fixation_WS"], [0.085889571])

    def test_get_amount_fixation_ALL(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_fixation_All()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_fixation_All"], [163])

#STDs

    def test_STD_fixation_length_Disgusted(self):
         self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
         # act
         self.Data_Object.get_STD_fixation_length_Disgusted()
         # assert
         self.assertEqual(self.Data_Object.output_data_dict["STD_fixation_length_Disgusted"], [117.107921])

    def test_STD_fixation_length_Neutral(self):
         self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
         # act
         self.Data_Object.get_STD_fixation_length_Neutral()
         # assert
         self.assertEqual(self.Data_Object.output_data_dict["STD_fixation_length_Neutral"], [194.8747584])


    def test_STD_fixation_length_White_Space(self):
         self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
         # act
         self.Data_Object.get_STD_fixation_length_White_Space()
         # assert
         self.assertEqual(self.Data_Object.output_data_dict["STD_fixation_length_White_Space"], [39.25349328])

    def test_STD_fixation_length_All(self):
         self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
         # act
         self.Data_Object.get_STD_fixation_length_All()
         # assert
         self.assertEqual(self.Data_Object.output_data_dict["STD_fixation_length_All"], [163.3214984])
#Ratios
    def test_get_ratio_D_DN(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_ratio_D_DN()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["Ratio D/D+N"], [0.476141129])

    def test_get_ratio_N_DN(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_ratio_N_DN()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["Ratio N/D+N"], [0.523858871])

    def test_get_ratio_WS_All(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_ratio_WS_All()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["Ratio WS/WS+N+D"], [0.264569865])

    def get_ratio_D_DN_2(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_ratio_D_DN_2()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["%threat - #2"], [0.350168535])

    def get_ratio_N_DN_2(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_ratio_N_DN_2()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["%neutral - #2"], [0.3852616])

    def test_var(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.var_threat_precentage_between_trials()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["var_ratio_D_DN"], [0.004641433])

#Second Excel

    def test_get_amount_DN_transitions(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_DN_transitions()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_DN_transitions"], [27])

    def test_get_amount_DN_transitions_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_DN_transitions()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["norm_amount_DN_transitions"], [0.16667])

    def test_get_amount_ND_transitions(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_ND_transitions()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_ND_transitions"], [23])

    def test_get_amount_ND_transitions_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_ND_transitions()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["norm_amount_ND_transitions"], [0.141975])

    def test_get_amount_DD_transitions(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_DD_transitions()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_DD_transitions"], [26])

    def test_get_amount_DD_transitions_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_DD_transitions()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["norm_amount_DD_transitions"], [0.160494])

    def test_get_amount_NN_transitions(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_NN_transitions()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_NN_transitions"], [56])

    def test_get_amount_NN_transitions_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_NN_transitions()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["norm_amount_NN_transitions"], [0.345679])

    def test_get_amount_diff_AOI_transitions(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_diff_AOI_transitions()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_diff_AOI_transitions"], [69])

    def test_get_amount_diff_AOI_transitions_norm(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.get_amount_diff_AOI_transitions()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["norm_amount_diff_AOI_transitions"], [0.425926])


    def test_amount_of_first_fixations_D(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.amount_of_first_fixations()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_of_first_fixations_on_threat"], [2])

    def test_amount_of_first_fixations_N(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.amount_of_first_fixations()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_of_first_fixations_on_neutral"], [3])

    def test_amount_of_first_fixations_WS(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.amount_of_first_fixations()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_of_first_fixations_on_WS"], [5])

    def test_amount_of_second_fixations_D(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.amount_of_second_fixations()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_of_second_fixations_on_threat"], [6])

    def test_amount_of_second_fixations_N(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.amount_of_second_fixations()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_of_second_fixations_on_neutral"], [3])

    def test_amount_of_second_fixations_WS(self):
        self.Data_Object = ExtractFeatures.Data(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        self.Data_Object.amount_of_second_fixations()
        # assert
        self.assertEqual(self.Data_Object.output_data_dict["amount_of_second_fixations_on_WS"], [1])

if __name__ == '__main__':
    unittest.main()
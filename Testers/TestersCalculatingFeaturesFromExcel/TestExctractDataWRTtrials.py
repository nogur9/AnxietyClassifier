import unittest
from CalculatingFeaturesFromExcel import ExctractFeaturesWRTtrials

Test_Data_path = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\Testers\\TrialsTester.xlsx"
FIXATION_DATA_SHEET = 'Sheet1'
#DEMOGRAPHICS_SHEET = 'Final all Results'


class TestFeatureExctraction(unittest.TestCase):
    Data_Object = None

    def test_get_sum_fixation_length(self):
        self.Data_Object = ExctractFeaturesWRTtrials.TrialsData(Test_Data_path, FIXATION_DATA_SHEET)
        #act
        ans = self.Data_Object.get_sum_fixation_length()
        print(ans)
        print([[[2785,2168],[1912,1450]],[[1754,2439],[2398,3089]],[[252,573],[512,324]],[[4791,5180],[4822,4863]],[[0.581298,0.418533],[0.3965,0.29817]],[[0.3661,0.4708],[0.4973,0.6352]],[[0.052599,0.1106],[0.106,0.0666]]])
        #return [sum_Disgusted, sum_Neutral, sum_WS, sum_All, norm_sum_Disgusted, norm_sum_Neutral, norm_sum_WS]
        #assert
        self.assertEqual(ans, [[[2785,2168],[1912,1450]],[[1754,2439],[2398,3089]],[[252,573],[512,324]],[[4791,5180],[4822,4863]],[[0.581298,0.418533],[0.3965,0.29817]],[[0.3661,0.4708],[0.4973,0.6352]],[[0.052599,0.1106],[0.106,0.0666]]])

#Averages
    def test_get_average_fixation_length(self):
        self.Data_Object = ExctractFeaturesWRTtrials.TrialsData(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        ans = self.Data_Object.get_average_fixation_length_each_trial()
        print(ans)
        print([[[309.4444,361.333],[239,362.5]],[[194.88,304.875],[479.6,308.9]],[[252,286.5],[256,324]],[[252.1579,323.75],[321.4667,324.2]],[[1.227,1.116],[0.7434,1.118]],[[0.7728,0.941699],[1.4919,0.9528]],[[0.999,0.884],[0.79635,0.999383]]])
        # assert
        #return [mean_Disgusted, mean_Neutral, mean_WS,mean_All,norm_mean_Disgusted,norm_mean_Neutral,norm_mean_WS]
        self.assertEqual(ans, [[[309.4444,361.333],[239,324]],[[194.88,304.875],[479.6,308.9]],[[252,286.5],[256,324]],[[252.1579,323.75],[321.4667,324.2]],[[1.227,1.116],[0.7434,1.118]],[[0.7728,0.941699],[1.4919,0.9528]],[[0.999,0.884],[0.79635,0.999383]]])

#Amounts
    def test_get_amount_fixation(self):
        self.Data_Object = ExctractFeaturesWRTtrials.TrialsData(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        ans = self.Data_Object.get_amount_fixations()
        print(ans)
        print([[[9,6],[8,4]],[[9,8],[5,10]],[[1,2],[2,1]],[[19,16],[15,15]],[[0.47368,0.375],[0.5333,0.2666]],[[0.4736,0.5],[0.3333,0.67777]],[[0.0526,0.125],[0.1333,0.06667]]])
        # assert
        #return [amount_Disgusted, amount_Neutral, amount_WS,amount_All,norm_amount_Disgusted,norm_amount_Neutral,norm_amount_WS]
        self.assertEqual(ans,[[[9,6],[8,4]],[[9,8],[5,10]],[[1,2],[2,1]],[[19,16],[15,15]],[[0.47368,0.375],[0.5333,0.2666]],[[0.4736,0.5],[0.3333,0.67777]],[[0.0526,0.125],[0.1333,0.06667]]])

#STDs
    def test_STD_fixation_length(self):
         self.Data_Object = ExctractFeaturesWRTtrials.TrialsData(Test_Data_path, FIXATION_DATA_SHEET)
         # act
         #return [STD_Disgusted, STD_Neutral, STD_WS,STD_All]
         ans =self.Data_Object.get_STD_fixation_length()
         # assert
         print(ans)
         print([[[94.45176, 165.854],[110.6662, 138.285]],[[64.2693,153.0044],[420.9977,194.3515]],[[0,111.5],[44,0]],[[96.38697,156.3835],[280.0062,175.5799]]])
         self.assertEqual(ans, [[[94.45176, 165.854],[110.6662, 138.285]],[[64.2693,153.0044],[420.9977,194.3515]],[[0,111.5],[44,0]],[[96.38697,156.3835],[280.0062,175.5799]]])

#Ratios
    def test_get_ratios(self):
        self.Data_Object = ExctractFeaturesWRTtrials.TrialsData(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        ans = self.Data_Object.get_Ratios()
        # assert
        print(ans)
        print([[[0.6135,0.47],[0.4436,0.3194]],[[0.3864,0.5294],[0.556,0.6805]],[[0.0525,0.1106],[0.106,0.0666]],[[0.581298,0.418533],[0.39516,0.29817]],[[0.366103,0.470849],[0.497304,0.635205]]])
        self.assertEqual(ans, [[[0.6135,0.47],[0.4436,0.3194]],[[0.3864,0.5294],[0.556,0.6805]],[[0.0525,0.1106],[0.106,0.0666]],[[0.581298,0.418533],[0.39516,0.29817]],[[0.366103,0.470849],[0.497304,0.635205]]])

#DIFF AOIs
    def test_get_mean_different_AOI_per_trial(self):
        self.Data_Object = ExctractFeaturesWRTtrials.TrialsData(Test_Data_path, FIXATION_DATA_SHEET)
        # act
        ans = self.Data_Object.get_mean_different_AOI_per_trial()
        # assert
        self.assertEqual(ans, [[12,12],[12,12]])


if __name__ == '__main__':
    unittest.main()
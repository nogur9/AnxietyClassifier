import numpy as np
import matplotlib.pyplot as plt
from CalculatingFeaturesFromExcel import ExctractFeaturesWRTtrials
from CalculatingFeaturesFromExcel.RegressionFunctions import sine, linear
def plot_timeline_between_trial_two_vars(TrialData):
    result = TrialData.get_average_fixation_length_each_trial()

    Disgusted_trials_mean = [np.mean([mean_Disgusted[i][j] for i in range(len(mean_Disgusted))]) for j in
                             range(mean_Disgusted[0])]
    Neutral_trials_mean = [np.mean([mean_Neutral[i][j] for i in range(len(mean_Neutral))]) for j in
                           range(mean_Disgusted[0])]
    N = len(mean_Disgusted[0])
    fig, ax = plt.subplots()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars
    p1 = ax.bar(ind, Disgusted_trials_mean, width, color='r')
    p2 = ax.bar(ind + width, Neutral_trials_mean, width, color='y')
    ax.set_xticklabels(range(len(Disgusted_trials_mean)))
    ax.set_title('Scores by trials and mean fixation AOI')
    ax.set_xticks(ind + width / 2)
    ax.legend((p1[0], p2[0]), ('D', 'N'))
    ax.autoscale_view()

    plt.show()



def plot_timeline_between_trial_single_var(results, titles):
    for result,title in zip(results,titles):
        result = result[len(result)==60]
#        trials_mean = [np.mean([result[i][j] for i in range(len(result))]) for j in
 #                                range(len(result[0]))]
        N = len(result)
        fig, ax = plt.subplots()
        ind = np.arange(N)  # the x locations for the groups
        width = 1.5  # the width of the bars
        ax.bar(ind, result, width, color='b')
        ax.set_xticklabels([0,'','','','','','','','','',10,'','','','','','','','','',20,'','','','','','','','','',30,'','','','','','','','','',40,'','','','','','','','','',50])
        ax.set_title(title)
        ax.set_xticks(ind + width / 2)
        ax.autoscale_view()
        plt.savefig(title)
        plt.close()


def runner():
    Test_Data_path_high = "C:\\Users\\user\PycharmProjects\\AnxietyClassifier\\AmitsData\\SAD.xlsx"
    #Test_Data_path_high = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\Testers\\301.xlsx"
   # Test_Data_path_low = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\Testers\\337.xlsx"
    #FIXATION_DATA_SHEET = 'Sheet1'
    FIXATION_DATA_SHEET = 'fixation data'
    high_data_object = ExctractFeaturesWRTtrials.TrialsData(Test_Data_path_high,FIXATION_DATA_SHEET)
    #low_data_object = ExctractFeaturesWRTtrials.TrialsData(Test_Data_path_low,FIXATION_DATA_SHEET)
    high_results_ratio = high_data_object.get_Ratios()
    linear(high_results_ratio[1], plot=1)
    high_results_avg = high_data_object.get_average_fixation_length_each_trial()
    high_results_amounts = high_data_object.get_amount_fixation_length()
    high_results_aois = high_data_object.get_mean_different_AOI_per_trial()
    high_results_sums = high_data_object.get_sum_fixation_length()
    high_results_std = high_data_object.get_STD_fixation_length()

    plot_timeline_between_trial_single_var(high_results_avg,['mean_Disgusted high', 'mean_Neutral high', 'mean_WS high','mean_All high','norm_mean_Disgusted high','norm_mean_Neutral high','norm_mean_WS high'])
    plot_timeline_between_trial_single_var(high_results_sums,['sum_Disgusted high', 'sum_Neutral high','sum_WS','sum_All high','norm_sum_Disgusted high','norm_sum_Neutral high','norm_sum_WS high'])
    plot_timeline_between_trial_single_var(high_results_std, ['STD_Disgusted high', 'STD_Neutral high', 'STD_WS high','STD_All high'])
    plot_timeline_between_trial_single_var(high_results_ratio, ['ratio_D_DN high', 'ratio_N_DN high', 'ratio_WS_all high', 'ratio_D_DN2 high', 'ratio_N_DN2 high'])
    plot_timeline_between_trial_single_var(high_results_aois, ['mean_AOIs high'])
    plot_timeline_between_trial_single_var(high_results_amounts, ['amount_Disgusted high', 'amount_Neutral high', 'amount_WS high','amount_All high','norm_amount_Disgusted high','norm_amount_Neutral high','norm_amount_WS high'])
    #
    #
    #
    # low_results_avg = low_data_object.get_average_fixation_length_each_trial()
    # low_results_sums = low_data_object.get_sum_fixation_length()
    # low_results_std = low_data_object.get_STD_fixation_length()
    # low_results_ratio = low_data_object.get_Ratios()
    # low_results_amounts = low_data_object.get_amount_fixation_length()
    # low_results_aois = low_data_object.get_mean_different_AOI_per_trial()
    # plot_timeline_between_trial_single_var(low_results_avg,['mean_Disgusted low', 'mean_Neutral low', 'mean_WS low','mean_All low','norm_mean_Disgusted low','norm_mean_Neutral low','norm_mean_WS low'])
    # plot_timeline_between_trial_single_var(low_results_sums,['sum_Disgusted low', 'sum_Neutral low','sum_WS low','sum_All low','norm_sum_Disgusted low','norm_sum_Neutral low','norm_sum_WS low'])
    # plot_timeline_between_trial_single_var(low_results_std, ['STD_Disgusted low', 'STD_Neutral low', 'STD_WS low','STD_All low'])
    # plot_timeline_between_trial_single_var(low_results_ratio, ['ratio_D_DN low', 'ratio_N_DN low', 'ratio_WS_all low', 'ratio_D_DN2 low', 'ratio_N_DN2 low'])
    # plot_timeline_between_trial_single_var(low_results_aois, ['mean_AOIs low'])
    # plot_timeline_between_trial_single_var(low_results_amounts, ['amount_Disgusted low', 'amount_Neutral low', 'amount_WS low','amount_All low','norm_amount_Disgusted low','norm_amount_Neutral low','norm_amount_WS low'])

runner()
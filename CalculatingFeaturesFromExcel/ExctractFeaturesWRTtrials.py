import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from DataImporting import DataFromExcel

Fixation_length_cutoff = 100


class TrialsData:
    grouping_function = None
    fixation_dataset = None
    output_data_frame = None
    output_data_dict = {}
    demographic_dataset = None

    def __init__(self, path, fixation_dataset_sheet_name, demographic_dataset_sheet_name=None,
                 grouping_function=np.nansum):

        self.fixation_dataset = DataFromExcel.get_data(path, fixation_dataset_sheet_name)
        self.fixation_dataset = self.fixation_dataset[self.fixation_dataset.Fixation_Duration > Fixation_length_cutoff]
        self.grouping_function = grouping_function
        self.Trials_count = np.array([len(set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == j])) for j in
                                      sorted(set(self.fixation_dataset.Subject))])



    def get_Ratios(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]
        All_fixations = [
            [self.fixation_dataset[
                 (self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
             for j in trials[i]] for i in range(len(subjects))]

        Mean_Neutral = [[np.nanmean(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "N")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        Mean_Disgusted = [[np.nanstd(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "D")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        Mean_WS = [[np.nanstd(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "White Space")]) for
             j in range(len(trials[i]))] for i in range(len(subjects))]

        Mean_Neutral = [[0 if math.isnan(x) else x for x in i] for i in Mean_Neutral]
        Mean_Disgusted = [[0 if math.isnan(x) else x for x in i] for i in Mean_Disgusted]
        Mean_WS = [[0 if math.isnan(x) else x for x in i for i] in Mean_WS]

        ratio_N_DN2 = [Mean_Neutral[i] / float(Mean_WS[i] + Mean_Neutral[i] + Mean_Disgusted[i]) for i in
                 range(len(Mean_Disgusted))]


        ratio_D_DN2 = [[Mean_Disgusted[i][j] / float(Mean_WS[i][j] + Mean_Neutral[i][j] + Mean_Disgusted[i][j]) for j in
            range(Mean_Disgusted[0])]for i in range(len(Mean_Disgusted))]

        ratio_WS_all = [[Mean_WS[i][j] / float(Mean_WS[i][j] + Mean_Neutral[i][j] + Mean_Disgusted[i][j]) for j in range(len(Mean_Disgusted[0]))]
                 for i in range(len(Mean_Disgusted))]

        ratio_N_DN = [[Mean_Neutral[i][j] / float(Mean_Neutral[i][j] + Mean_Disgusted[i][j]) for j in range(len(Mean_Disgusted[0]))]for i in range(len(Mean_Disgusted))]

        ratio_D_DN = [[Mean_Disgusted[i][j] / float(Mean_Neutral[i][j] + Mean_Disgusted[i][j]) for j in range(len(Mean_Disgusted[0]))] for i in range(len(Mean_Disgusted))]

        return [ratio_D_DN, ratio_N_DN, ratio_WS_all, ratio_D_DN2, ratio_N_DN2]


    def get_mean_different_AOI_per_trial(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]
        All_fixations = [
            [self.fixation_dataset[
                 (self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
             for j in trials[i]] for i in range(len(subjects))]

        mean_AOIs = [[len(set(self.All_fixations[i][j].Area_of_Interest[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Trial==j)])) for j in trials[i]] for i in range(len(subjects))]
        return [mean_AOIs]

    def get_STD_fixation_length(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]
        All_fixations = [
            [self.fixation_dataset[
                 (self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
             for j in trials[i]] for i in range(len(subjects))]

        STD_Neutral = [
            [np.nanstd(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "N")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        STD_Disgusted = [
            [np.nanstd(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "D")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        STD_WS = [[np.nanstd(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "White Space")]) for
             j in range(len(trials[i]))] for i in range(len(subjects))]

        STD_Neutral = [[0 if math.isnan(x) else x for x in i]for i in STD_Neutral]
        STD_Disgusted = [[0 if math.isnan(x) else x for x in i] for i in STD_Disgusted]
        STD_WS = [[0 if math.isnan(x) else x for x in i] for i in STD_WS]

        return [STD_Disgusted, STD_Neutral, STD_WS]


    def get_amount_fixation_length(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]
        All_fixations = [
            [self.fixation_dataset[
                 (self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
             for j in trials[i]] for i in range(len(subjects))]

        amount_Neutral = [
            [len(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "N")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        amount_Disgusted = [
            [len(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "D")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        amount_WS = [
            [len(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "White Space")]) for
             j in range(len(trials[i]))] for i in range(len(subjects))]

        return [amount_Disgusted, amount_Neutral, amount_WS]

    def get_sum_fixation_length(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]
        All_fixations = [
            [self.fixation_dataset[
                 (self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
             for j in trials[i]] for i in range(len(subjects))]

        sum_Neutral = [
            [np.nansum(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "N")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        sum_Disgusted = [
            [np.nansum(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "D")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        sum_WS = [
            [np.nansum(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "White Space")]) for
             j in range(len(trials[i]))] for i in range(len(subjects))]

        return [sum_Disgusted, sum_Neutral, sum_WS]

    def get_average_fixation_length_each_trial(self):
        # param order of trials

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]
        All_fixations = [[self.fixation_dataset[(self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
             for j in trials[i]] for i in range(len(subjects))]

        mean_Neutral = [
            [np.nanmean(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "N")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        mean_Disgusted = [
            [np.nanmean(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "D")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        mean_WS = [[np.mean(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "White Space")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        return [mean_Disgusted, mean_Neutral, mean_WS]

    def plot_timeline_between_trial(self):

        [mean_Disgusted, mean_Neutral] = self.get_average_fixation_length_each_trial()
        Disgusted_trials_mean = [np.mean([mean_Disgusted[i][j] for i in range(len(mean_Disgusted))]) for j in
                                 range(len(mean_Disgusted[0]))]
        Neutral_trials_mean = [np.mean([mean_Neutral[i][j] for i in range(len(mean_Neutral))]) for j in
                               range(len(mean_Neutral[0]))]
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

# Test_Data_path = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\Testers\\test data_ordered.xlsx"
# Test_Data_path = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\Testers\\301.xlsx"
# FIXATION_DATA_SHEET = 'Sheet1'
# Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
# Data_Object.get_amount_diff_AOI_transitions()
# Test_Data_path = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\Testers\\337.xlsx"
# FIXATION_DATA_SHEET = 'Sheet1'
# Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
# Data_Object.plot_timeline_between_trial()
# Data_Object.var_threat_precentage_between_trials()
# Data_Object.plot_timeline()
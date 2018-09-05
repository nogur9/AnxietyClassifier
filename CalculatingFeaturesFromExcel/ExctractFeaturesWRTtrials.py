import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from DataImporting import ImportData
from CalculatingFeaturesFromExcel.RegressionFunctions import sine, linear
Fixation_length_cutoff = 100


class TrialsData:
    grouping_function = None
    fixation_dataset = None
    output_data_frame = None
    output_data_dict = {}
    demographic_dataset = None

    def __init__(self, path, fixation_dataset_sheet_name, demographic_dataset_sheet_name=None,
                 grouping_function=np.nansum):

        self.fixation_dataset = ImportData.get_data(path, fixation_dataset_sheet_name)
        self.fixation_dataset = self.fixation_dataset[self.fixation_dataset.Fixation_Duration > Fixation_length_cutoff]
        self.grouping_function = grouping_function
        self.Trials_count = np.array([len(set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == j])) for j in
                                      sorted(set(self.fixation_dataset.Subject))])



    def get_Ratios(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = sorted([set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects])
        All_fixations = [
            [self.fixation_dataset[
                 (self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
             for j in trials[i]] for i in range(len(subjects))]

        Mean_Neutral = [[np.sum(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "N")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        Mean_Disgusted = [[np.sum(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "D")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        Mean_WS = [[np.sum(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "White Space")]) for
             j in range(len(trials[i]))] for i in range(len(subjects))]

        Mean_Neutral = [[0 if math.isnan(x) else x for x in i] for i in Mean_Neutral]
        Mean_Disgusted = [[0 if math.isnan(x) else x for x in i] for i in Mean_Disgusted]
        Mean_WS = [[0 if math.isnan(x) else x for x in i] for i in Mean_WS]



        ratio_N_DN2 = [[Mean_Neutral[i][j] / float(Mean_WS[i][j] + Mean_Neutral[i][j] + Mean_Disgusted[i][j]) for j in
            range(len(Mean_Disgusted[i]))] for i in range(len(Mean_Disgusted))]

        ratio_D_DN2 = [[Mean_Disgusted[i][j] / float(Mean_WS[i][j] + Mean_Neutral[i][j] + Mean_Disgusted[i][j]) for j in
            range(len(Mean_Disgusted[i]))]for i in range(len(Mean_Disgusted))]

        ratio_WS_all = [[Mean_WS[i][j] / float(Mean_WS[i][j] + Mean_Neutral[i][j] + Mean_Disgusted[i][j]) for j in range(len(Mean_Disgusted[i]))]
                 for i in range(len(Mean_Disgusted))]

        ratio_N_DN = [[Mean_Neutral[i][j] / float(Mean_Neutral[i][j] + Mean_Disgusted[i][j]) for j in range(len(Mean_Disgusted[i]))]for i in range(len(Mean_Disgusted))]

        ratio_D_DN = [[Mean_Disgusted[i][j] / float(Mean_Neutral[i][j] + Mean_Disgusted[i][j]) for j in range(len(Mean_Disgusted[i]))] for i in range(len(Mean_Disgusted))]

        #est_std, est_phase, est_mean
        reggression_ratio_N_DN2 = sine(ratio_N_DN2)
        self.output_data_dict["trials_ratio_N_DN2_est_std"] = reggression_ratio_N_DN2[0]
        self.output_data_dict["trials_ratio_N_DN2_est_phase"] = reggression_ratio_N_DN2[1]
        self.output_data_dict["trials_ratio_N_DN2_est_mean"] = reggression_ratio_N_DN2[2]

        reggression_ratio_D_DN2 = sine(ratio_D_DN2)
        self.output_data_dict["trials_ratio_D_DN2_est_std"] = reggression_ratio_D_DN2[0]
        self.output_data_dict["trials_ratio_D_DN2_est_phase"] = reggression_ratio_D_DN2[1]
        self.output_data_dict["trials_ratio_D_DN2_est_mean"] = reggression_ratio_D_DN2[2]

        reggression_ratio_WS_all = sine(ratio_WS_all)
        self.output_data_dict["trials_ratio_WS_all_est_std"] = reggression_ratio_WS_all[0]
        self.output_data_dict["trials_ratio_WS_all_est_phase"] = reggression_ratio_WS_all[1]
        self.output_data_dict["trials_ratio_WS_all_est_mean"] = reggression_ratio_WS_all[2]

        reggression_ratio_D_DN = sine(ratio_D_DN)
        self.output_data_dict["trials_ratio_D_DN_est_std"] = reggression_ratio_D_DN[0]
        self.output_data_dict["trials_ratio_D_DN_est_phase"] = reggression_ratio_D_DN[1]
        self.output_data_dict["trials_ratio_D_DN_est_mean"] = reggression_ratio_D_DN[2]

        reggression_ratio_N_DN = sine(ratio_N_DN)
        self.output_data_dict["trials_ratio_N_DN_est_std"] = reggression_ratio_N_DN[0]
        self.output_data_dict["trials_ratio_N_DN_est_phase"] = reggression_ratio_N_DN[1]
        self.output_data_dict["trials_ratio_N_DN_est_mean"] = reggression_ratio_N_DN[2]

        return [ratio_D_DN, ratio_N_DN, ratio_WS_all, ratio_D_DN2, ratio_N_DN2]


    def get_mean_different_AOI_per_trial(self):
        #add linear
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = sorted([set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects])
        All_fixations = [
            [self.fixation_dataset[
                 (self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
             for j in trials[i]] for i in range(len(subjects))]

        mean_AOIs = [[len(set(All_fixations[i][k].Area_of_Interest[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Trial == j)])) for j,k in zip(trials[i],range(len(trials[i])))] for i in range(len(subjects))]

        #est_std, est_phase, est_mean
        reggression_mean_AOIs = sine(mean_AOIs)
        self.output_data_dict["trials_mean_AOIs_est_std"] = reggression_mean_AOIs[0]
        self.output_data_dict["trials_mean_AOIs_est_phase"] = reggression_mean_AOIs[1]
        self.output_data_dict["trials_mean_AOIs_est_mean"] = reggression_mean_AOIs[2]


        #linear
        linear_mean_AOIs = linear(mean_AOIs)
        self.output_data_dict["trials_mean_AOIs_1coeff"] = linear_mean_AOIs[0]
        self.output_data_dict["trials_mean_AOIs_2coeff"] = linear_mean_AOIs[1]


        return [mean_AOIs]

    def get_STD_fixation_length(self):
        #add linear
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = sorted([set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects])
        print(trials)
        All_fixations = [
            [self.fixation_dataset[
                 (self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
             for j in trials[i]] for i in range(len(subjects))]
        STD_All = [
            [np.nanstd(All_fixations[i][j].Fixation_Duration) for j in
             range(len(trials[i]))] for i in range(len(subjects))]


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

        return [STD_Disgusted, STD_Neutral, STD_WS,STD_All]


    def get_amount_fixation_length(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = sorted([set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects])
        All_fixations = [
            [self.fixation_dataset[
                 (self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
             for j in trials[i]] for i in range(len(subjects))]
        amount_All = [[len(All_fixations[i][j].Fixation_Duration) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        amount_Neutral = [
            [len(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "N")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        amount_Disgusted = [
            [len(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "D")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        amount_WS = [
            [len(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "White Space")]) for
             j in range(len(trials[i]))] for i in range(len(subjects))]
        norm_amount_Neutral =[[amount_Neutral[i][j]/float(amount_All[i][j])for j in
             range(len(trials[i]))] for i in range(len(subjects))]
        norm_amount_Disgusted=[[amount_Disgusted[i][j]/float(amount_All[i][j])for j in
             range(len(trials[i]))] for i in range(len(subjects))]
        norm_amount_WS =[[amount_WS[i][j]/float(amount_All[i][j])for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        reggression_amount_All = sine(amount_All)
        self.output_data_dict["trials_amount_All_est_std"] = reggression_amount_All[0]
        self.output_data_dict["trials_amount_All_est_phase"] = reggression_amount_All[1]
        self.output_data_dict["trials_amount_All_est_mean"] = reggression_amount_All[2]

        # linear
        linear_amount_All = linear(amount_All)
        self.output_data_dict["trials_amount_All_1coeff"] = linear_amount_All[0]
        self.output_data_dict["trials_amount_All_2coeff"] = linear_amount_All[1]

        reggression_amount_Disgusted = sine(amount_Disgusted)
        self.output_data_dict["trials_amount_Disgusted_est_std"] = reggression_amount_Disgusted[0]
        self.output_data_dict["trials_amount_Disgusted_est_phase"] = reggression_amount_Disgusted[1]
        self.output_data_dict["trials_amount_Disgusted_est_mean"] = reggression_amount_Disgusted[2]

        # linear
        linear_amount_Disgusted = linear(amount_Disgusted)
        self.output_data_dict["trials_amount_Disgusted_1coeff"] = linear_amount_Disgusted[0]
        self.output_data_dict["trials_amount_Disgusted_2coeff"] = linear_amount_Disgusted[1]

        reggression_amount_Neutral = sine(amount_Neutral)
        self.output_data_dict["trials_amount_Neutral_est_std"] = reggression_amount_Neutral[0]
        self.output_data_dict["trials_amount_Neutral_est_phase"] = reggression_amount_Neutral[1]
        self.output_data_dict["trials_amount_Neutral_est_mean"] = reggression_amount_Neutral[2]

        # linear
        linear_amount_Neutral = linear(amount_Neutral)
        self.output_data_dict["trials_amount_Neutral_1coeff"] = linear_amount_Neutral[0]
        self.output_data_dict["trials_amount_Neutral_2coeff"] = linear_amount_Neutral[1]

        reggression_amount_WS = sine(amount_WS)
        self.output_data_dict["trials_amount_WS_est_std"] = reggression_amount_WS[0]
        self.output_data_dict["trials_amount_WS_est_phase"] = reggression_amount_WS[1]
        self.output_data_dict["trials_amount_WS_est_mean"] = reggression_amount_WS[2]

        # linear
        linear_amount_WS = linear(amount_WS)
        self.output_data_dict["trials_amount_WS_1coeff"] = linear_amount_WS[0]
        self.output_data_dict["trials_amount_WS_2coeff"] = linear_amount_WS[1]

        reggression_norm_amount_Disgusted = sine(norm_amount_Disgusted)
        self.output_data_dict["trials_norm_amount_Disgusted_est_std"] = reggression_norm_amount_Disgusted[0]
        self.output_data_dict["trials_norm_amount_Disgusted_est_phase"] = reggression_norm_amount_Disgusted[1]
        self.output_data_dict["trials_norm_amount_Disgusted_mean"] = reggression_norm_amount_Disgusted[2]

        # linear
        linear_norm_amount_Disgusted = linear(norm_amount_Disgusted)
        self.output_data_dict["trials_norm_amount_Disgusted_1coeff"] = linear_norm_amount_Disgusted[0]
        self.output_data_dict["trials_norm_amount_Disgusted_2coeff"] = linear_norm_amount_Disgusted[1]

        reggression_norm_amount_Neutral = sine(norm_amount_Neutral)
        self.output_data_dict["trials_norm_amount_Neutral_est_std"] = reggression_norm_amount_Neutral[0]
        self.output_data_dict["trials_norm_amount_Neutral_est_phase"] = reggression_norm_amount_Neutral[1]
        self.output_data_dict["trials_norm_amount_Neutral_est_mean"] = reggression_norm_amount_Neutral[2]

        # linear
        linear_norm_amount_Neutral = linear(norm_amount_Neutral)
        self.output_data_dict["trials_norm_amount_Neutral_1coeff"] = linear_norm_amount_Neutral[0]
        self.output_data_dict["trials_norm_amount_Neutral_2coeff"] = linear_norm_amount_Neutral[1]

        reggression_norm_amount_WS = sine(norm_amount_WS)
        self.output_data_dict["trials_norm_amount_WS_est_std"] = reggression_norm_amount_WS[0]
        self.output_data_dict["trials_norm_amount_WS_est_phase"] = reggression_norm_amount_WS[1]
        self.output_data_dict["trials_norm_amount_WS_est_mean"] = reggression_norm_amount_WS[2]

        # linear
        linear_norm_amount_WS = linear(norm_amount_WS)
        self.output_data_dict["trials_norm_amount_WS_1coeff"] = linear_norm_amount_WS[0]
        self.output_data_dict["trials_norm_amount_WS_2coeff"] = linear_norm_amount_WS[1]

        return [amount_Disgusted, amount_Neutral, amount_WS,amount_All,norm_amount_Disgusted,norm_amount_Neutral,norm_amount_WS]

    def get_sum_fixation_length(self):
# maybe add linear
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = sorted([set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects])
        All_fixations = [[self.fixation_dataset[(self.fixation_dataset.Subject == subjects[i]) &
                                                (self.fixation_dataset.Trial == j)] for j in trials[i]]
                                                for i in range(len(subjects))]
        sum_All = [[np.nansum(All_fixations[i][j].Fixation_Duration) for j in
                                                range(len(trials[i]))] for i in range(len(subjects))]

        sum_Neutral = [
            [np.nansum(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "N")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        sum_Disgusted = [
            [np.nansum(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "D")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        sum_WS = [
            [np.nansum(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "White Space")]) for
             j in range(len(trials[i]))] for i in range(len(subjects))]
        norm_sum_Neutral =[[sum_Neutral[i][j]/float(sum_All[i][j])for j in
             range(len(trials[i]))] for i in range(len(subjects))]
        norm_sum_Disgusted=[[sum_Disgusted[i][j]/float(sum_All[i][j])for j in
             range(len(trials[i]))] for i in range(len(subjects))]
        norm_sum_WS =[[sum_WS[i][j]/float(sum_All[i][j])for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        return [sum_Disgusted, sum_Neutral, sum_WS,sum_All,norm_sum_Disgusted,norm_sum_Neutral,norm_sum_WS]

    def get_average_fixation_length_each_trial(self):
        # param order of trials
#add linear
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = sorted([set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects])
        All_fixations = [[self.fixation_dataset[(self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
             for j in trials[i]] for i in range(len(subjects))]
        mean_All = [[np.nanmean(All_fixations[i][j].Fixation_Duration) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        mean_Neutral = [
            [np.nanmean(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "N")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        mean_Disgusted = [
            [np.nanmean(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "D")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        mean_WS = [[np.mean(All_fixations[i][j].Fixation_Duration[(self.fixation_dataset.AOI_Group == "White Space")]) for j in
             range(len(trials[i]))] for i in range(len(subjects))]
        norm_mean_Neutral =[[mean_Neutral[i][j]/float(mean_All[i][j])for j in
             range(len(trials[i]))] for i in range(len(subjects))]
        norm_mean_Disgusted=[[mean_Disgusted[i][j]/float(mean_All[i][j])for j in
             range(len(trials[i]))] for i in range(len(subjects))]
        norm_mean_WS =[[mean_WS[i][j]/float(mean_All[i][j])for j in
             range(len(trials[i]))] for i in range(len(subjects))]

        reggression_mean_All = sine(mean_All)
        self.output_data_dict["trials_mean_All_est_std"] = reggression_mean_All[0]
        self.output_data_dict["trials_mean_All_est_phase"] = reggression_mean_All[1]
        self.output_data_dict["trials_mean_All_est_mean"] = reggression_mean_All[2]


        # linear
        linear_mean_All = linear(mean_All)
        self.output_data_dict["trials_mean_All_1coeff"] = linear_mean_All[0]
        self.output_data_dict["trials_mean_All_2coeff"] = linear_mean_All[1]


        reggression_mean_Neutral = sine(mean_Neutral)
        self.output_data_dict["trials_mean_Neutral_est_std"] = reggression_mean_Neutral[0]
        self.output_data_dict["trials_mean_Neutral_est_phase"] = reggression_mean_Neutral[1]
        self.output_data_dict["trials_mean_Neutral_est_mean"] = reggression_mean_Neutral[2]


        # linear
        linear_mean_Neutral = linear(mean_Neutral)
        self.output_data_dict["trials_mean_Neutral_1coeff"] = linear_mean_Neutral[0]
        self.output_data_dict["trials_mean_Neutral_2coeff"] = linear_mean_Neutral[1]


        reggression_mean_Disgusted = sine(mean_Disgusted)
        self.output_data_dict["trials_mean_Disgusted_est_std"] = reggression_mean_Disgusted[0]
        self.output_data_dict["trials_mean_Disgusted_est_phase"] = reggression_mean_Disgusted[1]
        self.output_data_dict["trials_mean_Disgusted_est_mean"] = reggression_mean_Disgusted[2]


        # linear
        linear_mean_Disgusted = linear(mean_Disgusted)
        self.output_data_dict["trials_mean_Disgusted_1coeff"] = linear_mean_Disgusted[0]
        self.output_data_dict["trials_mean_Disgusted_2coeff"] = linear_mean_Disgusted[1]


        reggression_mean_WS = sine(mean_WS)
        self.output_data_dict["trials_mean_WS_est_std"] = reggression_mean_WS[0]
        self.output_data_dict["trials_mean_WS_est_phase"] = reggression_mean_WS[1]
        self.output_data_dict["trials_mean_WS_est_mean"] = reggression_mean_WS[2]


        # linear
        linear_mean_WS = linear(mean_WS)
        self.output_data_dict["trials_mean_WS_1coeff"] = linear_mean_WS[0]
        self.output_data_dict["trials_mean_WS_2coeff"] = linear_mean_WS[1]


        reggression_norm_mean_Neutral = sine(norm_mean_Neutral)
        self.output_data_dict["trials_norm_mean_Neutral_est_std"] = reggression_norm_mean_Neutral[0]
        self.output_data_dict["trials_norm_mean_Neutral_est_phase"] = reggression_norm_mean_Neutral[1]
        self.output_data_dict["trials_norm_mean_Neutral_mean"] = reggression_norm_mean_Neutral[2]


        # linear
        linear_norm_mean_Neutral = linear(norm_mean_Neutral)
        self.output_data_dict["trials_norm_mean_Neutral_1coeff"] = linear_norm_mean_Neutral[0]
        self.output_data_dict["trials_norm_mean_Neutral_2coeff"] = linear_norm_mean_Neutral[1]


        reggression_norm_mean_Disgusted = sine(norm_mean_Disgusted)
        self.output_data_dict["trials_norm_mean_Disgusted_est_std"] = reggression_norm_mean_Disgusted[0]
        self.output_data_dict["trials_norm_mean_Disgusted_est_phase"] = reggression_norm_mean_Disgusted[1]
        self.output_data_dict["trials_norm_mean_Disgusted_est_mean"] = reggression_norm_mean_Disgusted[2]


        # linear
        linear_norm_mean_Disgusted = linear(norm_mean_Disgusted)
        self.output_data_dict["trials_norm_mean_Disgusted_1coeff"] = linear_norm_mean_Disgusted[0]
        self.output_data_dict["trials_norm_mean_Disgusted_2coeff"] = linear_norm_mean_Disgusted[1]


        reggression_norm_mean_WS = sine(norm_mean_WS)
        self.output_data_dict["trials_norm_mean_WS_est_std"] = reggression_norm_mean_WS[0]
        self.output_data_dict["trials_norm_mean_WS_est_phase"] = reggression_norm_mean_WS[1]
        self.output_data_dict["trials_norm_mean_WS_est_mean"] = reggression_norm_mean_WS[2]


        # linear
        linear_norm_mean_WS = linear(norm_mean_WS)
        self.output_data_dict["trials_norm_mean_WS_1coeff"] = linear_norm_mean_WS[0]
        self.output_data_dict["trials_norm_mean_WS_2coeff"] = linear_norm_mean_WS[1]


        return [mean_Disgusted, mean_Neutral, mean_WS,mean_All,norm_mean_Disgusted,norm_mean_Neutral,norm_mean_WS]


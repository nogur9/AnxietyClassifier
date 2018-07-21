import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from DataImporting import DataFromExcel
Fixation_length_cutoff = 100

class Data:
    grouping_function = None
    fixation_dataset = None
    output_data_frame = None
    output_data_dict = {}
    demographic_dataset = None

    def __init__(self, path, fixation_dataset_sheet_name,demographic_dataset_sheet_name = None, grouping_function=np.nansum):

        self.fixation_dataset = DataFromExcel.get_data(path, fixation_dataset_sheet_name)
        self.fixation_dataset = self.fixation_dataset[self.fixation_dataset.Fixation_Duration > Fixation_length_cutoff]
        self.grouping_function = grouping_function

        if not demographic_dataset_sheet_name is None:
            self.demographic_dataset = DataFromExcel.get_data(path, demographic_dataset_sheet_name)

        self.Trials_count = np.array([len(set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == j])) for j in
                   sorted(set(self.fixation_dataset.Subject))])

    def get_age (self):
        Age = [self.demographic_dataset.Age[i] for i in range(len(self.demographic_dataset.Age))]
        self.output_data_dict["Age"] = Age


    def get_PHQ9 (self):
        PHQ9 = [self.demographic_dataset.PHQ[i] for i in range(len(self.demographic_dataset.PHQ))]
        self.output_data_dict["PHQ9"] = PHQ9



    def get_subject_number (self):
        subject_number = list(sorted(set(self.fixation_dataset.Subject)))
        self.output_data_dict["Subject_Number"] = subject_number

    def get_trial (self):
        trial_number = list(sorted(set(self.fixation_dataset.Trial)))
        self.output_data_dict["Trial"] = trial_number
    def get_group(self):
        group = [self.demographic_dataset.group[i] for i in range(len(self.demographic_dataset.group))]
        self.output_data_dict["group"] = group



# feature extraction - features need to be computed

    def get_DT_each_stimulus_pet_trial (self):
 #       subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials =list(set(self.fixation_dataset.Stimulus))
        images = list(set(self.fixation_dataset.Area_of_Interest))

        for aoi in images:
            for mat in trials:
                sum_fix = np.nansum(self.fixation_dataset.Fixation_Duration[
                                                    (self.fixation_dataset.Stimulus == mat) & (
                                                    self.fixation_dataset.Area_of_Interest == aoi)])


                title_name = "sum_fixations_on_stimulus_{0} Trial {1}".format(aoi,mat)
                self.output_data_dict[title_name] = [sum_fix]

    def get_sum_fixation_length_Disgusted (self):
        norm_factor = self.get_sum_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        Sum_Disgusted = [self.grouping_function([np.sum(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Trial==j)&(self.fixation_dataset.AOI_Group == "D")]) for j in trials[i]]) for i in range(len(subjects))]
        Sum_Disgusted = [0 if math.isnan(x) else x for x in Sum_Disgusted]
        norm_disgusted =[Sum_Disgusted[i] / float(norm_factor[i]) for i in range(len(Sum_Disgusted))]
        norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]
        self.output_data_dict["sum_fixation_length_Disgusted"] = Sum_Disgusted
        self.output_data_dict["normalized_sum_fixation_length_Disgusted"] = norm_disgusted


    def get_sum_fixation_length_Neutral (self):
        norm_factor = self.get_sum_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        Sum_Neutral = [self.grouping_function([np.sum(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Trial==j)&(self.fixation_dataset.AOI_Group == "N")]) for j in trials[i]]) for i in range(len(subjects))]
        Sum_Neutral = [0 if math.isnan(x) else x for x in Sum_Neutral]
        norm_neutral =[Sum_Neutral[i] / float(norm_factor[i]) for i in range(len(Sum_Neutral))]
        norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict["sum_fixation_length_Neutral"] = Sum_Neutral
        self.output_data_dict["normalized_sum_fixation_length_Neutral"] = norm_neutral

    def get_sum_fixation_length_White_Space (self):
        norm_factor = self.get_sum_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        Sum_White_Space = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Trial==j)&(self.fixation_dataset.AOI_Group == "White Space")]) for j in trials[i]]) for i in range(len(subjects))]
        Sum_White_Space = [0 if math.isnan(x) else x for x in Sum_White_Space]
        norm_WS =[Sum_White_Space[i] / float(norm_factor[i]) for i in range(len(Sum_White_Space))]
        norm_WS = [0 if math.isnan(x) else x for x in norm_WS]
        self.output_data_dict["sum_fixation_length_White_Space"] = Sum_White_Space

        self.output_data_dict["normalized_sum_fixation_length_WS"] = norm_WS


    def get_sum_fixation_length_All (self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        Sum_All = [self.grouping_function([np.sum(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Trial==j)]) for j in trials[i]]) for i in range(len(subjects))]
        Sum_All = [0 if math.isnan(x) else x for x in Sum_All]

        self.output_data_dict["sum_fixation_length_All"] = Sum_All
        return Sum_All

    def get_average_fixation_length_Disgusted (self):
        norm_factor = self.get_average_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        mean_Disgusted = [np.mean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "D")]) for i in range(len(subjects))]
        mean_Disgusted = [0 if math.isnan(x) else x for x in mean_Disgusted]
        norm_disgusted =[mean_Disgusted[i] / float(norm_factor[i]) for i in range(len(mean_Disgusted))]
        norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]

        self.output_data_dict["average_fixation_length_Disgusted"] = mean_Disgusted
        self.output_data_dict["normalized_mean_fixation_length_Disgusted"] = norm_disgusted

    def get_average_fixation_length_Neutral (self):
        norm_factor = self.get_average_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        mean_Neutral = [np.mean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "N")]) for i in range(len(subjects))]
        mean_Neutral = [0 if math.isnan(x) else x for x in mean_Neutral]
        norm_neutral =[mean_Neutral[i] / float(norm_factor[i]) for i in range(len(mean_Neutral))]
        norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict["average_fixation_length_Neutral"] = mean_Neutral
        self.output_data_dict["normalized_mean_fixation_length_Neutral"] = norm_neutral

    def get_average_fixation_length_White_Space (self):
        norm_factor = self.get_average_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        mean_White_Space = [np.mean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "White Space")]) for i in range(len(subjects))]
        mean_White_Space = [0 if math.isnan(x) else x for x in mean_White_Space]
        norm_WS =[mean_White_Space[i] / float(norm_factor[i]) for i in range(len(mean_White_Space))]
        norm_WS = [0 if math.isnan(x) else x for x in norm_WS]

        self.output_data_dict ["average_fixation_length_White_Space"] = mean_White_Space
        self.output_data_dict["normalized_mean_fixation_length_WS"] = norm_WS

    def get_average_fixation_length_All (self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        mean_All = [np.mean(self.fixation_dataset.Fixation_Duration[self.fixation_dataset.Subject == subjects[i]]) for i in range(len(subjects))]
        mean_All = [0 if math.isnan(x) else x for x in mean_All]

        self.output_data_dict["average_fixation_length_All"] = mean_All
        return mean_All

    def get_amount_fixation_Disgusted (self):
        norm_factor = self.get_amount_fixation_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        amount_Disgusted = [self.grouping_function([len(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Trial==j)&(self.fixation_dataset.AOI_Group == "D")]) for j in trials[i]]) for i in range(len(subjects))]
        amount_Disgusted = [0 if math.isnan(x) else x for x in amount_Disgusted]
        norm_disgusted =[amount_Disgusted[i] / float(norm_factor[i]) for i in range(len(amount_Disgusted))]
        norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]

        self.output_data_dict ["amount_fixation_Disgusted"] = amount_Disgusted
        self.output_data_dict["normalized_amount_fixation_Disgusted"] = norm_disgusted

    def get_amount_fixation_Neutral (self):
        norm_factor = self.get_amount_fixation_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        amount_Neutral = [self.grouping_function([len(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Trial==j)&(self.fixation_dataset.AOI_Group == "N")]) for j in trials[i]]) for i in range(len(subjects))]
        amount_Neutral = [0 if math.isnan(x) else x for x in amount_Neutral]
        norm_neutral =[amount_Neutral[i] / float(norm_factor[i]) for i in range(len(amount_Neutral))]
        norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict ["amount_fixation_Neutral"] = amount_Neutral
        self.output_data_dict["normalized_amount_fixation_Neutral"] = norm_neutral


    def get_amount_fixation_White_Space (self):
        norm_factor = self.get_amount_fixation_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        amount_White_Space = [self.grouping_function([len(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Trial==j)&(self.fixation_dataset.AOI_Group == "White Space")]) for j in trials[i]]) for i in range(len(subjects))]
        amount_White_Space = [0 if math.isnan(x) else x for x in amount_White_Space]
        norm_WS =[amount_White_Space[i] / float(norm_factor[i]) for i in range(len(amount_White_Space))]
        norm_WS = [0 if math.isnan(x) else x for x in norm_WS]

        self.output_data_dict["amount_fixation_White_Space"] = amount_White_Space
        self.output_data_dict["normalized_amount_fixation_WS"] = norm_WS

    def get_amount_fixation_All (self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        amount_All = [self.grouping_function([len(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Trial==j)]) for j in trials[i]]) for i in range(len(subjects))]
        amount_All = [0 if math.isnan(x) else x for x in amount_All]

        self.output_data_dict["amount_fixation_All"] = amount_All
        return amount_All



    def get_STD_fixation_length_Disgusted(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        STD_Disgusted = [np.std(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "D")]) for i in range(len(subjects))]
        STD_Disgusted = [0 if math.isnan(x) else x for x in STD_Disgusted]

        self.output_data_dict["STD_fixation_length_Disgusted"] = STD_Disgusted


    def get_STD_fixation_length_Neutral(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        STD_Neutral = [np.std(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "N")]) for i in range(len(subjects))]
        STD_Neutral = [0 if math.isnan(x) else x for x in STD_Neutral]

        self.output_data_dict ["STD_fixation_length_Neutral"] = STD_Neutral


    def get_STD_fixation_length_White_Space(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        STD_White_Space = [np.std(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.AOI_Group == "White Space")])for i in range(len(subjects))]

        STD_White_Space = [0 if math.isnan(x) else x for x in STD_White_Space]

        self.output_data_dict["STD_fixation_length_White_Space"] = STD_White_Space



    def get_STD_fixation_length_All(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        STD_All = [np.std(self.fixation_dataset.Fixation_Duration[self.fixation_dataset.Subject == subjects[i]])for i in range(len(subjects))]
        STD_All = [0 if math.isnan(x) else x for x in STD_All]

        self.output_data_dict["STD_fixation_length_All"] = STD_All



    def get_ratio_D_DN (self):
        """
        
        :return: the ratio of the sum fixation length of disgusted and neutral fixations 
        """
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        mean_Disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "D")])  for i in range(len(subjects))]

        mean_Neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "N")])  for i in range(len(subjects))]

        ratio = [mean_Disgusted[i]/float(mean_Neutral[i]+mean_Disgusted[i]) for i in range(len(mean_Disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["Ratio D/D+N"] = ratio


    def get_ratio_N_DN(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        mean_Disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i]) &(self.fixation_dataset.AOI_Group == "D")])
                         for i in range(len(subjects))]

        mean_Neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                           (self.fixation_dataset.Subject == subjects[i]) &  (
                                                       self.fixation_dataset.AOI_Group == "N")])
                       for i in range(len(subjects))]

        ratio = [mean_Neutral[i] / float(mean_Neutral[i] + mean_Disgusted[i]) for i in range(len(mean_Disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["Ratio N/D+N"] = ratio

    def get_ratio_WS_All(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        mean_Disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                             (self.fixation_dataset.Subject == subjects[i]) & (
                                                         self.fixation_dataset.AOI_Group == "D")]) for i in range(len(subjects))]
        mean_Neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                           (self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "N")]) for i in range(len(subjects))]
        mean_WS = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])& (self.fixation_dataset.AOI_Group == "White Space")]) for i in range(len(subjects))]

        ratio = [mean_WS[i] / float(mean_WS[i]+ mean_Neutral[i] + mean_Disgusted[i]) for i in range(len(mean_Disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["Ratio WS/WS+N+D"] = ratio


    def get_ratio_D_DN_2(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        mean_Disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                             (self.fixation_dataset.Subject == subjects[i]) &
                                                         (self.fixation_dataset.AOI_Group == "D")])
                         for i in range(len(subjects))]
        mean_Neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                           (self.fixation_dataset.Subject == subjects[i]) &
                                                       (self.fixation_dataset.AOI_Group == "N")])
                       for i in range(len(subjects))]
        mean_WS = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                           (self.fixation_dataset.Subject == subjects[i])  & (self.fixation_dataset.AOI_Group == "White Space")])
                       for i in range(len(subjects))]

        ratio = [mean_Disgusted[i] / float(mean_WS[i]+ mean_Neutral[i] + mean_Disgusted[i]) for i in range(len(mean_Disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["%threat - #2"] = ratio


    def get_ratio_N_DN_2(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        mean_Disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                             (self.fixation_dataset.Subject == subjects[i]) &
                                                         (self.fixation_dataset.AOI_Group == "D")])
                         for i in range(len(subjects))]
        mean_Neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                           (self.fixation_dataset.Subject == subjects[i]) &
                                                       (self.fixation_dataset.AOI_Group == "N")])
                       for i in range(len(subjects))]
        mean_WS = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                           (self.fixation_dataset.Subject == subjects[i])& (self.fixation_dataset.AOI_Group == "White Space")])
                       for i in range(len(subjects))]

        ratio = [mean_Neutral[i] / float(mean_WS[i]+ mean_Neutral[i] + mean_Disgusted[i]) for i in range(len(mean_Disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["%neutral - #2"] = ratio



    def get_amount_DN_transitions(self):
        norm_factor = self.get_amount_fixation_All()
        #param - in each trial the cells are sorted by fixation start
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        one_hot_D = [np.array((self.fixation_dataset.Subject == subjects[i])& (self.fixation_dataset.AOI_Group == "D")) for i in range(len(subjects))]

        one_hot_N = [np.array((self.fixation_dataset.Subject == subjects[i])& (self.fixation_dataset.AOI_Group == "N")) for i in range(len(subjects))]
        length_one_hot_D = [len(one_hot_D[i])-1 for i in range(len(subjects))]
        one_hot_D = [np.append(one_hot_D[i][length_one_hot_D[i]::], one_hot_D[i][:length_one_hot_D[i]:]) for i in range(len(subjects))]
        indexes = [np.where(self.fixation_dataset[self.fixation_dataset.Subject == subjects[i]].Number < 2)
                   for i in range(len(subjects))]
        for i in range(len(subjects)):
            one_hot_D[i][indexes[i][0]] = False
        DN_transitions = [np.sum(one_hot_D[i]&one_hot_N[i]) for i in range(len(subjects))]

        norm_DN_transitions = [DN_transitions[i]/float(norm_factor[i]-1) for i in range(len(subjects))]

        self.output_data_dict["amount_DN_transitions"] = DN_transitions
        self.output_data_dict["norm_amount_DN_transitions"] = norm_DN_transitions


    def get_amount_ND_transitions(self):
        norm_factor = self.get_amount_fixation_All()
        # param - in each trial the cells are sorted by fixation start

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        one_hot_D = [np.array((self.fixation_dataset.Subject == subjects[i])& (self.fixation_dataset.AOI_Group == "D")) for i in range(len(subjects))]
        one_hot_N = [np.array((self.fixation_dataset.Subject == subjects[i])& (self.fixation_dataset.AOI_Group == "N")) for i in range(len(subjects))]

        length_one_hot_N = [len(one_hot_N[i])-1 for i in range(len(subjects))]
        one_hot_N = [np.append(one_hot_N[i][length_one_hot_N[i]::], one_hot_N[i][:length_one_hot_N[i]:]) for i in range(len(subjects))]
        indexes = [np.where(self.fixation_dataset[self.fixation_dataset.Subject == subjects[i]].Number < 2) for i in range(len(subjects))]
        for i in range(len(subjects)):
            one_hot_N[i][indexes[i][0]] = False
        ND_transitions = [np.sum(one_hot_D[i]&one_hot_N[i]) for i in range(len(subjects))]
        norm_ND_transitions = [ND_transitions[i] / float(norm_factor[i] - 1) for i in range(len(subjects))]
        self.output_data_dict["amount_ND_transitions"] = ND_transitions
        self.output_data_dict["norm_amount_ND_transitions"] = norm_ND_transitions


    def get_amount_DD_transitions(self):
        norm_factor = self.get_amount_fixation_All()
        # param - in each trial the cells are sorted by fixation start

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        one_hot_D_1 = [np.array((self.fixation_dataset.Subject == subjects[i])& (self.fixation_dataset.AOI_Group == "D")) for i in range(len(subjects))]
        one_hot_D_2 = [np.array((self.fixation_dataset.Subject == subjects[i])& (self.fixation_dataset.AOI_Group == "D")) for i in range(len(subjects))]
        length_one_hot_D_1 = [len(one_hot_D_1[i])-1 for i in range(len(subjects))]
        one_hot_D_1 = [np.append(one_hot_D_1[i][length_one_hot_D_1[i]::], one_hot_D_1[i][:length_one_hot_D_1[i]:]) for i in range(len(subjects))]
        indexes = [np.where(self.fixation_dataset[self.fixation_dataset.Subject == subjects[i]].Number < 2) for i in range(len(subjects))]
        for i in range(len(subjects)):
            one_hot_D_1[i][indexes[i][0]] = False
        DD_transitions = [np.sum(one_hot_D_1[i]&one_hot_D_2[i]) for i in range(len(subjects))]
        norm_DD_transitions = [DD_transitions[i] / float(norm_factor[i] - 1) for i in range(len(subjects))]
        self.output_data_dict["amount_DD_transitions"] = DD_transitions
        self.output_data_dict["norm_amount_DD_transitions"] = norm_DD_transitions


    def get_amount_NN_transitions(self):
        # param - in each trial the cells are sorted by fixation start
        norm_factor = self.get_amount_fixation_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        one_hot_N_1 = [np.array((self.fixation_dataset.Subject == subjects[i])& (self.fixation_dataset.AOI_Group == "N")) for i in range(len(subjects))]
        one_hot_N_2 = [np.array((self.fixation_dataset.Subject == subjects[i])& (self.fixation_dataset.AOI_Group == "N")) for i in range(len(subjects))]

        length_one_hot_N_1 = [len(one_hot_N_1[i])-1 for i in range(len(subjects))]
        one_hot_N_1 = [np.append(one_hot_N_1[i][length_one_hot_N_1[i]::], one_hot_N_1[i][:length_one_hot_N_1[i]:]) for i in range(len(subjects))]
        indexes = [np.where(self.fixation_dataset[self.fixation_dataset.Subject == subjects[i]].Number < 2) for i in range(len(subjects))]
        for i in range(len(subjects)):
            one_hot_N_1[i][indexes[i][0]] = False
        NN_transitions = [np.sum(one_hot_N_1[i]&one_hot_N_2[i]) for i in range(len(subjects))]

        norm_NN_transitions = [NN_transitions[i] / float(norm_factor[i] - 1) for i in range(len(subjects))]
        self.output_data_dict["amount_NN_transitions"] = NN_transitions
        self.output_data_dict["norm_amount_NN_transitions"] = norm_NN_transitions

    def get_amount_diff_AOI_transitions(self):
        #didnt add the trials splitting option
        norm_factor = self.get_amount_fixation_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        AOI_before = [self.fixation_dataset.AOI_Group[self.fixation_dataset.Subject == subjects[i]] for i in range(len(subjects))]
        AOI_after = [self.fixation_dataset.AOI_Group[self.fixation_dataset.Subject == subjects[i]] for i in range(len(subjects))]

        length_AOI_before = [len(AOI_before[i])-1 for i in range(len(subjects))]
        AOI_before = [np.append(AOI_before[i][length_AOI_before[i]::], AOI_before[i][:length_AOI_before[i]:]) for i in range(len(subjects))]
        indexes = [np.where(self.fixation_dataset[self.fixation_dataset.Subject == subjects[i]].Number < 2) for i in range(len(subjects))]
        for i in range(len(subjects)):
            AOI_before[i][indexes[i][0]] = ""
        diff_AOI_transitions = [np.sum(AOI_before[i]!= AOI_after[i])-len(indexes[i][0]) for i in range(len(subjects))]
        norm_diff_AOI_transitions = [diff_AOI_transitions[i] / float(norm_factor[i] - 1) for i in range(len(subjects))]
        self.output_data_dict["amount_diff_AOI_transitions"] = diff_AOI_transitions
        self.output_data_dict["norm_amount_diff_AOI_transitions"] = norm_diff_AOI_transitions


    def var_threat_precentage_between_trials(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        Mean_Disgusted = [[np.nanmean(self.fixation_dataset.Fixation_Duration[
                                             (self.fixation_dataset.Subject == subjects[i]) & (
                                                         self.fixation_dataset.Trial == j) & (
                                                         self.fixation_dataset.AOI_Group == "D")]) for j in trials[i]]
                         for i in range(len(subjects))]
        Mean_Neutral = [[np.nanmean(self.fixation_dataset.Fixation_Duration[
                                           (self.fixation_dataset.Subject == subjects[i]) & (
                                                       self.fixation_dataset.Trial == j) & (
                                                       self.fixation_dataset.AOI_Group == "N")]) for j in trials[i]]
                       for i in range(len(subjects))]

        ratio = [np.var([(Mean_Disgusted[i][j] / float(Mean_Neutral[i][j] + Mean_Disgusted[i][j])) for j in range(len(trials[i]))]) for i in range(len(Mean_Disgusted))]
        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["var_ratio_D_DN"] = ratio


    def amount_of_first_fixations (self):
        # param - in each trial the cells are sorted by fixation start

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]
        All_fixations = [[self.fixation_dataset.AOI_Group[(self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)] for j in trials[i]] for i in range(len(subjects))]
        amount_Disgusted = [np.sum([np.array(All_fixations[i][j].values[0] == "D") for j in range(len(trials[i]))])for i in range(len(subjects))]
        amount_Neutral = [np.sum([np.array(All_fixations[i][j].values[0] == "N") for j in range(len(trials[i]))]) for i in
                            range(len(subjects))]
        amount_WS = [np.sum([np.array(All_fixations[i][j].values[0] == "White Space") for j in range(len(trials[i]))]) for i in
                            range(len(subjects))]

        self.output_data_dict["amount_of_first_fixations_on_threat"] = amount_Disgusted
        self.output_data_dict["amount_of_first_fixations_on_neutral"] = amount_Neutral
        self.output_data_dict["amount_of_first_fixations_on_WS"] = amount_WS


    def amount_of_second_fixations(self):
        # param - in each trial the cells are sorted by fixation start

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]
        All_fixations = [[self.fixation_dataset.AOI_Group[(self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)] for j in trials[i]] for i in range(len(subjects))]
        amount_Disgusted = [np.sum([np.array(All_fixations[i][j].values[1] == "D") for j in range(len(trials[i]))])for i in range(len(subjects))]
        amount_Neutral = [np.sum([np.array(All_fixations[i][j].values[1] == "N") for j in range(len(trials[i]))]) for i in
                            range(len(subjects))]
        amount_WS = [np.sum([np.array(All_fixations[i][j].values[1] == "White Space") for j in range(len(trials[i]))]) for i in
                            range(len(subjects))]

        self.output_data_dict["amount_of_second_fixations_on_threat"] = amount_Disgusted
        self.output_data_dict["amount_of_second_fixations_on_neutral"] = amount_Neutral
        self.output_data_dict["amount_of_second_fixations_on_WS"] = amount_WS



    def get_average_pupil_size_Disgusted (self):
        norm_factor = self.get_average_pupil_size_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        #mean_Disgusted = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "D")&(self.fixation_dataset.Average_Pupil_Diameter!='-')]) for i in range(len(subjects))]
        mean_Disgusted = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[
                                      (self.fixation_dataset.Subject == subjects[i]) & (
                                                  self.fixation_dataset.AOI_Group == "D")]) for i in
                          range(len(subjects))]
        mean_Disgusted = [0 if math.isnan(x) else x for x in mean_Disgusted]
        norm_disgusted =[mean_Disgusted[i] / float(norm_factor[i]) for i in range(len(mean_Disgusted))]
        norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]

        self.output_data_dict["average_pupil_size_Disgusted"] = mean_Disgusted
        self.output_data_dict["normalized_mean_pupil_size_Disgusted"] = norm_disgusted

    def get_average_pupil_size_Neutral (self):
        norm_factor = self.get_average_pupil_size_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        #mean_Neutral = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "N")&(self.fixation_dataset.Average_Pupil_Diameter!='-')]) for i in range(len(subjects))]
        mean_Neutral = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[
                                    (self.fixation_dataset.Subject == subjects[i]) & (
                                                self.fixation_dataset.AOI_Group == "N")]) for i in
                        range(len(subjects))]
        mean_Neutral = [0 if math.isnan(x) else x for x in mean_Neutral]
        norm_neutral =[mean_Neutral[i] / float(norm_factor[i]) for i in range(len(mean_Neutral))]
        norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict["average_pupil_size_Neutral"] = mean_Neutral
        self.output_data_dict["normalized_mean_pupil_size_Neutral"] = norm_neutral

    def get_average_pupil_size_White_Space (self):
        norm_factor = self.get_average_pupil_size_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        #mean_White_Space = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "White Space")&(self.fixation_dataset.Average_Pupil_Diameter!='-')]) for i in range(len(subjects))]
        mean_White_Space = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[
                                        (self.fixation_dataset.Subject == subjects[i]) & (
                                                    self.fixation_dataset.AOI_Group == "White Space")]) for i in
                            range(len(subjects))]
        mean_White_Space = [0 if math.isnan(x) else x for x in mean_White_Space]
        norm_WS =[mean_White_Space[i] / float(norm_factor[i]) for i in range(len(mean_White_Space))]
        norm_WS = [0 if math.isnan(x) else x for x in norm_WS]

        self.output_data_dict["average_pupil_size_White_Space"] = mean_White_Space
        self.output_data_dict["normalized_mean_pupil_size_WS"] = norm_WS

    def get_average_pupil_size_All (self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        #mean_All = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Average_Pupil_Diameter!='-')]) for i in range(len(subjects))]
        mean_All = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[
                                (self.fixation_dataset.Subject == subjects[i])]) for i in
                    range(len(subjects))]
        mean_All = [0 if math.isnan(x) else x for x in mean_All]

        self.output_data_dict["average_pupil_size_All"] = mean_All
        return mean_All


    def get_STD_pupil_size_Disgusted(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        #STD_Disgusted = [np.std(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "D")&(self.fixation_dataset.Average_Pupil_Diameter!='-')]) for i in range(len(subjects))]
        STD_Disgusted = [np.std(self.fixation_dataset.Average_Pupil_Diameter[
                                    (self.fixation_dataset.Subject == subjects[i]) & (
                                                self.fixation_dataset.AOI_Group == "D")]) for i in
                         range(len(subjects))]
        STD_Disgusted = [0 if math.isnan(x) else x for x in STD_Disgusted]

        self.output_data_dict["STD_pupil_size_Disgusted"] = STD_Disgusted


    def get_STD_pupil_size_Neutral(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        #STD_Neutral = [np.std(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "N")&(self.fixation_dataset.Average_Pupil_Diameter!='-')]) for i in range(len(subjects))]
        STD_Neutral = [np.std(self.fixation_dataset.Average_Pupil_Diameter[
                                  (self.fixation_dataset.Subject == subjects[i]) & (
                                              self.fixation_dataset.AOI_Group == "N")]) for i in
                       range(len(subjects))]
        STD_Neutral = [0 if math.isnan(x) else x for x in STD_Neutral]

        self.output_data_dict["STD_pupil_size_Neutral"] = STD_Neutral


    def get_STD_pupil_size_White_Space(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        #STD_White_Space = [np.std(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.AOI_Group == "White Space")&(self.fixation_dataset.Average_Pupil_Diameter!='-')])for i in range(len(subjects))]
        STD_White_Space = [np.std(self.fixation_dataset.Average_Pupil_Diameter[
                                      (self.fixation_dataset.Subject == subjects[i]) & (
                                                  self.fixation_dataset.AOI_Group == "White Space")]) for i in
                           range(len(subjects))]
        STD_White_Space = [0 if math.isnan(x) else x for x in STD_White_Space]

        self.output_data_dict["STD_pupil_size_White_Space"] = STD_White_Space


    def get_STD_pupil_size_All(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        STD_All = [np.std(self.fixation_dataset.Average_Pupil_Diameter[
                              (self.fixation_dataset.Subject == subjects[i])]) for i in
                   range(len(subjects))]
        STD_All = [0 if math.isnan(x) else x for x in STD_All]
        self.output_data_dict["STD_pupil_size_All"] = STD_All


    def get_mean_different_AOI_per_trial (self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

        mean_AOIs = [np.nanmean([len(set(self.fixation_dataset.Area_of_Interest[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.Trial==j)])) for j in trials[i]]) for i in range(len(subjects))]
        mean_AOIs = [0 if math.isnan(x) else x for x in mean_AOIs]

        self.output_data_dict["mean_different_AOI_per_trial"] = mean_AOIs

    def get_difference_between_medians(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]
        medians = [[np.median(self.fixation_dataset.Number[(self.fixation_dataset.Subject == subjects[i]) &
                                                        (self.fixation_dataset.Trial == j)])
                                                        for j in trials[i]] for i in range(len(subjects))]
        #get sums

        sum_white_space_first_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                            [(self.fixation_dataset.Subject == subjects[i]) &
                            (self.fixation_dataset.Trial == j) & (self.fixation_dataset.AOI_Group == "White Space")
                             & (self.fixation_dataset.Number <= medians[i][j])])
                            for j in trials[i]]) for i in range(len(subjects))]

        sum_white_space_second_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                            [(self.fixation_dataset.Subject == subjects[i]) &
                            (self.fixation_dataset.Trial == j) & (self.fixation_dataset.AOI_Group == "White Space")
                             (self.fixation_dataset.Number > medians[i][j])])
                            for j in trials[i]]) for i in range(len(subjects))]

        sum_disgusted_first_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                            [(self.fixation_dataset.Subject == subjects[i]) &
                            (self.fixation_dataset.Trial == j) & (self.fixation_dataset.AOI_Group == "D")
                             & (self.fixation_dataset.Number <= medians[i][j])])
                            for j in trials[i]]) for i in range(len(subjects))]

        sum_disgusted_second_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                            [(self.fixation_dataset.Subject == subjects[i]) &
                            (self.fixation_dataset.Trial == j) & (self.fixation_dataset.AOI_Group == "D")
                            & (self.fixation_dataset.Number > medians[i][j])])
                            for j in trials[i]]) for i in range(len(subjects))]

        sum_neutral_first_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                            [(self.fixation_dataset.Subject == subjects[i]) &
                            (self.fixation_dataset.Trial == j) & (self.fixation_dataset.AOI_Group == "N")
                             & (self.fixation_dataset.Number <= medians[i][j])])
                            for j in trials[i]]) for i in range(len(subjects))]

        sum_neutral_second_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                            [(self.fixation_dataset.Subject == subjects[i]) &
                            (self.fixation_dataset.Trial == j) & (self.fixation_dataset.AOI_Group == "N")
                            & (self.fixation_dataset.Number > medians[i][j])])
                            for j in trials[i]]) for i in range(len(subjects))]

        sum_all_first_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                            [(self.fixation_dataset.Subject == subjects[i]) &
                            (self.fixation_dataset.Trial == j)
                             & (self.fixation_dataset.Number <= medians[i][j])])
                            for j in trials[i]]) for i in range(len(subjects))]

        sum_all_second_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                            [(self.fixation_dataset.Subject == subjects[i]) &
                            (self.fixation_dataset.Trial == j) &
                             (self.fixation_dataset.Number > medians[i][j])])
                            for j in trials[i]]) for i in range(len(subjects))]
        #get norm sums
        sum_all_first_median = [0 if math.isnan(x) else x for x in sum_all_first_median]
        sum_all_second_median = [0 if math.isnan(x) else x for x in sum_all_second_median]

        sum_white_space_first_median = [0 if math.isnan(x) else x for x in sum_white_space_first_median]
        norm_WS_first_median =[sum_white_space_first_median[i] / float(sum_all_first_median[i]) for i in range(len(sum_white_space_first_median))]
        norm_WS_first_median = [0 if math.isnan(x) else x for x in norm_WS_first_median]

        sum_white_space_second_median = [0 if math.isnan(x) else x for x in sum_white_space_second_median]
        norm_WS_second_median = [sum_white_space_second_median[i] / float(sum_all_second_median[i]) for i in range(len(sum_white_space_second_median))]
        norm_WS_second_median = [0 if math.isnan(x) else x for x in norm_WS_second_median]

        sum_disgusted_first_median = [0 if math.isnan(x) else x for x in sum_disgusted_first_median]
        norm_disgusted_first_median = [sum_disgusted_first_median[i] / float(sum_all_first_median[i]) for i in range(len(sum_disgusted_first_median))]
        norm_disgusted_first_median = [0 if math.isnan(x) else x for x in norm_disgusted_first_median]

        sum_disgusted_second_median = [0 if math.isnan(x) else x for x in sum_disgusted_second_median]
        norm_disgusted_second_median = [sum_disgusted_second_median[i] / float(sum_all_second_median[i]) for i in range(len(sum_disgusted_second_median))]
        norm_disgusted_second_median = [0 if math.isnan(x) else x for x in norm_disgusted_second_median]

        sum_neutral_first_median = [0 if math.isnan(x) else x for x in sum_neutral_first_median]
        norm_neutral_first_median =[sum_neutral_first_median[i] / float(sum_all_first_median[i]) for i in range(len(sum_neutral_first_median))]
        norm_neutral_first_median = [0 if math.isnan(x) else x for x in norm_neutral_first_median]

        sum_neutral_second_median = [0 if math.isnan(x) else x for x in sum_neutral_second_median]
        norm_neutral_second_median = [sum_neutral_second_median[i] / float(sum_all_second_median[i]) for i in range(len(sum_neutral_second_median))]
        norm_neutral_second_median = [0 if math.isnan(x) else x for x in norm_neutral_second_median]

        #get stds

        std_disgusted_first_median = [np.std(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "D") & (self.fixation_dataset.Number <= medians[i][j])]) for i in range(len(subjects))]
        std_disgusted_first_median = [0 if math.isnan(x) else x for x in std_disgusted_first_median]

        std_disgusted_second_median = [np.std(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "D") (self.fixation_dataset.Number > medians[i][j])]) for i in range(len(subjects))]
        std_disgusted_second_median = [0 if math.isnan(x) else x for x in std_disgusted_second_median]

        std_neutral_first_median = [np.std(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "N") & (self.fixation_dataset.Number <= medians[i][j])]) for i in range(len(subjects))]
        std_neutral_first_median = [0 if math.isnan(x) else x for x in std_neutral_first_median]

        std_neutral_second_median = [np.std(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i])&(self.fixation_dataset.AOI_Group == "N") (self.fixation_dataset.Number > medians[i][j])]) for i in range(len(subjects))]
        std_neutral_second_median = [0 if math.isnan(x) else x for x in std_neutral_second_median]

        self.output_data_dict["STD_Disgusted_difference_between_medians"] = (std_disgusted_first_median/float(std_disgusted_second_median))
        self.output_data_dict["STD_Neutral_difference_between_medians"] = (std_neutral_first_median/float(std_neutral_second_median))
        self.output_data_dict["sum_disgusted_difference_between_medians"] = (sum_disgusted_first_median/float(sum_disgusted_second_median))
        self.output_data_dict["norm_sum_disgusted_difference_between_medians"] = (norm_disgusted_first_median/float(norm_disgusted_second_median))
        self.output_data_dict["sum_neutral_difference_between_medians"] = (sum_neutral_first_median/float(sum_neutral_second_median))
        self.output_data_dict["norm_sum_disgusted_difference_between_medians"] = (norm_neutral_first_median/float(norm_neutral_second_median))
        self.output_data_dict["sum_WS_difference_between_medians"] = (sum_white_space_first_median/float(sum_white_space_second_median))
        self.output_data_dict["norm_sum_WS_difference_between_medians"] = (norm_WS_first_median/float(norm_WS_second_median))
        self.output_data_dict["sum_all_difference_between_medians"] = (sum_all_first_median/float(sum_all_second_median))


#Test_Data_path = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\Testers\\test data_ordered.xlsx"
#Test_Data_path = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\Testers\\301.xlsx"
#FIXATION_DATA_SHEET = 'Sheet1'
#Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
#Data_Object.get_amount_diff_AOI_transitions()
#Test_Data_path = "C:\\Users\\user\\PycharmProjects\\AnxietyClassifier\\Testers\\337.xlsx"
#FIXATION_DATA_SHEET = 'Sheet1'
#Data_Object = Data(Test_Data_path, FIXATION_DATA_SHEET)
#Data_Object.plot_timeline_between_trial()
#Data_Object.var_threat_precentage_between_trials()
#Data_Object.plot_timeline()
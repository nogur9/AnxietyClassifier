import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from DataImporting import ImportData
import pandas as pd
Fixation_length_cutoff = 100


class Data:
    grouping_function = None
    fixation_dataset = None
    output_data_frame = None
    output_data_dict = {}
    demographic_dataset = None

    def __init__(self, path, fixation_dataset_sheet_name, demographic_dataset_sheet_name=None,
                 grouping_function=np.nanmean):

        self.fixation_dataset = ImportData.get_data(path, fixation_dataset_sheet_name)
        self.fixation_dataset = self.fixation_dataset[self.fixation_dataset.Fixation_Duration > Fixation_length_cutoff]

        #self.fixation_dataset = self.fixation_dataset.reindex(np.arange(len(self.fixation_dataset)))

        # make it ignore fixation after the first 30
        #trials = self.fixation_dataset.groupby("Subject")["Trial"].unique().apply(lambda x: x[29])
        #ts = trials[self.fixation_dataset.Subject]
        #ts.index = np.arange(len(trials[self.fixation_dataset.Subject]))
        #indexer = self.fixation_dataset.Trial.reindex(np.arange(len(trials[self.fixation_dataset.Subject]))) < ts

        # self.fixation_dataset = self.fixation_dataset[indexer]

        self.grouping_function = grouping_function

        if not demographic_dataset_sheet_name is None:
            self.demographic_dataset = ImportData.get_data(path, demographic_dataset_sheet_name)

        self.Trials_count = np.array([len(set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == j])) for j in
                                      sorted(set(self.fixation_dataset.Subject))])

    def get_age(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        Age = [self.demographic_dataset.Age[(self.fixation_dataset.Subject == subjects[i])] for i in range(len(subjects))]
        self.output_data_dict["Age"] = Age

    def get_PHQ9(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        PHQ9 = [self.demographic_dataset.PHQ[(self.fixation_dataset.Subject == subjects[i])] for i in range(len(subjects))]
        self.output_data_dict["PHQ9"] = PHQ9

    def get_lsas(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        lsas = []
        for subject in subjects:
            lsas.append([self.demographic_dataset['LSAS_Total'][(self.demographic_dataset.Subject == subject)]][0].values[0])
        self.output_data_dict["LSAS"] = lsas

    def get_subject_number(self):
        subject_number = list(sorted(set(self.fixation_dataset.Subject)))
        self.output_data_dict["Subject_Number"] = subject_number

    def get_group(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        group = []
        for subject in subjects:
            group.append([self.demographic_dataset.group[(self.demographic_dataset.Subject == subject)]][0].values[0])
        self.output_data_dict["group"] = group

    # feature extraction - features need to be computed
    def get_Amits_feature(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique()[:30] for i in subjects]
        Sum_Disgusted = [np.nanmean([np.nansum(self.fixation_dataset.Fixation_Duration[
                                                            (self.fixation_dataset.Subject == subjects[i]) & (
                                                                    self.fixation_dataset.Trial == j) & (
                                                                    self.fixation_dataset.AOI_Group == "D")]) for j
                                                 in trials[i]]) for i in range(len(subjects))]

        self.output_data_dict["Amits"] = Sum_Disgusted

    def get_DT_each_stimulus_pet_trial(self):
        #       subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = list(set(self.fixation_dataset.Stimulus))
        images = list(set(self.fixation_dataset.Area_of_Interest))

        for aoi in images:
            for mat in trials:
                sum_fix = np.nansum(self.fixation_dataset.Fixation_Duration[
                                        (self.fixation_dataset.Stimulus == mat) & (
                                                self.fixation_dataset.Area_of_Interest == aoi)])

                title_name = "sum_fixations_on_stimulus_{0} Trial {1}".format(aoi, mat)
                self.output_data_dict[title_name] = [sum_fix]

    def get_sum_fixation_length_Disgusted(self):
        #norm_factor = self.get_sum_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique() for i in subjects]

        Sum_Disgusted = [np.nanmean([np.sum(self.fixation_dataset.Fixation_Duration[
                                                            (self.fixation_dataset.Subject == subjects[i]) & (
                                                                    self.fixation_dataset.Trial == j) & (
                                                                    self.fixation_dataset.AOI_Group == "D")]) for j
                                                 in trials[i]]) for i in range(len(subjects))]
        Sum_Disgusted = [0 if math.isnan(x) else x for x in Sum_Disgusted]
        #norm_disgusted = [Sum_Disgusted[i] / float(norm_factor[i]) for i in range(len(Sum_Disgusted))]
        #norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]
        self.output_data_dict["avg_of_sum_fixation_length_Disgusted"] = Sum_Disgusted
        #self.output_data_dict["normalized_sum_fixation_length_Disgusted"] = norm_disgusted

    def get_sum_fixation_length_Neutral(self):
        #norm_factor = self.get_sum_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique() for i in subjects]

        Sum_Neutral = [np.nanmean([np.sum(self.fixation_dataset.Fixation_Duration[
                                                          (self.fixation_dataset.Subject == subjects[i]) & (
                                                                  self.fixation_dataset.Trial == j) & (
                                                                  self.fixation_dataset.AOI_Group == "N")]) for j in
                                               trials[i]]) for i in range(len(subjects))]
        Sum_Neutral = [0 if math.isnan(x) else x for x in Sum_Neutral]
        #norm_neutral = [Sum_Neutral[i] / float(norm_factor[i]) for i in range(len(Sum_Neutral))]
        #norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict["avg_of_sum_fixation_length_Neutral"] = Sum_Neutral
        #self.output_data_dict["normalized_sum_fixation_length_Neutral"] = norm_neutral

    def get_sum_fixation_length_White_Space(self):
        #norm_factor = self.get_sum_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique() for i in subjects]

        Sum_White_Space = [np.nanmean([np.nansum(self.fixation_dataset.Fixation_Duration[
                                                                 (self.fixation_dataset.Subject == subjects[i]) & (
                                                                         self.fixation_dataset.Trial == j) & (
                                                                         self.fixation_dataset.AOI_Group == "White Space")])
                                                   for j in trials[i]]) for i in range(len(subjects))]
        Sum_White_Space = [0 if math.isnan(x) else x for x in Sum_White_Space]
        #norm_WS = [Sum_White_Space[i] / float(norm_factor[i]) for i in range(len(Sum_White_Space))]
        #norm_WS = [0 if math.isnan(x) else x for x in norm_WS]
        self.output_data_dict["avg_of_sum_fixation_length_White_Space"] = Sum_White_Space

        #self.output_data_dict["normalized_sum_fixation_length_WS"] = norm_WS

    def get_sum_fixation_length_All(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique() for i in subjects]

        Sum_All = [np.nanmean([np.sum(self.fixation_dataset.Fixation_Duration[
                                                      (self.fixation_dataset.Subject == subjects[i]) & (
                                                              self.fixation_dataset.Trial == j)]) for j in
                                           trials[i]]) for i in range(len(subjects))]
        Sum_All = [0 if math.isnan(x) else x for x in Sum_All]

        self.output_data_dict["avg_of_sum_fixation_length_All"] = Sum_All
        return Sum_All

    def get_average_fixation_length_Disgusted(self):
        # norm_factor = self.get_average_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        mean_Disgusted = [np.mean(self.fixation_dataset.Fixation_Duration[
                                      (self.fixation_dataset.Subject == subjects[i]) & (
                                              self.fixation_dataset.AOI_Group == "D")]) for i in
                          range(len(subjects))]
        mean_Disgusted = [0 if math.isnan(x) else x for x in mean_Disgusted]
        # norm_disgusted = [mean_Disgusted[i] / float(norm_factor[i]) for i in range(len(mean_Disgusted))]
        # norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]

        self.output_data_dict["average_fixation_length_Disgusted"] = mean_Disgusted
        # self.output_data_dict["normalized_mean_fixation_length_Disgusted"] = norm_disgusted

    def get_average_fixation_length_Neutral(self):
        # norm_factor = self.get_average_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        mean_Neutral = [np.mean(self.fixation_dataset.Fixation_Duration[
                                    (self.fixation_dataset.Subject == subjects[i]) & (
                                            self.fixation_dataset.AOI_Group == "N")]) for i in range(len(subjects))]
        mean_Neutral = [0 if math.isnan(x) else x for x in mean_Neutral]
        # norm_neutral = [mean_Neutral[i] / float(norm_factor[i]) for i in range(len(mean_Neutral))]
        # norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict["average_fixation_length_Neutral"] = mean_Neutral
        # self.output_data_dict["normalized_mean_fixation_length_Neutral"] = norm_neutral

    def get_average_fixation_length_White_Space(self):

        # norm_factor = self.get_average_fixation_length_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        mean_White_Space = [np.mean(self.fixation_dataset.Fixation_Duration[
                                        (self.fixation_dataset.Subject == subjects[i]) & (
                                                self.fixation_dataset.AOI_Group == "White Space")]) for i in
                            range(len(subjects))]
        mean_White_Space = [0 if math.isnan(x) else x for x in mean_White_Space]
        # norm_WS = [mean_White_Space[i] / float(norm_factor[i]) for i in range(len(mean_White_Space))]
        # norm_WS = [0 if math.isnan(x) else x for x in norm_WS]

        self.output_data_dict["average_fixation_length_White_Space"] = mean_White_Space
        # self.output_data_dict["normalized_mean_fixation_length_WS"] = norm_WS

    def get_average_fixation_length_All(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        mean_All = [np.mean(self.fixation_dataset.Fixation_Duration[self.fixation_dataset.Subject == subjects[i]]) for i
                    in range(len(subjects))]
        mean_All = [0 if math.isnan(x) else x for x in mean_All]

        self.output_data_dict["average_fixation_length_All"] = mean_All
        return mean_All

    def get_amount_fixation_Disgusted(self):
        #norm_factor = self.get_amount_fixation_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique() for i in subjects]

        amount_Disgusted = [np.nanmean([len(self.fixation_dataset.Fixation_Duration[
                                                            (self.fixation_dataset.Subject == subjects[i]) & (
                                                                    self.fixation_dataset.Trial == j) & (
                                                                    self.fixation_dataset.AOI_Group == "D")]) for j
                                                    in trials[i]]) for i in range(len(subjects))]
        amount_Disgusted = [0 if math.isnan(x) else x for x in amount_Disgusted]
        #norm_disgusted = [amount_Disgusted[i] / float(norm_factor[i]) for i in range(len(amount_Disgusted))]
        #norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]

        self.output_data_dict["avg_of_amount_fixation_Disgusted"] = amount_Disgusted
        #self.output_data_dict["normalized_amount_fixation_Disgusted"] = norm_disgusted

    def get_amount_fixation_Neutral(self):
        # norm_factor = self.get_amount_fixation_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique() for i in subjects]

        amount_Neutral = [np.nanmean([len(self.fixation_dataset.Fixation_Duration[
                                                          (self.fixation_dataset.Subject == subjects[i]) & (
                                                                  self.fixation_dataset.Trial == j) & (
                                                                  self.fixation_dataset.AOI_Group == "N")]) for j in
                                                  trials[i]]) for i in range(len(subjects))]
        amount_Neutral = [0 if math.isnan(x) else x for x in amount_Neutral]
        # norm_neutral = [amount_Neutral[i] / float(norm_factor[i]) for i in range(len(amount_Neutral))]
        # norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict["avg_of_amount_fixation_Neutral"] = amount_Neutral
        # self.output_data_dict["normalized_amount_fixation_Neutral"] = norm_neutral

    def get_amount_fixation_White_Space(self):
        # norm_factor = self.get_amount_fixation_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique() for i in subjects]

        amount_White_Space = [np.nanmean([len(self.fixation_dataset.Fixation_Duration[
                                                              (self.fixation_dataset.Subject == subjects[i]) & (
                                                                      self.fixation_dataset.Trial == j) & (
                                                                      self.fixation_dataset.AOI_Group == "White Space")])
                                                      for j in trials[i]]) for i in range(len(subjects))]
        amount_White_Space = [0 if math.isnan(x) else x for x in amount_White_Space]
        # norm_WS = [amount_White_Space[i] / float(norm_factor[i]) for i in range(len(amount_White_Space))]
        #norm_WS = [0 if math.isnan(x) else x for x in norm_WS]

        self.output_data_dict["avg_of_amount_fixation_White_Space"] = amount_White_Space
        # self.output_data_dict["normalized_amount_fixation_WS"] = norm_WS

    def get_amount_fixation_All(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique() for i in subjects]

        amount_All = [np.nanmean([len(self.fixation_dataset.Fixation_Duration[
                                                      (self.fixation_dataset.Subject == subjects[i]) & (
                                                              self.fixation_dataset.Trial == j)]) for j in
                                              trials[i]]) for i in range(len(subjects))]
        amount_All = [0 if math.isnan(x) else x for x in amount_All]

        self.output_data_dict["avg_of_amount_fixation_All"] = amount_All
        return amount_All

    def get_STD_fixation_length_Disgusted(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        STD_Disgusted = [np.std(self.fixation_dataset.Fixation_Duration[
                                    (self.fixation_dataset.Subject == subjects[i]) & (
                                            self.fixation_dataset.AOI_Group == "D")]) for i in range(len(subjects))]
        STD_Disgusted = [0 if math.isnan(x) else x for x in STD_Disgusted]

        self.output_data_dict["STD_fixation_length_Disgusted"] = STD_Disgusted

    def get_STD_fixation_length_Neutral(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        STD_Neutral = [np.std(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i]) & (
                self.fixation_dataset.AOI_Group == "N")]) for i in range(len(subjects))]
        STD_Neutral = [0 if math.isnan(x) else x for x in STD_Neutral]

        self.output_data_dict["STD_fixation_length_Neutral"] = STD_Neutral

    def get_STD_fixation_length_White_Space(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        STD_White_Space = [np.std(self.fixation_dataset.Fixation_Duration[
                                      (self.fixation_dataset.Subject == subjects[i]) & (
                                              self.fixation_dataset.AOI_Group == "White Space")]) for i in
                           range(len(subjects))]

        STD_White_Space = [0 if math.isnan(x) else x for x in STD_White_Space]

        self.output_data_dict["STD_fixation_length_White_Space"] = STD_White_Space

    def get_STD_fixation_length_All(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        STD_All = [np.std(self.fixation_dataset.Fixation_Duration[self.fixation_dataset.Subject == subjects[i]]) for i
                   in range(len(subjects))]
        STD_All = [0 if math.isnan(x) else x for x in STD_All]

        self.output_data_dict["STD_fixation_length_All"] = STD_All

    def get_ratio_D_DN(self):
        """
        
        :return: the ratio of the sum fixation length of disgusted and neutral fixations 
        """
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        mean_Disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                         (self.fixation_dataset.Subject == subjects[i]) & (
                                                 self.fixation_dataset.AOI_Group == "D")]) for i in
                          range(len(subjects))]

        mean_Neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                       (self.fixation_dataset.Subject == subjects[i]) & (
                                               self.fixation_dataset.AOI_Group == "N")]) for i in
                        range(len(subjects))]

        ratio = [mean_Disgusted[i] / float(mean_Neutral[i] + mean_Disgusted[i]) for i in range(len(mean_Disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["Ratio D/D+N"] = ratio

    def get_ratio_N_DN(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        #trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique()[:30] for i in subjects]

        mean_Disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                         (self.fixation_dataset.Subject == subjects[i]) & (
                                                 self.fixation_dataset.AOI_Group == "D")])
                          for i in range(len(subjects))]

        mean_Neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                       (self.fixation_dataset.Subject == subjects[i]) & (
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
        #trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique()[:30] for i in subjects]

        mean_Disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                         (self.fixation_dataset.Subject == subjects[i]) & (
                                                 self.fixation_dataset.AOI_Group == "D")]) for i in
                          range(len(subjects))]
        mean_Neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                       (self.fixation_dataset.Subject == subjects[i]) & (
                                               self.fixation_dataset.AOI_Group == "N")]) for i in
                        range(len(subjects))]
        mean_WS = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i]) & (
                self.fixation_dataset.AOI_Group == "White Space")]) for i in range(len(subjects))]

        ratio = [mean_WS[i] / float(mean_WS[i] + mean_Neutral[i] + mean_Disgusted[i]) for i in
                 range(len(mean_Disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["Ratio WS/WS+N+D"] = ratio

    def get_ratio_D_DN_2(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        #trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique()[:30] for i in subjects]

        mean_Disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                         (self.fixation_dataset.Subject == subjects[i]) &
                                         (self.fixation_dataset.AOI_Group == "D")])
                          for i in range(len(subjects))]
        mean_Neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                       (self.fixation_dataset.Subject == subjects[i]) &
                                       (self.fixation_dataset.AOI_Group == "N")])
                        for i in range(len(subjects))]
        mean_WS = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                  (self.fixation_dataset.Subject == subjects[i]) & (
                                          self.fixation_dataset.AOI_Group == "White Space")])
                   for i in range(len(subjects))]

        ratio = [mean_Disgusted[i] / float(mean_WS[i] + mean_Neutral[i] + mean_Disgusted[i]) for i in
                 range(len(mean_Disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["%threat - #2"] = ratio

    def get_ratio_N_DN_2(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        #trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique()[:30] for i in subjects]

        mean_Disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                         (self.fixation_dataset.Subject == subjects[i]) &
                                         (self.fixation_dataset.AOI_Group == "D")])
                          for i in range(len(subjects))]
        mean_Neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                       (self.fixation_dataset.Subject == subjects[i]) &
                                       (self.fixation_dataset.AOI_Group == "N")])
                        for i in range(len(subjects))]
        mean_WS = [np.nanmean(self.fixation_dataset.Fixation_Duration[
                                  (self.fixation_dataset.Subject == subjects[i]) & (
                                          self.fixation_dataset.AOI_Group == "White Space")])
                   for i in range(len(subjects))]

        ratio = [mean_Neutral[i] / float(mean_WS[i] + mean_Neutral[i] + mean_Disgusted[i]) for i in
                 range(len(mean_Disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["%neutral - #2"] = ratio

    def get_amount_DN_transitions(self):
        norm_factor = self.get_amount_fixation_All()
        # param - in each trial the cells are sorted by fixation start
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        one_hot_D = [np.array((self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.AOI_Group == "D"))
                     for i in range(len(subjects))]

        one_hot_N = [np.array((self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.AOI_Group == "N"))
                     for i in range(len(subjects))]
        length_one_hot_D = [len(one_hot_D[i]) - 1 for i in range(len(subjects))]
        one_hot_D = [np.append(one_hot_D[i][length_one_hot_D[i]::], one_hot_D[i][:length_one_hot_D[i]:]) for i in
                     range(len(subjects))]
        indexes = [np.where(self.fixation_dataset[self.fixation_dataset.Subject == subjects[i]].Number < 2)
                   for i in range(len(subjects))]
        for i in range(len(subjects)):
            one_hot_D[i][indexes[i][0]] = False
        DN_transitions = [np.sum(one_hot_D[i] & one_hot_N[i]) for i in range(len(subjects))]

        norm_DN_transitions = [DN_transitions[i] / float(norm_factor[i] - 1) for i in range(len(subjects))]

        self.output_data_dict["amount_DN_transitions"] = DN_transitions
        self.output_data_dict["norm_amount_DN_transitions"] = norm_DN_transitions

    def get_amount_ND_transitions(self):
        norm_factor = self.get_amount_fixation_All()
        # param - in each trial the cells are sorted by fixation start

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        one_hot_D = [np.array((self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.AOI_Group == "D"))
                     for i in range(len(subjects))]
        one_hot_N = [np.array((self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.AOI_Group == "N"))
                     for i in range(len(subjects))]

        length_one_hot_N = [len(one_hot_N[i]) - 1 for i in range(len(subjects))]
        one_hot_N = [np.append(one_hot_N[i][length_one_hot_N[i]::], one_hot_N[i][:length_one_hot_N[i]:]) for i in
                     range(len(subjects))]
        indexes = [np.where(self.fixation_dataset[self.fixation_dataset.Subject == subjects[i]].Number < 2) for i in
                   range(len(subjects))]
        for i in range(len(subjects)):
            one_hot_N[i][indexes[i][0]] = False
        ND_transitions = [np.sum(one_hot_D[i] & one_hot_N[i]) for i in range(len(subjects))]
        norm_ND_transitions = [ND_transitions[i] / float(norm_factor[i] - 1) for i in range(len(subjects))]
        self.output_data_dict["amount_ND_transitions"] = ND_transitions
        self.output_data_dict["norm_amount_ND_transitions"] = norm_ND_transitions

    def get_amount_DD_transitions(self):
        norm_factor = self.get_amount_fixation_All()
        # param - in each trial the cells are sorted by fixation start

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        one_hot_D_1 = [
            np.array((self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.AOI_Group == "D")) for i in
            range(len(subjects))]
        one_hot_D_2 = [
            np.array((self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.AOI_Group == "D")) for i in
            range(len(subjects))]
        length_one_hot_D_1 = [len(one_hot_D_1[i]) - 1 for i in range(len(subjects))]
        one_hot_D_1 = [np.append(one_hot_D_1[i][length_one_hot_D_1[i]::], one_hot_D_1[i][:length_one_hot_D_1[i]:]) for i
                       in range(len(subjects))]
        indexes = [np.where(self.fixation_dataset[self.fixation_dataset.Subject == subjects[i]].Number < 2) for i in
                   range(len(subjects))]
        for i in range(len(subjects)):
            one_hot_D_1[i][indexes[i][0]] = False
        DD_transitions = [np.sum(one_hot_D_1[i] & one_hot_D_2[i]) for i in range(len(subjects))]
        norm_DD_transitions = [DD_transitions[i] / float(norm_factor[i] - 1) for i in range(len(subjects))]
        self.output_data_dict["amount_DD_transitions"] = DD_transitions
        self.output_data_dict["norm_amount_DD_transitions"] = norm_DD_transitions

    def get_amount_NN_transitions(self):

        # param - in each trial the cells are sorted by fixation start
        norm_factor = self.get_amount_fixation_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        one_hot_N_1 = [
            np.array((self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.AOI_Group == "N")) for i in
            range(len(subjects))]
        one_hot_N_2 = [
            np.array((self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.AOI_Group == "N")) for i in
            range(len(subjects))]

        length_one_hot_N_1 = [len(one_hot_N_1[i]) - 1 for i in range(len(subjects))]
        one_hot_N_1 = [np.append(one_hot_N_1[i][length_one_hot_N_1[i]::], one_hot_N_1[i][:length_one_hot_N_1[i]:]) for i
                       in range(len(subjects))]
        indexes = [np.where(self.fixation_dataset[self.fixation_dataset.Subject == subjects[i]].Number < 2) for i in
                   range(len(subjects))]
        for i in range(len(subjects)):
            one_hot_N_1[i][indexes[i][0]] = False
        NN_transitions = [np.sum(one_hot_N_1[i] & one_hot_N_2[i]) for i in range(len(subjects))]

        norm_NN_transitions = [NN_transitions[i] / float(norm_factor[i] - 1) for i in range(len(subjects))]
        self.output_data_dict["amount_NN_transitions"] = NN_transitions
        self.output_data_dict["norm_amount_NN_transitions"] = norm_NN_transitions


    def get_amount_diff_AOI_transitions(self):
        # didnt add the trials splitting option
        norm_factor = self.get_amount_fixation_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        AOI_before = [self.fixation_dataset.AOI_Group[self.fixation_dataset.Subject == subjects[i]] for i in
                      range(len(subjects))]
        AOI_after = [self.fixation_dataset.AOI_Group[self.fixation_dataset.Subject == subjects[i]] for i in
                     range(len(subjects))]

        length_AOI_before = [len(AOI_before[i]) - 1 for i in range(len(subjects))]
        AOI_before = [np.append(AOI_before[i][length_AOI_before[i]::], AOI_before[i][:length_AOI_before[i]:]) for i in
                      range(len(subjects))]
        indexes = [np.where(self.fixation_dataset[self.fixation_dataset.Subject == subjects[i]].Number < 2) for i in
                   range(len(subjects))]
        for i in range(len(subjects)):
            AOI_before[i][indexes[i][0]] = ""
        diff_AOI_transitions = [np.sum(AOI_before[i] != AOI_after[i]) - len(indexes[i][0]) for i in
                                range(len(subjects))]
        norm_diff_AOI_transitions = [diff_AOI_transitions[i] / float(norm_factor[i] - 1) for i in range(len(subjects))]
        self.output_data_dict["amount_diff_AOI_transitions"] = diff_AOI_transitions
        self.output_data_dict["norm_amount_diff_AOI_transitions"] = norm_diff_AOI_transitions


    def var_threat_precentage_between_trials(self):
        """

        :return:
        """
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique() for i in subjects]

        Mean_Disgusted = [[np.nanmean(self.fixation_dataset.Fixation_Duration[
                                          (self.fixation_dataset.Subject == subjects[i]) &
                                          (self.fixation_dataset.Trial == j) &
                                          (self.fixation_dataset.AOI_Group == "D")]) for j in trials[i]]
                          for i in range(len(subjects))]
        Mean_Neutral = [[np.nanmean(self.fixation_dataset.Fixation_Duration[
                                        (self.fixation_dataset.Subject == subjects[i]) &
                                        (self.fixation_dataset.Trial == j) & (self.fixation_dataset.AOI_Group == "N")])
                         for j in trials[i]] for i in range(len(subjects))]

        ratio = [np.var([(Mean_Disgusted[i][j] / float(Mean_Neutral[i][j] + Mean_Disgusted[i][j]))
                         for j in range(len(trials[i]))]) for i in range(len(Mean_Disgusted))]
        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["var_ratio_D_DN"] = ratio

    def amount_of_first_fixations(self):
        # param - in each trial the cells are sorted by fixation start
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique()[:30] for i in subjects]
        All_fixations = [[self.fixation_dataset.AOI_Group[
                              (self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
                          for j in trials[i]] for i in range(len(subjects))]
        amount_Disgusted = [np.sum([np.array(All_fixations[i][j].values[0] == "D") for j in range(len(trials[i]))])
                            for i in range(len(subjects))]
        amount_Neutral = [np.sum([np.array(All_fixations[i][j].values[0] == "N") for j in range(len(trials[i]))])
                          for i in range(len(subjects))]
        amount_WS = [np.sum([np.array(All_fixations[i][j].values[0] == "White Space") for j in range(len(trials[i]))])
                     for i in range(len(subjects))]

        self.output_data_dict["amount_of_first_fixations_on_threat"] = amount_Disgusted
        self.output_data_dict["amount_of_first_fixations_on_neutral"] = amount_Neutral
        self.output_data_dict["amount_of_first_fixations_on_WS"] = amount_WS

    def amount_of_second_fixations(self):
        # param - in each trial the cells are sorted by fixation start

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique()[:30] for i in subjects]
        All_fixations = [[self.fixation_dataset.AOI_Group[
                              (self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.Trial == j)]
                          for j in trials[i]] for i in range(len(subjects))]
        amount_Disgusted = [np.sum([np.array(All_fixations[i][j].values[1] == "D") for j in range(len(trials[i]))])
                            for i in range(len(subjects))]
        amount_Neutral = [np.sum([np.array(All_fixations[i][j].values[1] == "N") for j in range(len(trials[i]))])
                          for i in range(len(subjects))]
        amount_WS = [np.sum([np.array(All_fixations[i][j].values[1] == "White Space") for j in range(len(trials[i]))])
                     for i in range(len(subjects))]

        self.output_data_dict["amount_of_second_fixations_on_threat"] = amount_Disgusted
        self.output_data_dict["amount_of_second_fixations_on_neutral"] = amount_Neutral
        self.output_data_dict["amount_of_second_fixations_on_WS"] = amount_WS

    def get_average_pupil_size_Disgusted(self):
        # norm_factor = self.get_average_pupil_size_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        mean_Disgusted = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[
                                      (self.fixation_dataset.Subject == subjects[i]) & (
                                              self.fixation_dataset.AOI_Group == "D")]) for i in
                          range(len(subjects))]
        mean_Disgusted = [0 if math.isnan(x) else x for x in mean_Disgusted]
        # norm_disgusted = [mean_Disgusted[i] / float(norm_factor[i]) for i in range(len(mean_Disgusted))]
        # norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]

        self.output_data_dict["average_pupil_size_Disgusted"] = mean_Disgusted
        # self.output_data_dict["normalized_mean_pupil_size_Disgusted"] = norm_disgusted

    def get_average_pupil_size_Neutral(self):
        # norm_factor = self.get_average_pupil_size_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        mean_Neutral = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[
                                    (self.fixation_dataset.Subject == subjects[i]) & (
                                            self.fixation_dataset.AOI_Group == "N")]) for i in
                        range(len(subjects))]
        mean_Neutral = [0 if math.isnan(x) else x for x in mean_Neutral]
        # norm_neutral = [mean_Neutral[i] / float(norm_factor[i]) for i in range(len(mean_Neutral))]
        # norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict["average_pupil_size_Neutral"] = mean_Neutral
#        self.output_data_dict["normalized_mean_pupil_size_Neutral"] = norm_neutral

    def get_average_pupil_size_White_Space(self):
        # norm_factor = self.get_average_pupil_size_All()
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        mean_White_Space = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[
                                        (self.fixation_dataset.Subject == subjects[i]) & (
                                                self.fixation_dataset.AOI_Group == "White Space")]) for i in
                            range(len(subjects))]
        mean_White_Space = [0 if math.isnan(x) else x for x in mean_White_Space]
        # norm_WS = [mean_White_Space[i] / float(norm_factor[i]) for i in range(len(mean_White_Space))]
        # norm_WS = [0 if math.isnan(x) else x for x in norm_WS]

        self.output_data_dict["average_pupil_size_White_Space"] = mean_White_Space
        # self.output_data_dict["normalized_mean_pupil_size_WS"] = norm_WS

    def get_average_pupil_size_All(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        mean_All = [np.mean(self.fixation_dataset.Average_Pupil_Diameter[
                                (self.fixation_dataset.Subject == subjects[i])]) for i in
                    range(len(subjects))]
        mean_All = [0 if math.isnan(x) else x for x in mean_All]

        self.output_data_dict["average_pupil_size_All"] = mean_All
        return mean_All

    def get_STD_pupil_size_Disgusted(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        STD_Disgusted = [np.std(self.fixation_dataset.Average_Pupil_Diameter[
                                    (self.fixation_dataset.Subject == subjects[i]) & (
                                            self.fixation_dataset.AOI_Group == "D")]) for i in
                         range(len(subjects))]
        STD_Disgusted = ["" if math.isnan(x) else x for x in STD_Disgusted]

        self.output_data_dict["STD_pupil_size_Disgusted"] = STD_Disgusted

    def get_STD_pupil_size_Neutral(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        STD_Neutral = [np.std(self.fixation_dataset.Average_Pupil_Diameter[
                                  (self.fixation_dataset.Subject == subjects[i]) & (
                                          self.fixation_dataset.AOI_Group == "N")]) for i in
                       range(len(subjects))]
        STD_Neutral = ["" if math.isnan(x) else x for x in STD_Neutral]

        self.output_data_dict["STD_pupil_size_Neutral"] = STD_Neutral

    def get_STD_pupil_size_White_Space(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        # STD_White_Space = [np.std(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subjects[i]) & (self.fixation_dataset.AOI_Group == "White Space")&(self.fixation_dataset.Average_Pupil_Diameter!='-')])for i in range(len(subjects))]
        STD_White_Space = [np.std(self.fixation_dataset.Average_Pupil_Diameter[
                                      (self.fixation_dataset.Subject == subjects[i]) & (
                                              self.fixation_dataset.AOI_Group == "White Space")]) for i in
                           range(len(subjects))]
        STD_White_Space = ["" if math.isnan(x) else x for x in STD_White_Space]

        self.output_data_dict["STD_pupil_size_White_Space"] = STD_White_Space

    def get_STD_pupil_size_All(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        STD_All = [np.std(self.fixation_dataset.Average_Pupil_Diameter[
                              (self.fixation_dataset.Subject == subjects[i])]) for i in
                   range(len(subjects))]
        STD_All = ["" if math.isnan(x) else x for x in STD_All]
        self.output_data_dict["STD_pupil_size_All"] = STD_All

    def get_mean_different_AOI_per_trial(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique()[:30] for i in subjects]

        mean_AOIs = [np.nanmean([len(set(self.fixation_dataset.Area_of_Interest[
                                             (self.fixation_dataset.Subject == subjects[i]) & (
                                                     self.fixation_dataset.Trial == j)])) for j in trials[i]]) for i
                     in range(len(subjects))]
        mean_AOIs = [0 if math.isnan(x) else x for x in mean_AOIs]

        self.output_data_dict["mean_different_AOI_per_trial"] = mean_AOIs

    def get_difference_between_medians(self):

        subjects = list(sorted(set(self.fixation_dataset.Subject)))
        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique()[:30] for i in subjects]
        medians = [[np.median(self.fixation_dataset.Number[(self.fixation_dataset.Subject == subjects[i]) &
                                                           (self.fixation_dataset.Trial == j)])
                    for j in trials[i]] for i in range(len(subjects))]
        # get sums

        # sum_white_space_first_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
        #                     [(self.fixation_dataset.Subject == subjects[i]) &
        #                     (self.fixation_dataset.Trial == j) & (self.fixation_dataset.AOI_Group == "White Space")
        #                      & (self.fixation_dataset.Number <= medians[i][j])])
        #                     for j in trials[i]]) for i in range(len(subjects))]
        #
        # sum_white_space_second_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
        #                     [(self.fixation_dataset.Subject == subjects[i]) &
        #                     (self.fixation_dataset.Trial == j) & (self.fixation_dataset.AOI_Group == "White Space")
        #                      (self.fixation_dataset.Number > medians[i][j])])
        #                     for j in trials[i]]) for i in range(len(subjects))]

        sum_disgusted_first_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                                                                        [(self.fixation_dataset.Subject == subjects[
                i]) &
                                                                         (self.fixation_dataset.Trial ==
                                                                          list(trials[i])[j]) & (
                                                                                 self.fixation_dataset.AOI_Group == "D")
                                                                         & (self.fixation_dataset.Number <= medians[i][
                j])])
                                                              for j in range(len(trials[i]))]) for i in
                                      range(len(subjects))]

        sum_disgusted_second_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                                                                         [(self.fixation_dataset.Subject == subjects[
                i]) &
                                                                          (self.fixation_dataset.Trial ==
                                                                           list(trials[i])[j]) & (
                                                                                  self.fixation_dataset.AOI_Group == "D")
                                                                          & (self.fixation_dataset.Number > medians[i][
                j])])
                                                               for j in range(len(trials[i]))]) for i in
                                       range(len(subjects))]

        sum_neutral_first_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                                                                      [(self.fixation_dataset.Subject == subjects[i]) &
                                                                       (self.fixation_dataset.Trial == list(trials[i])[
                                                                           j]) & (
                                                                               self.fixation_dataset.AOI_Group == "N")
                                                                       & (self.fixation_dataset.Number <= medians[i][
                j])])
                                                            for j in range(len(trials[i]))]) for i in
                                    range(len(subjects))]

        sum_neutral_second_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                                                                       [(self.fixation_dataset.Subject == subjects[i]) &
                                                                        (self.fixation_dataset.Trial == list(trials[i])[
                                                                            j]) & (
                                                                                self.fixation_dataset.AOI_Group == "N")
                                                                        & (self.fixation_dataset.Number > medians[i][
                j])])
                                                             for j in range(len(trials[i]))]) for i in
                                     range(len(subjects))]

        sum_all_first_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                                                                  [(self.fixation_dataset.Subject == subjects[i]) &
                                                                   (self.fixation_dataset.Trial == list(trials[i])[j])
                                                                   & (self.fixation_dataset.Number <= medians[i][j])])
                                                        for j in range(len(trials[i]))]) for i in range(len(subjects))]

        sum_all_second_median = [self.grouping_function([np.nansum(self.fixation_dataset.Fixation_Duration
                                                                   [(self.fixation_dataset.Subject == subjects[i]) &
                                                                    (self.fixation_dataset.Trial == list(trials[i])[
                                                                        j]) &
                                                                    (self.fixation_dataset.Number > medians[i][j])])
                                                         for j in range(len(trials[i]))]) for i in range(len(subjects))]
        # get norm sums
        sum_all_first_median = [0 if math.isnan(x) else x for x in sum_all_first_median]
        sum_all_second_median = [0 if math.isnan(x) else x for x in sum_all_second_median]

        # sum_white_space_first_median = [0 if math.isnan(x) else x for x in sum_white_space_first_median]
        # norm_WS_first_median =[sum_white_space_first_median[i] / float(sum_all_first_median[i]) for i in range(len(sum_white_space_first_median))]
        # norm_WS_first_median = [0 if math.isnan(x) else x for x in norm_WS_first_median]

        # sum_white_space_second_median = [0 if math.isnan(x) else x for x in sum_white_space_second_median]
        # norm_WS_second_median = [sum_white_space_second_median[i] / float(sum_all_second_median[i]) for i in range(len(sum_white_space_second_median))]
        # norm_WS_second_median = [0 if math.isnan(x) else x for x in norm_WS_second_median]

        sum_disgusted_first_median = [0 if math.isnan(x) else x for x in sum_disgusted_first_median]
        norm_disgusted_first_median = [sum_disgusted_first_median[i] / float(sum_all_first_median[i]) for i in
                                       range(len(sum_disgusted_first_median))]
        norm_disgusted_first_median = [0 if math.isnan(x) else x for x in norm_disgusted_first_median]

        sum_disgusted_second_median = [0 if math.isnan(x) else x for x in sum_disgusted_second_median]
        norm_disgusted_second_median = [sum_disgusted_second_median[i] / float(sum_all_second_median[i]) for i in
                                        range(len(sum_disgusted_second_median))]
        norm_disgusted_second_median = [0 if math.isnan(x) else x for x in norm_disgusted_second_median]

        sum_neutral_first_median = [0 if math.isnan(x) else x for x in sum_neutral_first_median]
        norm_neutral_first_median = [sum_neutral_first_median[i] / float(sum_all_first_median[i]) for i in
                                     range(len(sum_neutral_first_median))]
        norm_neutral_first_median = [0 if math.isnan(x) else x for x in norm_neutral_first_median]

        sum_neutral_second_median = [0 if math.isnan(x) else x for x in sum_neutral_second_median]
        norm_neutral_second_median = [sum_neutral_second_median[i] / float(sum_all_second_median[i]) for i in
                                      range(len(sum_neutral_second_median))]
        norm_neutral_second_median = [0 if math.isnan(x) else x for x in norm_neutral_second_median]

        # get stds
        std_disgusted_first_median = [self.grouping_function([np.std(self.fixation_dataset.Fixation_Duration
                                                                     [(self.fixation_dataset.Subject == subjects[i]) &
                                                                      (self.fixation_dataset.Trial == list(trials[i])[
                                                                          j]) & (self.fixation_dataset.AOI_Group == "D")
                                                                      & (self.fixation_dataset.Number <= medians[i][
                j])])
                                                              for j in range(len(trials[i]))]) for i in
                                      range(len(subjects))]
        std_disgusted_first_median = [0 if math.isnan(x) else x for x in std_disgusted_first_median]

        std_disgusted_second_median = [self.grouping_function([np.std(self.fixation_dataset.Fixation_Duration
                                                                      [(self.fixation_dataset.Subject == subjects[i]) &
                                                                       (self.fixation_dataset.Trial == list(trials[i])[
                                                                           j]) & (
                                                                               self.fixation_dataset.AOI_Group == "D")
                                                                       & (self.fixation_dataset.Number > medians[i][
                j])])
                                                               for j in range(len(trials[i]))]) for i in
                                       range(len(subjects))]
        std_disgusted_second_median = [0 if math.isnan(x) else x for x in std_disgusted_second_median]

        std_neutral_first_median = [self.grouping_function([np.std(self.fixation_dataset.Fixation_Duration
                                                                   [(self.fixation_dataset.Subject == subjects[i]) &
                                                                    (self.fixation_dataset.Trial == list(trials[i])[
                                                                        j]) & (self.fixation_dataset.AOI_Group == "N")
                                                                    & (self.fixation_dataset.Number <= medians[i][j])])
                                                            for j in range(len(trials[i]))]) for i in
                                    range(len(subjects))]
        std_neutral_first_median = [0 if math.isnan(x) else x for x in std_neutral_first_median]

        std_neutral_second_median = [self.grouping_function([np.std(self.fixation_dataset.Fixation_Duration
                                                                    [(self.fixation_dataset.Subject == subjects[i]) &
                                                                     (self.fixation_dataset.Trial == list(trials[i])[
                                                                         j]) & (self.fixation_dataset.AOI_Group == "N")
                                                                     & (self.fixation_dataset.Number > medians[i][j])])
                                                             for j in range(len(trials[i]))]) for i in
                                     range(len(subjects))]
        std_neutral_second_median = [0 if math.isnan(x) else x for x in std_neutral_second_median]

        ratio_std_disgusted = [
            (std_disgusted_first_median[i] / float(std_disgusted_second_median[i])) if std_disgusted_second_median[
                                                                                           i] > 0 else "" for i in
            range(len(std_disgusted_first_median))]
        self.output_data_dict["STD_Disgusted_difference_between_medians"] = ratio_std_disgusted

        ratio_std_neutral = [
            (std_neutral_first_median[i] / float(std_neutral_second_median[i])) if std_neutral_second_median[
                                                                                       i] > 0 else "" for i in
            range(len(std_neutral_first_median))]
        self.output_data_dict["STD_Neutral_difference_between_medians"] = ratio_std_neutral

        ratio_sum_disgusted = [
            (sum_disgusted_first_median[i] / float(sum_disgusted_second_median[i])) if sum_disgusted_second_median[
                                                                                           i] > 0 else "" for i in
            range(len(sum_disgusted_first_median))]
        self.output_data_dict["sum_disgusted_difference_between_medians"] = ratio_sum_disgusted

        ratio_norm_sum_disgustd = [
            (norm_disgusted_first_median[i] / float(norm_disgusted_second_median[i])) if norm_disgusted_second_median[
                                                                                             i] > 0 else "" for i in
            range(len(norm_disgusted_first_median))]
        self.output_data_dict["norm_sum_disgusted_difference_between_medians"] = ratio_norm_sum_disgustd

        ratio_sum_neutral = [
            (sum_neutral_first_median[i] / float(sum_neutral_second_median[i])) if std_neutral_second_median[
                                                                                       i] > 0 else "" for i in
            range(len(sum_neutral_first_median))]
        self.output_data_dict["sum_neutral_difference_between_medians"] = ratio_sum_neutral

        ratio_norm_sum_neutral = [
            (norm_neutral_first_median[i] / float(norm_neutral_second_median[i])) if norm_neutral_second_median[
                                                                                         i] > 0 else "" for i in
            range(len(norm_neutral_first_median))]
        self.output_data_dict["norm_sum_neutral_difference_between_medians"] = ratio_norm_sum_neutral

        ratio_sum_all = [
            (sum_all_first_median[i] / float(sum_all_second_median[i])) if sum_all_second_median[i] > 0 else "" for i in
            range(len(sum_all_first_median))]
        self.output_data_dict["sum_all_difference_between_medians"] = ratio_sum_all


    def calc_increamental_mean(self, probability_distribution_value, p_disgusted, learning_rate):
        increamental_mean = probability_distribution_value + learning_rate * (
                p_disgusted - probability_distribution_value)
        return increamental_mean


    def learning_rate_iter(self):
        i = 0
        while True:
            i += 1
            yield i ** (-1)


    def find_area(self, x_pos, y_pos):
        areas_map = [((1072, 759), (1279, 966)), ((851, 761), (1058, 968)),
                     ((623, 758), (830, 965)),
                     ((400, 758), (607, 965)),
                     ((1071, 535), (1278, 742)), ((850, 535), (1057, 742)),
                     ((622, 534), (829, 741)),
                     ((399, 534), (606, 741)),
                     ((1073, 308), (1280, 515)), ((848, 310), (1055, 517)),
                     ((624, 307), (831, 514)),
                     ((401, 307), (608, 514)),
                     ((1072, 84), (1279, 291)), ((847, 84), (1054, 291)),
                     ((623, 83), (830, 290)),
                     ((400, 83), (607, 290))]
        for index, elem in enumerate(areas_map):
            top = areas_map[index][1]
            bottom = areas_map[index][0]
            if bottom[0] <= x_pos <= top[0] and top[1] >= bottom[1] <= y_pos:
                return elem


    def get_probability_distribution(self, trial, subject, last_probability_distribution=None):
        num_of_aois = 16
        areas_map = [((1072, 759), (1279, 966)), ((851, 761), (1058, 968)), ((623, 758), (830, 965)),
                     ((400, 758), (607, 965)),
                     ((1071, 535), (1278, 742)), ((850, 535), (1057, 742)), ((622, 534), (829, 741)),
                     ((399, 534), (606, 741)),
                     ((1073, 308), (1280, 515)), ((848, 310), (1055, 517)), ((624, 307), (831, 514)),
                     ((401, 307), (608, 514)),
                     ((1072, 84), (1279, 291)), ((847, 84), (1054, 291)), ((623, 83), (830, 290)),
                     ((400, 83), (607, 290))]
        # get all watched aoi's in the last trial
        if trial == 0:
            return {areas_map[i]: [0.5, self.learning_rate_iter()] for i in range(num_of_aois)}
        else:
            trial_aois_tuple = (self.fixation_dataset.AOI_Group[(self.fixation_dataset.Subject == subject) & (
                    self.fixation_dataset.Trial == trial)],
                                self.fixation_dataset.Area_of_Interest[(self.fixation_dataset.Subject == subject) & (
                                        self.fixation_dataset.Trial == trial)])

            # update the probability distribution map
            # by the increamental_mean rule
            for i in range(num_of_aois):
                if 'AOI {}'.format(i + 1) in trial_aois_tuple[1].values:
                    p_disgusted = float((trial_aois_tuple[0].values[trial_aois_tuple[1].values ==
                                                                    'AOI {}'.format(i + 1)] == 'D')[0])
                    positions = (self.fixation_dataset.Position_X[(self.fixation_dataset.Subject == subject) & (
                            self.fixation_dataset.Trial == trial) & (
                                                                              self.fixation_dataset.Area_of_Interest == 'AOI {}'.format(
                                                                          i + 1))],
                                 self.fixation_dataset.Position_Y[(self.fixation_dataset.Subject == subject) & (
                                         self.fixation_dataset.Trial == trial) & (
                                                                              self.fixation_dataset.Area_of_Interest == 'AOI {}'.format(
                                                                          i + 1))])
                    area = self.find_area(positions[0].values[0], positions[1].values[0])
                    last_probability_distribution[area][0] = \
                        self.calc_increamental_mean(last_probability_distribution[area][0],
                                                    p_disgusted,
                                                    next(last_probability_distribution[area][1]))
            return last_probability_distribution


    def get_p_disgusted_times_first_fixation_duration(self):
        subjects = list(sorted(set(self.fixation_dataset.Subject)))

        trials = [(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]).unique() for i in subjects]
        all_fixations_aoi = [[self.fixation_dataset.Area_of_Interest[(self.fixation_dataset.Subject == subjects[i]) &
                                                                     (self.fixation_dataset.Trial == j)] for j in
                              trials[i]] for i in range(len(subjects))]
        all_fixations_durations = [
            [self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subjects[i]) &
                                                     (self.fixation_dataset.Trial == j)] for j in trials[i]] for i in
            range(len(subjects))]
        p_disgusted_times_first_fixation_duration = []
        for i, subject in enumerate(subjects):
            p_disgusted_times_first_fixation_duration.append([])
            p_map = self.get_probability_distribution(0, subject)
            for j, trial in enumerate(trials[i]):
                fixation_index = 0
                first_aoi = all_fixations_aoi[i][j].values[fixation_index]
                first_fixation_duration = all_fixations_durations[i][j].values[fixation_index]
                while first_aoi == 'White Space':
                    fixation_index += 1
                    first_aoi = all_fixations_aoi[i][j].values[fixation_index]
                    first_fixation_duration = all_fixations_durations[i][j].values[fixation_index]

                positions = (self.fixation_dataset.Position_X[(self.fixation_dataset.Subject == subject) &
                                                              (self.fixation_dataset.Trial == trial) & (
                                                                          self.fixation_dataset.Area_of_Interest == first_aoi)],
                             self.fixation_dataset.Position_Y[(self.fixation_dataset.Subject == subject) &
                                                              (self.fixation_dataset.Trial == trial) & (
                                                                          self.fixation_dataset.Area_of_Interest == first_aoi)])
                area = self.find_area(positions[0].values[0], positions[1].values[0])
                p_disgusted_times_first_fixation_duration[i].append(p_map[area][0] * first_fixation_duration)

                p_map = self.get_probability_distribution(trial, subject, last_probability_distribution=p_map)

        self.output_data_dict["p_disgusted_times_first_fixation_duration"] = \
            [np.mean(p_disgusted_times_first_fixation_duration[i]) for i in range(len(subjects))]


    def get_all_good_features(self):
        self.get_subject_number()
        self.get_lsas()
        self.get_group()
        self.get_Amits_feature()
        self.get_sum_fixation_length_Disgusted()
        self.get_sum_fixation_length_Neutral()
        self.get_sum_fixation_length_White_Space()
        self.get_average_fixation_length_Disgusted()
        self.get_average_fixation_length_Neutral()
        self.get_average_fixation_length_White_Space()
        self.get_amount_fixation_Disgusted()
        self.get_amount_fixation_Neutral()
        self.get_amount_fixation_White_Space()
        self.get_STD_fixation_length_Disgusted()
        self.get_STD_fixation_length_Neutral()
        self.get_STD_fixation_length_White_Space()
        self.get_STD_fixation_length_All()
        self.get_ratio_D_DN()
        self.get_ratio_N_DN()
        self.get_ratio_WS_All()
        self.get_ratio_D_DN_2()
        self.get_ratio_N_DN_2()
        self.get_amount_DN_transitions()
        self.get_amount_ND_transitions()
        self.get_amount_DD_transitions()
        self.get_amount_NN_transitions()
        self.get_amount_diff_AOI_transitions()
        self.var_threat_precentage_between_trials()
        self.get_average_pupil_size_Disgusted()
        self.get_average_pupil_size_Neutral()
        self.get_average_pupil_size_White_Space()
        self.get_average_pupil_size_All()
        self.get_STD_pupil_size_Disgusted()
        self.get_STD_pupil_size_Neutral()
        self.get_STD_pupil_size_White_Space()
        self.get_STD_pupil_size_All()
        self.get_mean_different_AOI_per_trial()
        self.get_difference_between_medians()




    def get_matrix_count_independant_features(self):

        self.get_subject_number()
        self.get_lsas()
        self.get_group()

        self.get_sum_fixation_length_Disgusted()
        self.get_sum_fixation_length_Neutral()
        self.get_sum_fixation_length_White_Space()

        self.get_average_fixation_length_Disgusted()
        self.get_average_fixation_length_Neutral()
        self.get_average_fixation_length_White_Space()

        self.get_amount_fixation_Disgusted()
        self.get_amount_fixation_Neutral()
        self.get_amount_fixation_White_Space()

        self.get_STD_fixation_length_Disgusted()
        self.get_STD_fixation_length_Neutral()
        self.get_STD_fixation_length_White_Space()
        self.get_STD_fixation_length_All()

        self.get_ratio_D_DN()
        self.get_ratio_N_DN()

        self.var_threat_precentage_between_trials()

        self.get_average_pupil_size_Disgusted()
        self.get_average_pupil_size_Neutral()
        self.get_average_pupil_size_White_Space()
        self.get_average_pupil_size_All()

        self.get_STD_pupil_size_Disgusted()
        self.get_STD_pupil_size_Neutral()
        self.get_STD_pupil_size_White_Space()
        self.get_STD_pupil_size_All()

        self.get_mean_different_AOI_per_trial()



    def get_features_for_prediction(self):
        self.get_subject_number()

        self.get_sum_fixation_length_Disgusted()
        self.get_sum_fixation_length_Neutral()
        self.get_sum_fixation_length_White_Space()

        self.get_average_fixation_length_Disgusted()
        self.get_average_fixation_length_Neutral()
        self.get_average_fixation_length_White_Space()

        self.get_amount_fixation_Disgusted()
        self.get_amount_fixation_Neutral()
        self.get_amount_fixation_White_Space()

        self.get_STD_fixation_length_Disgusted()
        self.get_STD_fixation_length_Neutral()
        self.get_STD_fixation_length_White_Space()
        self.get_STD_fixation_length_All()

        self.get_ratio_D_DN()
        self.get_ratio_N_DN()

        self.var_threat_precentage_between_trials()

        self.get_average_pupil_size_Disgusted()
        self.get_average_pupil_size_Neutral()
        self.get_average_pupil_size_White_Space()
        self.get_average_pupil_size_All()

        self.get_STD_pupil_size_Disgusted()
        self.get_STD_pupil_size_Neutral()
        self.get_STD_pupil_size_White_Space()
        self.get_STD_pupil_size_All()

        self.get_mean_different_AOI_per_trial()


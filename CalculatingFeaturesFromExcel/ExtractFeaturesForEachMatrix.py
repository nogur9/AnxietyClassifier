import itertools
import numpy as np
import math
from DataImporting import DataFromExcel

Fixation_length_cutoff = 100


class Data:
    grouping_function = None
    fixation_dataset = None
    output_data_frame = None
    output_data_dict = {}
    demographic_dataset = None
    trials_subject = []
    trials = []
    subjects = []

    def __init__(self, path, fixation_dataset_sheet_name, demographic_dataset_sheet_name=None):

        self.fixation_dataset = DataFromExcel.get_data(path, fixation_dataset_sheet_name)
        self.fixation_dataset = self.fixation_dataset[self.fixation_dataset.Fixation_Duration > Fixation_length_cutoff]

        if not demographic_dataset_sheet_name is None:
            self.demographic_dataset = DataFromExcel.get_data(path, demographic_dataset_sheet_name)

        self.Trials_count = np.array([len(set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == j])) for j in
                                      sorted(set(self.fixation_dataset.Subject))])

        self.subjects = list(sorted(set(self.fixation_dataset.Subject)))

        self.trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in self.subjects]

        for i in range(len(self.subjects)):
            for matrix in self.trials[i]:
                self.trials_subject.append([self.subjects[i], matrix])

    def get_age(self):
        Age = [self.demographic_dataset.Age[i] for i in range(len(self.demographic_dataset.Age))]
        self.output_data_dict["Age"] = Age

    def get_PHQ9(self):
        PHQ9 = [self.demographic_dataset.PHQ[i] for i in range(len(self.demographic_dataset.PHQ))]
        self.output_data_dict["PHQ9"] = PHQ9

    def get_subject_number(self):
        subject_number = list(sorted(set(self.fixation_dataset.Subject)))
        self.output_data_dict["Subject_Number"] = subject_number

    def get_trial(self):
        trial_number = list(sorted(set(self.fixation_dataset.Trial)))
        self.output_data_dict["Trial"] = trial_number

    def get_group(self):
        group = [self.demographic_dataset.group[i] for i in range(len(self.demographic_dataset.group))]
        self.output_data_dict["group"] = group

    # feature extraction - features need to be computed

    def get_sum_fixation_length_Disgusted(self):

        norm_factor = self.get_sum_fixation_length_All()

        sum_disgusted = [np.nansum(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)
                    & (self.fixation_dataset.AOI_Group == "D")]) for [subject,matrix] in self.trials_subject]

        sum_disgusted = [0 if math.isnan(x) else x for x in sum_disgusted]
        norm_disgusted = [sum_disgusted[i] / float(norm_factor[i]) for i in range(len(sum_disgusted))]
        norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]
        self.output_data_dict["sum_fixation_length_Disgusted"] = sum_disgusted
        self.output_data_dict["normalized_sum_fixation_length_Disgusted"] = norm_disgusted

    def get_sum_fixation_length_Neutral(self):

        norm_factor = self.get_sum_fixation_length_All()

        sum_neutral = [np.nansum(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)
                    & (self.fixation_dataset.AOI_Group == "N")]) for [subject,matrix] in self.trials_subject]

        sum_neutral = [0 if math.isnan(x) else x for x in sum_neutral]
        norm_neutral = [sum_neutral[i] / float(norm_factor[i]) for i in range(len(sum_neutral))]
        norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict["sum_fixation_length_Neutral"] = sum_neutral
        self.output_data_dict["normalized_sum_fixation_length_Neutral"] = norm_neutral

    def get_sum_fixation_length_White_Space(self):

        norm_factor = self.get_sum_fixation_length_All()
        sum_white_space = [np.nansum(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)
                    & (self.fixation_dataset.AOI_Group == "White Space")]) for [subject,matrix] in self.trials_subject]

        sum_white_space = [0 if math.isnan(x) else x for x in sum_white_space]
        norm_WS = [sum_white_space[i] / float(norm_factor[i]) for i in range(len(sum_white_space))]
        norm_WS = [0 if math.isnan(x) else x for x in norm_WS]
        self.output_data_dict["sum_fixation_length_White_Space"] = sum_white_space

        self.output_data_dict["normalized_sum_fixation_length_WS"] = norm_WS

    def get_sum_fixation_length_All(self):

        sum_all = [np.nansum(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)]) for [subject,matrix] in self.trials_subject]

        sum_all = [0 if math.isnan(x) else x for x in sum_all]

        self.output_data_dict["sum_fixation_length_All"] = sum_all
        return sum_all

    def get_average_fixation_length_Disgusted(self):

        norm_factor = self.get_average_fixation_length_All()
        mean_disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)
                    & (self.fixation_dataset.AOI_Group == "D")]) for [subject,matrix] in self.trials_subject]

        mean_disgusted = [0 if math.isnan(x) else x for x in mean_disgusted]
        norm_disgusted = [mean_disgusted[i] / float(norm_factor[i]) for i in range(len(mean_disgusted))]
        norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]

        self.output_data_dict["average_fixation_length_Disgusted"] = mean_disgusted
        self.output_data_dict["normalized_mean_fixation_length_Disgusted"] = norm_disgusted

    def get_average_fixation_length_Neutral(self):

        norm_factor = self.get_average_fixation_length_All()
        mean_neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)
                    & (self.fixation_dataset.AOI_Group == "N")]) for [subject,matrix] in self.trials_subject]

        mean_neutral = [0 if math.isnan(x) else x for x in mean_neutral]
        norm_neutral = [mean_neutral[i] / float(norm_factor[i]) for i in range(len(mean_neutral))]
        norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict["average_fixation_length_Neutral"] = mean_neutral
        self.output_data_dict["normalized_mean_fixation_length_Neutral"] = norm_neutral

    def get_average_fixation_length_White_Space(self):

        norm_factor = self.get_average_fixation_length_All()
        mean_white_space = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)
                    & (self.fixation_dataset.AOI_Group == "White Space")]) for [subject,matrix] in self.trials_subject]

        mean_white_space = [0 if math.isnan(x) else x for x in mean_white_space]
        norm_WS = [mean_white_space[i] / float(norm_factor[i]) for i in range(len(mean_white_space))]
        norm_WS = [0 if math.isnan(x) else x for x in norm_WS]

        self.output_data_dict["average_fixation_length_White_Space"] = mean_white_space
        self.output_data_dict["normalized_mean_fixation_length_WS"] = norm_WS

    def get_average_fixation_length_All(self):

        mean_all = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)]) for [subject,matrix] in self.trials_subject]

        mean_all = [0 if math.isnan(x) else x for x in mean_all]
        self.output_data_dict["average_fixation_length_All"] = mean_all
        return mean_all

    def get_amount_fixation_Disgusted(self):

        norm_factor = self.get_amount_fixation_All()

        amount_disgusted = [len(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)
                    & (self.fixation_dataset.AOI_Group == "D")]) for [subject,matrix] in self.trials_subject]

        amount_disgusted = [0 if math.isnan(x) else x for x in amount_disgusted]
        norm_disgusted = [amount_disgusted[i] / float(norm_factor[i]) for i in range(len(amount_disgusted))]
        norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]

        self.output_data_dict["amount_fixation_Disgusted"] = amount_disgusted
        self.output_data_dict["normalized_amount_fixation_Disgusted"] = norm_disgusted

    def get_amount_fixation_Neutral(self):
        norm_factor = self.get_amount_fixation_All()
        amount_neutral = [len(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)
                    & (self.fixation_dataset.AOI_Group == "N")]) for [subject,matrix] in self.trials_subject]
        amount_neutral = [0 if math.isnan(x) else x for x in amount_neutral]
        norm_neutral = [amount_neutral[i] / float(norm_factor[i]) for i in range(len(amount_neutral))]
        norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict["amount_fixation_Neutral"] = amount_neutral
        self.output_data_dict["normalized_amount_fixation_Neutral"] = norm_neutral

    def get_amount_fixation_White_Space(self):
        norm_factor = self.get_amount_fixation_All()

        amount_white_space = [len(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)
                    & (self.fixation_dataset.AOI_Group == "White Space")]) for [subject,matrix] in self.trials_subject]

        amount_white_space = [0 if math.isnan(x) else x for x in amount_white_space]
        norm_WS = [amount_white_space[i] / float(norm_factor[i]) for i in range(len(amount_white_space))]
        norm_WS = [0 if math.isnan(x) else x for x in norm_WS]

        self.output_data_dict["amount_fixation_White_Space"] = amount_white_space
        self.output_data_dict["normalized_amount_fixation_WS"] = norm_WS

    def get_amount_fixation_All(self):

        amount_all = [len(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                     & (self.fixation_dataset.Trial == matrix)])
                              for [subject, matrix] in self.trials_subject]

        amount_all = [0 if math.isnan(x) else x for x in amount_all]

        self.output_data_dict["amount_fixation_All"] = amount_all
        return amount_all

    def get_STD_fixation_length_Disgusted(self):

        std_disgusted = [np.nanstd(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                    & (self.fixation_dataset.Trial == matrix)
                    & (self.fixation_dataset.AOI_Group == "D")]) for [subject,matrix] in self.trials_subject]

        std_disgusted = [0 if math.isnan(x) else x for x in std_disgusted]

        self.output_data_dict["STD_fixation_length_Disgusted"] = std_disgusted

    def get_STD_fixation_length_Neutral(self):

        std_neutral = [np.nanstd(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "N")]) for
                         [subject, matrix] in self.trials_subject]

        std_neutral = [0 if math.isnan(x) else x for x in std_neutral]

        self.output_data_dict["STD_fixation_length_Neutral"] = std_neutral

    def get_STD_fixation_length_White_Space(self):

        std_white_space = [np.nanstd(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                           & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "White Space")])
                           for [subject, matrix] in self.trials_subject]

        std_white_space = [0 if math.isnan(x) else x for x in std_white_space]

        self.output_data_dict["STD_fixation_length_White_Space"] = std_white_space

    def get_STD_fixation_length_All(self):

        std_all = [np.nanstd(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                           & (self.fixation_dataset.Trial == matrix)])
                           for [subject, matrix] in self.trials_subject]
        std_all = [0 if math.isnan(x) else x for x in std_all]

        self.output_data_dict["STD_fixation_length_All"] = std_all

    def get_ratio_D_DN(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """

        mean_disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "D")]) for
                         [subject, matrix] in self.trials_subject]

        mean_neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "N")]) for
                         [subject, matrix] in self.trials_subject]

        ratio = [mean_disgusted[i] / float(mean_neutral[i] + mean_disgusted[i]) for i in range(len(mean_disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["Ratio D/D+N"] = ratio

    def get_ratio_N_DN(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """

        mean_disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "D")]) for
                         [subject, matrix] in self.trials_subject]

        mean_neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "N")]) for
                         [subject, matrix] in self.trials_subject]

        ratio = [mean_neutral[i] / float(mean_neutral[i] + mean_disgusted[i]) for i in range(len(mean_disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["Ratio N/D+N"] = ratio

    def get_ratio_WS_All(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """
        mean_disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "D")]) for
                         [subject, matrix] in self.trials_subject]

        mean_neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "N")]) for
                         [subject, matrix] in self.trials_subject]

        mean_WS = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "White Space")]) for
                         [subject, matrix] in self.trials_subject]

        ratio = [mean_WS[i] / float(mean_WS[i] + mean_neutral[i] + mean_disgusted[i]) for i in
                 range(len(mean_disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["Ratio WS/WS+N+D"] = ratio

    def get_ratio_D_DN_2(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """

        mean_disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "D")]) for
                         [subject, matrix] in self.trials_subject]

        mean_neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "N")]) for
                         [subject, matrix] in self.trials_subject]

        mean_WS = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "White Space")]) for
                         [subject, matrix] in self.trials_subject]

        ratio = [mean_disgusted[i] / float(mean_WS[i] + mean_neutral[i] + mean_disgusted[i]) for i in
                 range(len(mean_disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["%threat - #2"] = ratio

    def get_ratio_N_DN_2(self):
        """

        :return: the ratio of the sum fixation length of disgusted and neutral fixations
        """

        mean_disgusted = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "D")]) for
                         [subject, matrix] in self.trials_subject]

        mean_neutral = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "N")]) for
                         [subject, matrix] in self.trials_subject]

        mean_WS = [np.nanmean(self.fixation_dataset.Fixation_Duration[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "White Space")]) for
                         [subject, matrix] in self.trials_subject]

        ratio = [mean_neutral[i] / float(mean_WS[i] + mean_neutral[i] + mean_disgusted[i]) for i in
                 range(len(mean_disgusted))]

        ratio = [0 if math.isnan(x) else x for x in ratio]
        self.output_data_dict["%neutral - #2"] = ratio

    def get_average_pupil_size_Disgusted(self):

        norm_factor = self.get_average_pupil_size_All()
        mean_disgusted = [np.nanmean(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "D")]) for
                         [subject, matrix] in self.trials_subject]

        mean_disgusted = [0 if math.isnan(x) else x for x in mean_disgusted]
        norm_disgusted = [mean_disgusted[i] / float(norm_factor[i]) for i in range(len(mean_disgusted))]
        norm_disgusted = [0 if math.isnan(x) else x for x in norm_disgusted]

        self.output_data_dict["average_pupil_size_Disgusted"] = mean_disgusted
        self.output_data_dict["normalized_mean_pupil_size_Disgusted"] = norm_disgusted

    def get_average_pupil_size_Neutral(self):

        norm_factor = self.get_average_pupil_size_All()
        mean_neutral = [np.nanmean(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "N")]) for
                         [subject, matrix] in self.trials_subject]

        mean_neutral = [0 if math.isnan(x) else x for x in mean_neutral]
        norm_neutral = [mean_neutral[i] / float(norm_factor[i]) for i in range(len(mean_neutral))]
        norm_neutral = [0 if math.isnan(x) else x for x in norm_neutral]

        self.output_data_dict["average_pupil_size_Neutral"] = mean_neutral
        self.output_data_dict["normalized_mean_pupil_size_Neutral"] = norm_neutral

    def get_average_pupil_size_White_Space(self):

        norm_factor = self.get_average_pupil_size_All()

        mean_white_space = [np.nanmean(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "White Space")]) for
                         [subject, matrix] in self.trials_subject]
        mean_white_space = [0 if math.isnan(x) else x for x in mean_white_space]
        norm_WS = [mean_white_space[i] / float(norm_factor[i]) for i in range(len(mean_white_space))]
        norm_WS = [0 if math.isnan(x) else x for x in norm_WS]

        self.output_data_dict["average_pupil_size_White_Space"] = mean_white_space
        self.output_data_dict["normalized_mean_pupil_size_WS"] = norm_WS

    def get_average_pupil_size_All(self):

        mean_all = [np.nanmean(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix)]) for
                         [subject, matrix] in self.trials_subject]
        mean_all = [0 if math.isnan(x) else x for x in mean_all]

        self.output_data_dict["average_pupil_size_All"] = mean_all
        return mean_all

    def get_STD_pupil_size_Disgusted(self):
        std_disgusted = [np.nanstd(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "D")]) for
                         [subject, matrix] in self.trials_subject]
        std_disgusted = [0 if math.isnan(x) else x for x in std_disgusted]

        self.output_data_dict["STD_pupil_size_Disgusted"] = std_disgusted

    def get_STD_pupil_size_Neutral(self):
        std_neutral = [np.nanstd(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "N")]) for
                         [subject, matrix] in self.trials_subject]
        std_neutral = [0 if math.isnan(x) else x for x in std_neutral]

        self.output_data_dict["STD_pupil_size_Neutral"] = std_neutral

    def get_STD_pupil_size_White_Space(self):

        std_white_space = [np.nanstd(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix) & (self.fixation_dataset.AOI_Group == "White Space")]) for
                         [subject, matrix] in self.trials_subject]
        std_white_space = [0 if math.isnan(x) else x for x in std_white_space]

        self.output_data_dict["STD_pupil_size_White_Space"] = std_white_space

    def get_STD_pupil_size_All(self):

        std_all = [np.nanstd(self.fixation_dataset.Average_Pupil_Diameter[(self.fixation_dataset.Subject == subject)
                        & (self.fixation_dataset.Trial == matrix)]) for
                         [subject, matrix] in self.trials_subject]
        std_all = [0 if math.isnan(x) else x for x in std_all]
        self.output_data_dict["STD_pupil_size_All"] = std_all

    def get_difference_between_medians(self):

        medians = [np.median(self.fixation_dataset.Number[(self.fixation_dataset.Subject == subject) &
                                (self.fixation_dataset.Trial == matrix)]) for [subject, matrix] in self.trials_subject]
        # get sums
        sum_white_space_first_median = [np.nansum(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.AOI_Group == "White Space") &
                                                    (self.fixation_dataset.Number <= medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]

        sum_white_space_second_median = [np.nansum(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.AOI_Group == "White Space") &
                                                    (self.fixation_dataset.Number > medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]

        sum_disgusted_first_median = [np.nansum(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.AOI_Group == "D") &
                                                    (self.fixation_dataset.Number <= medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]
        sum_disgusted_second_median = [np.nansum(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.AOI_Group == "D") &
                                                    (self.fixation_dataset.Number > medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]

        sum_neutral_first_median = [np.nansum(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.AOI_Group == "N") &
                                                    (self.fixation_dataset.Number <= medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]
        sum_neutral_second_median = [np.nansum(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.AOI_Group == "N") &
                                                    (self.fixation_dataset.Number > medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]
        sum_all_first_median = [np.nansum(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.Number <= medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]

        sum_all_second_median = [np.nansum(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.Number > medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]
        # get norm sums
        sum_all_first_median = [0 if math.isnan(x) else x for x in sum_all_first_median]
        sum_all_second_median = [0 if math.isnan(x) else x for x in sum_all_second_median]

        sum_white_space_first_median = [0 if math.isnan(x) else x for x in sum_white_space_first_median]
        norm_WS_first_median = [sum_white_space_first_median[i] / float(sum_all_first_median[i]) for i in
                                range(len(sum_white_space_first_median))]
        norm_WS_first_median = [0 if math.isnan(x) else x for x in norm_WS_first_median]

        sum_white_space_second_median = [0 if math.isnan(x) else x for x in sum_white_space_second_median]
        norm_WS_second_median = [sum_white_space_second_median[i] / float(sum_all_second_median[i]) for i in
                                 range(len(sum_white_space_second_median))]
        norm_WS_second_median = [0 if math.isnan(x) else x for x in norm_WS_second_median]

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

        std_disgusted_first_median = [np.nanstd(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.AOI_Group == "D") &
                                                    (self.fixation_dataset.Number <= medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]
        std_disgusted_first_median = [0 if math.isnan(x) else x for x in std_disgusted_first_median]

        std_disgusted_second_median =[np.nanstd(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.AOI_Group == "D") &
                                                    (self.fixation_dataset.Number > medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]
        std_disgusted_second_median = [0 if math.isnan(x) else x for x in std_disgusted_second_median]

        std_neutral_first_median = [np.nansum(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.AOI_Group == "N") &
                                                    (self.fixation_dataset.Number <= medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]
        std_neutral_first_median = [0 if math.isnan(x) else x for x in std_neutral_first_median]

        std_neutral_second_median = [np.nanstd(self.fixation_dataset.Fixation_Duration
                                                    [(self.fixation_dataset.Subject == subject) &
                                                    (self.fixation_dataset.Trial == matrix) &
                                                    (self.fixation_dataset.AOI_Group == "N") &
                                                    (self.fixation_dataset.Number > medians[i])])
                                                    for i, [subject, matrix] in enumerate(self.trials_subject)]
        std_neutral_second_median = [0 if math.isnan(x) else x for x in std_neutral_second_median]

        self.output_data_dict["STD_Disgusted_difference_between_medians"] = (
                    std_disgusted_first_median / float(std_disgusted_second_median))
        self.output_data_dict["STD_Neutral_difference_between_medians"] = (
                    std_neutral_first_median / float(std_neutral_second_median))
        self.output_data_dict["sum_disgusted_difference_between_medians"] = (
                    sum_disgusted_first_median / float(sum_disgusted_second_median))
        self.output_data_dict["norm_sum_disgusted_difference_between_medians"] = (
                    norm_disgusted_first_median / float(norm_disgusted_second_median))
        self.output_data_dict["sum_neutral_difference_between_medians"] = (
                    sum_neutral_first_median / float(sum_neutral_second_median))
        self.output_data_dict["norm_sum_disgusted_difference_between_medians"] = (
                    norm_neutral_first_median / float(norm_neutral_second_median))
        self.output_data_dict["sum_WS_difference_between_medians"] = (
                    sum_white_space_first_median / float(sum_white_space_second_median))
        self.output_data_dict["norm_sum_WS_difference_between_medians"] = (
                    norm_WS_first_median / float(norm_WS_second_median))
        self.output_data_dict["sum_all_difference_between_medians"] = (
                    sum_all_first_median / float(sum_all_second_median))

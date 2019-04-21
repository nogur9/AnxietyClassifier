import numpy as np
import pandas as pd
import datetime
import xlsxwriter

omers_maps_path = "../OmersData/Omers_map.xlsx"
gals_map_path = "../OmersData/Gals_map.xlsx"
maps = pd.read_excel(gals_map_path)
Fixation_length_cutoff = 100

def get_AOI_group(row):
    map_dict = {1: "N", 2: "D", "NE": "N", "DI": "D"}

    CURRENT_FIX_INTEREST_AREAS, imagefile = row.split('|')
    CURRENT_FIX_INTEREST_AREAS = eval(CURRENT_FIX_INTEREST_AREAS)
    print(CURRENT_FIX_INTEREST_AREAS)
    print(imagefile)

    imagefile = imagefile.replace("_", " ")
    WS = "White Space"
    if CURRENT_FIX_INTEREST_AREAS == []:
        return WS
    else:
        aoi = maps[maps['SlideImage'] == imagefile]['Cell' + str(CURRENT_FIX_INTEREST_AREAS[0])]
        aoi = aoi.values[0]
        print(map_dict[aoi])
        return map_dict[aoi]

class EyeLinkData:

    smi_to_eye_link_direct_tranform = {

        'Stimulus': 'imagefile',
        'Fixation_Duration': 'CURRENT_FIX_DURATION',
        'Position_X': 'CURRENT_FIX_X',
        'Position_Y': 'CURRENT_FIX_Y',
        'Average_Pupil_Diameter': 'CURRENT_FIX_PUPIL',
        'Number': 'CURRENT_FIX_INDEX'}

    def __init__(self, data_path, subj=None, block=None):
        data_path = data_path
        self.df = pd.read_excel(data_path)
        print ("in init")
        # removing first fixations
        self.df = self.df[self.df['CURRENT_FIX_INDEX'] != 1]
        self.df = self.df[self.df['CURRENT_FIX_DURATION'] > Fixation_length_cutoff]
        self.subj = subj
        self.block = block


    def transform_data(self, output_path=None):
        # init
        output_path = "C:\‏‏PycharmProjects\AnxietyClassifier\\100_training_set\eyelink_proccessor_output\gals training set {}.xlsx".format(datetime.datetime.now().strftime('%Y-%m-%d'))
        output_df = pd.DataFrame()

        # direct transformation columns
        for col in self.smi_to_eye_link_direct_tranform:
            output_df[col] = self.df[self.smi_to_eye_link_direct_tranform[col]]

        # creating the AOI_group column
        tmp = self.df['CURRENT_FIX_INTEREST_AREAS'] + '|' + self.df['imagefile']
        output_df['AOI_Group'] = tmp.apply(get_AOI_group)

        # create AOI column
        output_df['Area_of_Interest'] = self.df['CURRENT_FIX_INTEREST_AREAS'].apply(
            lambda x: 'White Space' if len(x) == 3 else 'AOI {}'.format(eval(x)[0]))

        # get subject number
        if self.subj is not None:
            output_df['Subject'] = self.subj
        else:
            output_df['Subject'] = self.df['RECORDING_SESSION_LABEL'].astype('str').str.extract(r"^([0-9]+)")

        # get Trial number
        if self.block is not None:
            #self.df['block_num'] = self.block
            output_df['Trial'] = self.df['identifier']
        else:
            self.df['block_num'] = pd.to_numeric(self.df['RECORDING_SESSION_LABEL'].str.extract(r"([0-9]+)$"))
            first_block_num = min(self.df['block_num'].unique())
            output_df['Trial'] = self.df['identifier'] + 30*(self.df['block_num'] - first_block_num)

        # writing the output Data/Frame
        excel_writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

        demographic_df = pd.DataFrame(output_df['Subject'].unique(), columns=['Subject'])
        demographic_df.to_excel(excel_writer, sheet_name="demographic")

        output_df.to_excel(excel_writer, sheet_name="fixation_data")
        excel_writer.save()
path = "C:\‏‏PycharmProjects\AnxietyClassifier\\100_training_set\\Gal Training Set Final.xlsx"

#path1 = "..\\test_data\machine learning data full dataset first 1.xlsx"
#path2 = "..\\test_data\machine learning data full dataset first 2.xlsx"
#path3 = "..\\test_data\machine learning data full dataset first 3.xlsx"
el = EyeLinkData(path, block = 13)
el.transform_data()